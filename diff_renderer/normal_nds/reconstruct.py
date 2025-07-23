import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import os
from .nds.core import (
    Mesh, Renderer, View, Camera
)
from .nds.losses import (
    mask_loss, normal_consistency_loss, laplacian_loss, shading_loss, normal_map_loss, side_loss,
    offset_map_loss, mesh_smoothness_loss, mesh_face_loss, normal_consistency_loss_l2, color_loss, chamfer_loss,
    lbs_regularizer
)
from .nds.modules import (
    SpaceNormalization, ViewSampler
)
from .nds.utils import AABB, write_mesh, mesh2open3d
# import pymeshlab
import matplotlib.pyplot as plt
import trimesh
from diff_renderer.normal_nds.utils import batch_rigid_transform, diff_forward_skinning, pose_extraction, subdivide_concat, export_obj
from sklearn.neighbors import KDTree
import smplx
from pyremesh import remesh_botsch

# import itertools
# from pysdf import SDF


class NormalNDS(nn.Module):
    def __init__(self, args, cam_params, device='cpu'):
        super(NormalNDS, self).__init__()
        self.smpl_mesh = None
        self.initial_mesh = None
        self.up_axis = args.up_axis  # 1 corresponds to y-axis
        self.align_yaw = args.align_yaw
        self.yaw_inverse_mat = None
        self.scale = 1.0
        self.device = device
        self.replace_hands = args.replace_hands
        self.cam_params=cam_params

        # optimization
        self.start_iteration = args.start_iteration
        self.iterations = args.iterations
        self.iterations_color = args.iterations_color
        self.optim_only_visible = args.optim_only_visible
        self.upsample_interval = args.upsample_interval
        self.upsample_iterations = list(range(args.upsample_start, args.iterations, args.upsample_interval))
        self.initial_num_vertex = args.initial_num_vertex
        self.refine_num_vertex = args.refine_num_vertex
        self.lr_vertices = args.lr_vertices
        self.loss_weights = {
            "mask": args.loss.weight_mask,
            "normal": args.loss.weight_normal,
            "laplacian": args.loss.weight_laplacian,
            "shading": args.loss.weight_shading,
            "side": args.loss.weight_side,
            "color": args.loss.weight_color,
            "pose": args.loss.weight_pose,
        }
        self.visualization_frequency = args.visualization_frequency
        self.visualization_views = args.visualization_views
        self.save_frequency = args.save_frequency

        self.aabb = None
        self.space_normalization = None

        self.orthographic = (args.camera == 'orthographic')
        self.renderer = Renderer(near=self.cam_params['near'], far=self.cam_params['far'], orthographic=self.orthographic, device=self.device)

        self.views = []
        self.view_sampler_args = ViewSampler.get_parameters(args)
        self.side_views = []

        self.keep_hand = False
        self.not_hand_mask = None

        self.refine_color = args.refine_color 
        self.refine_geo = args.refine_geo

        self.subdivide = args.subdivide
        self.use_uv = args.use_uv
        self.decimation = args.decimation
        self.upsampling = args.upsampling
        
        self.body_scale = 1.0
        self.body_vmed = 0.0
        
    def set_initial_mesh(self, smpl_mesh, smpl_infos, smpl_models, posed_meshes=None, canon_mesh=None, config=None):
        # fix lbs regardless of smpl model
        # self.lbs = smpl_infos[0]['lbs']
        self.lbs = None
        
        self.tpose = config.canonfusion.tpose
        
        if config.canonfusion.use_mesh and posed_meshes is not None:
            self.posed_meshes = []
            for posed_mesh, smpl_param, smpl_model in zip(posed_meshes, smpl_infos, smpl_model):
                posed_mesh = Mesh(vertices=torch.tensor(posed_mesh.vertices).contiguous(),
                                indices=torch.tensor(posed_mesh.faces).contiguous(),
                                colors=torch.tensor(posed_mesh.visual.vertex_colors).contiguous(),
                                device=self.device)
                self.posed_meshes.append(posed_mesh)
        else:
            self.posed_meshes = None

        uv = None

        if self.subdivide and self.use_uv: # subdivide smplx uv mesh
            vertices, faces, uv = subdivide_concat(smpl_mesh, smpl_mesh.visual.uv)
            initial_mesh = trimesh.Trimesh(vertices, faces, process=False)
            initial_mesh.metadata['vertex_texture'] = uv
               
        elif self.subdivide and not self.use_uv: # subdivide smplx mesh
            vertices, faces = subdivide_concat(smpl_mesh)
            initial_mesh = trimesh.Trimesh(vertices, faces, process=False)     
            
        elif self.use_uv: # smplx uv mesh
            initial_mesh = smpl_mesh
            uv = smpl_mesh.visual.uv
            faces = smpl_mesh.faces
            
        elif config.canonfusion.use_initial_mesh and canon_mesh is not None: # initial canonical mesh decimation
            initial_mesh = mesh2open3d(canon_mesh, texture=None)  
            
        else: # smpl mesh decimation
            # load smpl with open3d for further decimation
            # initial_mesh = mesh2open3d(smpl_mesh, texture=None) 
            import pymeshlab
            ms = pymeshlab.MeshSet()
            smpl_mesh_ml = pymeshlab.Mesh(vertex_matrix=np.asarray(smpl_mesh.vertices),
                                          face_matrix=smpl_mesh.faces)

            ms.add_mesh(smpl_mesh_ml)
            ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=self.initial_num_vertex,
                                    preserveboundary=True, preservenormal=True, preservetopology=True)
            initial_mesh = ms.current_mesh()
            # decimate smpl mesh to 3000 vertices for mesh optimization
            # initial_mesh = initial_mesh.simplify_quadric_decimation(target_number_of_triangles=self.initial_num_vertex)
            faces = initial_mesh.face_matrix()

        self.initial_mesh = Mesh(vertices=torch.tensor(np.asarray(initial_mesh.vertex_matrix())).contiguous(),
                    indices=torch.tensor(np.asarray(faces)).contiguous(),
                    colors=torch.zeros(np.asarray(initial_mesh.vertex_matrix()).shape).contiguous(),
                    uvs=torch.tensor(np.asarray(uv)).contiguous() if uv is not None else torch.zeros(np.asarray(uv).shape).contiguous(),
                    device=self.device)
        self.initial_mesh.compute_connectivity()

    def set_views_from_normal_maps(self, body_normal_maps: dict, ):
        """
            body_normal_maps: dict with view angles as keys and body normal maps as values
            face_normal_maps: dict with view angles as keys and face normal maps as values
        """
        # generate view objects
        print(f'Found {len(body_normal_maps.keys())} views for each frame, total {len(body_normal_maps[0])} frame(s)')
        views = []
        for view_angle in body_normal_maps.keys():
            idx = 0
            for i, [body_normal_map_cam, body_color_map_cam, view_pose, smpl_info, smpl_model] in enumerate(
                    body_normal_maps[view_angle]):
                body_normal_map_world = View.to_world(body_normal_map_cam, view_angle)
                # body_normal_map = 0.5 * (
                #             torch.clamp(body_normal_map_world.permute(1, 2, 0), -1, 1) + 1)  # [C, H, W] => [H, W, C]
                body_normal_map = torch.clamp(body_normal_map_cam.permute(1, 2, 0), -1, 1) # [C, H, W] => [H, W, C]
                # body_color_map_world = View.to_world(body_color_map_cam, view_angle)
                body_color_map = torch.clamp(body_color_map_cam.permute(1, 2, 0), 0, 1)  # [C, H, W] => [H, W, C]

                camera = Camera.camera_with_angle(scale=self.body_scale, center=self.body_vmed,
                                                  view_angle=view_angle, orthographic=self.orthographic, cam_params=self.cam_params,
                                                  device=self.device)
                views.append(View(normal=body_normal_map[:, :, :3], mask=torch.clamp(body_normal_map[:, :, 3:], 0, 1),
                                  color=body_color_map[:, :, :3],
                                  camera=camera, view_angle=view_angle, view_pose=(view_pose, idx), # pose index for further pose optimization
                                  orthographic=self.orthographic, smpl_param=smpl_info, smpl_model=smpl_model,
                                  device=self.device))
                idx += 1
        self.views = views

    def image_to_render(self, input_frames, input_masks, input_normals=None, config=None):
        color_multi_set, normal_multi_set = [], []
        
        for (colors, masks, normals) in zip(input_frames, input_masks, input_normals):
            normal_maps = []
            color_maps = []
            for (color, mask, normal) in zip(colors, masks, normals):
                color = torch.FloatTensor(color).to(self.device)
                mask = torch.FloatTensor(mask).to(self.device)
                normal = torch.FloatTensor(normal).to(self.device)

                normal = (normal / 255.0) * mask - (1 - mask)  # set normals of masked region to [-1, -1, -1]
                normal_with_alpha = torch.concat([normal, 2 * mask - 1], dim=-1)                    
                normal_maps.append(normal_with_alpha.permute(2, 0, 1))
                    
                color = color / 255.0
                color = color * mask
                color_with_alpha = torch.concat([color, 2 * mask - 1], dim=-1)                    
                color_maps.append(color_with_alpha.permute(2, 0, 1))
                
            normal_multi_set.append(normal_maps)
            color_multi_set.append(color_maps)
        return normal_multi_set, color_multi_set

    def render_mesh(self, smpl_infos, config):
        color_multi_set, normal_multi_set = [], []

        for i, pose_mesh in enumerate(self.posed_meshes):
            normal_maps = []
            color_maps = []
            for view_angle in config.canonfusion.view_angle:
                # for multi view setting
                normal_map, color_map = self.render_target_view(pose_mesh, view_angle, smpl_infos[i], return_vis_mask=False,
                                                                return_color=True)
                normal_maps.append(normal_map.permute(2, 0, 1))  # [H, W, C] => [C, H, W]
                color_maps.append(color_map.permute(2, 0, 1))
            normal_multi_set.append(normal_maps)
            color_multi_set.append(color_maps)
        return normal_multi_set, color_multi_set

    def render_target_view(self, mesh, target_view_angle, smpl_info, return_vis_mask=True, return_color=False, closeup=False,
                           h=512, w=512):
        mesh_copy = Mesh(vertices=mesh.vertices.clone(), indices=mesh.indices, colors=mesh.colors, device=self.device)
        mesh_copy.vertices *= self.scale

        # # Apply the normalizing affine transform, which maps the bounding box to
        # # a 2-cube centered at (0, 0, 0), to the views, the mesh, and the bounding box
        mesh_copy = self.space_normalization.normalize_mesh(mesh_copy)

        # generate next view image with visible vertices
        target_view_camera = Camera.camera_with_angle(scale=self.scale, center=(0, 0, 0),
                                                      view_angle=target_view_angle,
                                                      orthographic=self.orthographic, device=self.device)
        target_view_R = target_view_camera.R
        target_view_camera.normalize_camera(self.space_normalization, device=self.device)
        resolution = (h, w)
        target_gbuffer = self.renderer.render_with_camera(target_view_camera, mesh_copy, resolution,
                                                          vis_mask=None, with_antialiasing=True)
        mask = target_gbuffer["mask"]
        position = target_gbuffer["position"]
        target_view_normal = target_gbuffer["normal"]
        if return_color:
            target_view_color = target_gbuffer["color"] / 255.0
            target_view_color = target_view_color * mask
        else:
            target_view_color = None

        target_view_normal = target_view_normal * mask - (1 - mask)  # set normals of masked region to [-1, -1, -1]
        target_view_normal_with_alpha = torch.concat([target_view_normal, 2 * mask - 1], dim=-1)

        if return_vis_mask:
            # find faces visible from image_views
            vis_mask = self.renderer.get_face_visibility(self.views, mesh_copy)

            pix_to_face = target_gbuffer["pix_to_face"].long()
            out_of_fov = (pix_to_face == 0)
            target_view_mask = vis_mask[pix_to_face - 1].float()
            target_view_mask[out_of_fov] = 0

            return target_view_normal_with_alpha, target_view_mask

        return target_view_normal_with_alpha, target_view_color

    def get_face_length(self, vts, faces):
        # check faces.
        areas = []
        for k in range(faces.shape[0]):
            x, y, z = faces[k]
            a = sum((vts[x, :] - vts[y, :]) ** 2) ** 2
            b = sum((vts[y, :] - vts[z, :]) ** 2) ** 2
            c = sum((vts[x, :] - vts[z, :]) ** 2) ** 2
            s = a + b + c
            if s < 1:
                areas.append(True)
            else:
                areas.append(False)
        return areas


    def forward(self, output_dir):
        num_views = len(self.views)
        num_frames = self.views[0].view_pose[1] + 1 # for pose index
        loss_weights = self.loss_weights.copy()
        lr_vertices = self.lr_vertices

        if self.upsampling:
            for i in range(len(self.upsample_iterations)):
                loss_weights['laplacian'] *= 4
                loss_weights['normal'] *= 4
                lr_vertices *= 0.25
                loss_weights['side'] *= 0.25
        if (self.visualization_frequency > 0):
            save_normal_path = output_dir / 'normals'
            save_normal_path.mkdir(exist_ok=True)
        if (self.visualization_frequency > 0):
            save_color_path = output_dir / 'colors'
            save_color_path.mkdir(exist_ok=True)
        if (self.save_frequency > 0):
            save_mesh_path = output_dir / 'meshes'
            save_mesh_path.mkdir(exist_ok=True)

        # Configure the view sampler
        view_sampler = ViewSampler(views=self.views, **self.view_sampler_args)

        # Create the optimizer for the vertex positions
        # (we optimize offsets from the initial vertex position)
        vertex_offsets = nn.Parameter(torch.zeros_like(self.initial_mesh.vertices)) # N X 3
        empty_pose = torch.empty(num_frames, 55, 3).to(self.device) # #frames X 55 X 3 (multi frame setting -> N pose offsets)
        pose_offsets = nn.Parameter(torch.zeros_like(empty_pose)) 
        vertex_colors = nn.Parameter(torch.zeros_like(self.initial_mesh.vertices))
        if not self.optim_only_visible:
            # optimizer_vertices = torch.optim.Adam([vertex_offsets, pose_offsets], lr=lr_vertices)
            optimizer_vertices = torch.optim.Adam([vertex_offsets], lr=lr_vertices)

        # Initialize the loss weights and losses
        losses = {k: torch.tensor(0.0, device=self.device) for k in loss_weights}

        if self.tpose:
            initial_mesh = self.initial_mesh
            kdtree = KDTree(self.smpl_mesh.vertices.detach().cpu().numpy(), leaf_size=30, metric='euclidean')
            kd_idx = kdtree.query(initial_mesh.vertices.clone().detach().cpu().numpy(), k=1, return_distance=False)
            lbs = torch.Tensor(self.lbs[kd_idx.squeeze(), :]).to(self.device)
            homogen_coord = torch.ones([1, initial_mesh.vertices.shape[0], 1], dtype=torch.float32).to(self.device)
            self.body_vmed = torch.Tensor(self.body_vmed).to(self.device)
            self.body_scale = torch.Tensor([self.body_scale]).to(self.device)
        else:
            initial_mesh = self.initial_mesh
            
        # generate pose mask to update only body pose ([1:22])
        # pose_mask = torch.zeros((55, 3)).to(self.device)
        # pose_mask[1:22] = torch.ones((21, 3)).to(self.device)
        # self.pose_mask = pose_mask[:, 0] > 0
            
        progress_bar = tqdm(range(self.start_iteration, self.iterations_color))
        for iteration in progress_bar:
            progress_bar.set_description(desc=f'Iteration {iteration}')
            update_color = False
            loss_weights['color'] = 0.0

            if iteration in self.upsample_iterations and self.decimation:
                # Upsample the mesh by remeshing the surface with half the average edge length
                # mesh = trimesh.Trimesh(mesh.vertices.detach().cpu().numpy(), mesh.indices.detach().cpu().numpy())
                # confidence = self.get_face_length(mesh.vertices, mesh.faces)
                # mesh.update_faces(confidence)
                # mesh.remove_degenerate_faces()
                # mesh = Mesh(mesh.vertices, mesh.faces, np.ascontiguousarray(np.zeros((mesh.vertices.shape))), np.ascontiguousarray(np.zeros((mesh.vertices.shape[0], 2))), device=self.device)

                # NDS default : pyremesh
                e0, e1 = mesh.edges.unbind(1)
                average_edge_length = torch.linalg.norm(mesh.vertices[e0] - mesh.vertices[e1], dim=-1).mean()

                v_upsampled, f_upsampled, = remesh_botsch(mesh.vertices.cpu().detach().numpy().astype(np.float64),
                                                          mesh.indices.cpu().numpy().astype(np.int32),
                                                          h=float(average_edge_length / 2))

                v_upsampled = np.ascontiguousarray(v_upsampled)
                f_upsampled = np.ascontiguousarray(f_upsampled)

                # TODO: how to maintain color value on mesh upsampling?
                colors = np.ascontiguousarray(np.zeros((v_upsampled.shape)))

                initial_mesh = Mesh(v_upsampled, f_upsampled, colors, device=self.device)
                initial_mesh.compute_connectivity()

                if self.tpose:
                    initial_mesh = Mesh(torch.Tensor(v_upsampled), torch.Tensor(f_upsampled), torch.Tensor(colors),
                                           device=self.device)
                    initial_mesh.compute_connectivity()
                    kdtree = KDTree(self.smpl_mesh.vertices.clone().detach().cpu().numpy(), leaf_size=30,
                                    metric='euclidean')
                    kd_idx = kdtree.query(initial_mesh.vertices.clone().detach().cpu().numpy(), k=1,
                                          return_distance=False)
                    lbs = torch.Tensor(self.lbs[kd_idx.squeeze(), :]).to(self.device)
                    homogen_coord = torch.ones([1, initial_mesh.vertices.shape[0], 1], dtype=torch.float32).to(
                        self.device)

                # Adjust weights and step size
                loss_weights['laplacian'] *= 4
                loss_weights['normal'] *= 4
                lr_vertices *= 0.25
                loss_weights['side'] *= 0.25
                # Create a new optimizer for the vertex offsets
                vertex_offsets = nn.Parameter(torch.zeros_like(initial_mesh.vertices))
                vertex_colors = nn.Parameter(torch.zeros_like(initial_mesh.vertices))

                if not self.optim_only_visible:
                    optimizer_vertices = torch.optim.Adam([vertex_offsets], lr=lr_vertices)
                    optimizer_colors = torch.optim.Adam([vertex_colors], lr=self.lr_vertices)

                if (self.save_frequency > 0) and ((iteration == 0) or ((iteration + 1) % self.save_frequency == 0)):
                    with torch.no_grad():
                        mesh_for_writing = initial_mesh.detach().to('cpu')
                        if self.yaw_inverse_mat is not None:
                            mesh_for_writing.vertices = mesh_for_writing.vertices @ self.yaw_inverse_mat
                        mesh_for_writing.vertices /= self.scale
                        write_mesh(save_mesh_path / f"{num_views}views_{iteration:04d}_upsample.obj", mesh_for_writing)
            elif iteration > self.upsample_iterations[-1]:
                update_color = True
                loss_weights['color'] = self.loss_weights['color']

            # Sample a view subset
            views_subset = view_sampler(self.views)
            # transl_smplx = views_subset[0].smpl_param['transl'] # views_subset: only one view list => views_subset[0].xxx..
            # scale_smplx = views_subset[0].smpl_param['scale']
            # smpl_model = views_subset[0].smpl_model
            # joints = views_subset[0].smpl_param['joints']
            # smpl_verts = views_subset[0].smpl_param['vertices']

            transl_smplx = None
            scale_smplx = None
            smpl_model = None
            joints = None
            smpl_verts = None
            
            # calculate A matrix for deformed pose
            # 

            # Deform the initial mesh
            mesh = initial_mesh.with_vertices(initial_mesh.vertices + vertex_offsets)            
            if update_color:
                mesh = mesh.with_colors(vertex_colors)
            if self.tpose:
                mesh = mesh.detach().to('cpu')
                pose_idx = views_subset[0].view_pose[1]
                pose = views_subset[0].view_pose[0].to(self.device) + pose_offsets[pose_idx] * self.pose_mask[:, None]  # pose -> initial full_pose (55x3) + pose offsets
                Deform_Mat = torch.Tensor(pose_extraction(pose.clone(), views_subset[0].smpl_param, smpl_model, self.device)).to(self.device)  # (# poses) -> A matrix for forward skinning
                deformed_vert = diff_forward_skinning(mesh.vertices.unsqueeze(0).float().to(self.device),
                                                                A=Deform_Mat, lbs=lbs.unsqueeze(0),
                                                                homogen_coord=homogen_coord, device='cuda')  # A = forward skinning matrix
                # scaling
                deformed_vert = ((deformed_vert.squeeze() + transl_smplx) * scale_smplx).detach().cpu()
                deformed_mesh = mesh.with_vertices(deformed_vert)

            # find vertices visible from image views
            if self.optim_only_visible:
                if self.tpose:
                    deformed_mesh = initial_mesh.with_vertices(deformed_mesh.vertices.to(self.device))
                    vis_mask = self.renderer.get_vert_visibility(views_subset, deformed_mesh)
                else:
                    vis_mask = self.renderer.get_vert_visibility(views_subset, mesh)
                if self.not_hand_mask is not None and iteration < self.upsample_interval:
                    vis_mask *= torch.tensor(self.not_hand_mask, device=self.device)
                target_vertices = nn.Parameter(vertex_offsets[vis_mask].clone())
                detach_vertices = vertex_offsets[~vis_mask].detach()

                # target_pose_offsets = nn.Parameter(pose_offsets[pose_idx][self.pose_mask].clone())
                # detach_pose_offsets = pose_offsets[pose_idx][~self.pose_mask].detach()
                # optimizer_vertices = torch.optim.Adam([target_vertices, target_pose_offsets], lr=lr_vertices)
                optimizer_vertices = torch.optim.Adam([target_vertices], lr=lr_vertices)

                if self.tpose:
                    mesh_vertices = initial_mesh.vertices.detach().clone()
                    mesh_vertices[vis_mask] += target_vertices
                    mesh_vertices[~vis_mask] += detach_vertices
                    
                    mesh_colors = initial_mesh.colors.detach().clone()
                    mesh_colors[vis_mask] += target_vertice_colors
                    mesh_colors[~vis_mask] += detach_vertice_colors
                    
                    mesh = initial_mesh.with_vertices(mesh_vertices)
                    mesh = mesh.with_colors(mesh_colors)
                    
                    pose = views_subset[0].view_pose[0].to(self.device)
                    pose[self.pose_mask] += target_pose_offsets

                    # posed_smpl = smpl_model(body_pose=pose[1:22].reshape([1, -1]), return_full_pose=True,
                    #                         return_verts=True)
                    # posed_smpl_vert = posed_smpl.vertices.squeeze()

                    rot_mats = smplx.lbs.batch_rodrigues(pose.clone().view(-1, 3)).view([1, -1, 3, 3])
                    joints_warped, A_pose = batch_rigid_transform(rot_mats.clone(),
                                                                              joints[:, :55, :].detach().clone(),
                                                                              smpl_model.parents.detach().clone(),
                                                                              inverse=False, dtype=torch.float32)
                    #
                    # A_pose = torch.Tensor(
                    #     self.pose_extraction(pose.clone(), self.smpl_info, self.smpl_model, self.device)).to(
                    #     self.device)

                    # deform canonical mesh to posed mesh
                    posed_vert = diff_forward_skinning(mesh.vertices.unsqueeze(0).float(), A=A_pose,
                                                                    lbs=lbs.unsqueeze(0), homogen_coord=homogen_coord,
                                                                    device='cuda')  # A = pose
                    # scaling
                    mesh_vertices = (posed_vert.squeeze() + transl_smplx) * scale_smplx

                else:
                    mesh_vertices = initial_mesh.vertices.detach().clone()
                    mesh_vertices[vis_mask] += target_vertices
                    mesh_vertices[~vis_mask] += detach_vertices
                    mesh = initial_mesh.with_vertices(mesh_vertices)
                    
                    if update_color:
                        target_vertice_colors = nn.Parameter(vertex_colors[vis_mask].clone())
                        detach_vertice_colors = vertex_colors[~vis_mask].detach()
                        optimizer_colors = torch.optim.Adam([target_vertice_colors], lr=self.lr_vertices * 10)
                        mesh_colors = initial_mesh.colors.detach().clone()
                        mesh_colors[vis_mask] += target_vertice_colors
                        mesh_colors[~vis_mask] += detach_vertice_colors
                        
                        mesh = mesh.with_colors(mesh_colors)

            # Render the mesh from the views
            # Perform antialiasing here because we cannot antialias after shading if we only shade a some of the pixels
            if self.tpose:
                gbuffers = self.renderer.render(views_subset, deformed_mesh, channels=['mask', 'normal', 'color'],
                                                with_antialiasing=True)
            else:
                gbuffers = self.renderer.render(views_subset, mesh, channels=['mask', 'normal', 'color'], with_antialiasing=True)

            # Combine losses and weights
            if loss_weights['mask'] > 0:
                losses['mask'] = mask_loss(views_subset, gbuffers)
                # plt.imshow(views_subset[0].mask.detach().cpu().numpy())
                # plt.imshow(gbuffers[0]['mask'].detach().cpu().numpy())


            if loss_weights['color'] > 0:
                losses['color'] = color_loss(views_subset, gbuffers)

                    
            if loss_weights['normal'] > 0:
                if self.tpose:
                    losses['normal'] = normal_consistency_loss(deformed_mesh)
                else:
                    losses['normal'] = normal_consistency_loss(mesh)
            if loss_weights['laplacian'] > 0:
                if self.tpose:
                    losses['laplacian'] = laplacian_loss(deformed_mesh)
                else:
                    losses['laplacian'] = laplacian_loss(mesh)
            if loss_weights['shading'] > 0:
                losses['shading'] = normal_map_loss(views_subset, gbuffers)

            # if loss_weights['pose'] > 0:
            #     losses['pose'] = chamfer_loss(mesh_vertices.unsqueeze(0), torch.Tensor(smpl_verts).to(self.device).unsqueeze(0))

            loss = torch.tensor(0., device=self.device)
            for k, v in losses.items():
                loss += v * loss_weights[k]


            # Optimize
            optimizer_vertices.zero_grad()
            if update_color:
                optimizer_colors.zero_grad()
            loss.backward()  # retain_graph=True
            optimizer_vertices.step()
            if update_color:
                optimizer_colors.step()

            if self.optim_only_visible:
                vertex_offsets = torch.zeros_like(vertex_offsets)
                vertex_offsets[vis_mask] = target_vertices
                vertex_offsets[~vis_mask] = detach_vertices
                
                if update_color:
                    vertex_colors = torch.zeros_like(vertex_colors)
                    vertex_colors[vis_mask] = torch.clamp(target_vertice_colors, 0, 1)
                    vertex_colors[~vis_mask] = torch.clamp(detach_vertice_colors, 0, 1)
                
                # pose_offsets = pose_offsets.clone()
                # pose_offsets[pose_idx][self.pose_mask] = target_pose_offsets

            progress_bar.set_postfix({'loss': loss.detach().cpu()})

            # Visualizations
            if (self.visualization_frequency > 0) and (
                    (iteration == 0) or ((iteration + 1) % self.visualization_frequency == 0)):
                import matplotlib.pyplot as plt
                with torch.no_grad():
                    use_fixed_views = len(self.visualization_views) > 0
                    view_indices = self.visualization_views if use_fixed_views else [
                        np.random.choice(list(range(len(views_subset))))]
                    
                    mesh = initial_mesh.with_vertices(initial_mesh.vertices + vertex_offsets)      
                    mesh = mesh.with_colors(vertex_colors)      
                    for vi in view_indices:
                        debug_view = self.views[vi] if use_fixed_views else views_subset[vi]
                        debug_gbuffer = \
                        self.renderer.render([debug_view], mesh, channels=['mask', 'position', 'normal', 'color'],
                                             with_antialiasing=True)[0]
                        position = debug_gbuffer["position"]
                        normal = debug_gbuffer["normal"]
                        color = debug_gbuffer["color"]

                        normal_image = (0.5 * (normal + 1)) * debug_gbuffer["mask"] + (
                                    1 - debug_gbuffer["mask"])  # global normal
                    
                        color_image = (color) * debug_gbuffer["mask"] + (
                                    1 - debug_gbuffer["mask"])  # global normal
                        plt.imsave(
                            save_normal_path / f'{num_views}views_{(iteration + 1):04d}_{debug_view.view_angle}.png',
                            normal_image.cpu().numpy())
                        plt.imsave(
                            save_color_path / f'{num_views}views_{(iteration + 1):04d}_{debug_view.view_angle}.png',
                            color_image.cpu().numpy())

            if (self.save_frequency > 0) and ((iteration == 0) or ((iteration + 1) % self.save_frequency == 0)):
                with torch.no_grad():
                    mesh = initial_mesh.with_vertices(initial_mesh.vertices + vertex_offsets)
                    mesh = mesh.with_colors(vertex_colors)

                    mesh_for_writing = mesh.detach().to('cpu')
                    if self.yaw_inverse_mat is not None:
                        mesh_for_writing.vertices = mesh_for_writing.vertices @ self.yaw_inverse_mat
                    mesh_for_writing.vertices /= self.scale
                    write_mesh(save_mesh_path / f"{num_views}views_{(iteration + 1):04d}.obj", mesh_for_writing)

        nds_result = mesh.detach().to('cpu')
        if self.yaw_inverse_mat is not None:
            nds_result.vertices = nds_result.vertices @ self.yaw_inverse_mat
        nds_result.vertices /= self.scale
        if self.tpose:
            return nds_result, self.lbs
        else:
            return nds_result
        # if self.refine_color:
        #     if (self.save_frequency > 0):
        #         save_color_path = output_dir / 'colors'
        #         save_color_path.mkdir(exist_ok=True)
        #     if (self.save_frequency > 0):
        #         save_mesh_path = output_dir / 'meshes'
        #         save_mesh_path.mkdir(exist_ok=True)

        #     color_mesh = Mesh(vertices=mesh.vertices.detach().cpu().numpy(),
        #                       indices=mesh.indices.detach().cpu().numpy(),
        #                       colors=np.ones((mesh.vertices.shape)),
        #                       device=self.device)

        #     # Configure the view sampler
        #     view_sampler = ViewSampler(views=self.views, **self.view_sampler_args)

        #     # Create the optimizer for the vertex positions
        #     # (we optimize offsets from the initial vertex position)
        #     vertex_colors = nn.Parameter(torch.ones_like(color_mesh.colors)*0.5)

        #     if not self.optim_only_visible:
        #         optimizer_colors = torch.optim.Adam([vertex_colors], lr=self.lr_vertices)

        #     # Initialize the loss weights and losses
        #     losses = {k: torch.tensor(0.0, device=self.device) for k in loss_weights}

        #     progress_bar = tqdm(range(iteration, 4000))  ## revise 4000 ... -> hyperparameter
        #     for iteration in progress_bar:
        #         progress_bar.set_description(desc=f'Iteration {iteration}')

        #         # Sample a view subset
        #         views_subset = view_sampler(self.views)

        #         transl_smplx = torch.tensor(views_subset[0].smpl_param['transl']).to(self.device)
        #         scale_smplx = torch.tensor(views_subset[0].smpl_param['scale']).to(self.device)
        #         smpl_model = views_subset[0].smpl_model
        #         joints = views_subset[0].smpl_param['joints']

        #         # calculate A matrix for deformed pose
        #         pose_idx = views_subset[0].view_pose[1]
        #         pose = views_subset[0].view_pose[0].to(self.device) + pose_offsets[pose_idx] * self.pose_mask[:,
        #                                                                                        None]  # pose -> initial full_pose (55x3) + pose offsets
        #         A_pose = torch.Tensor(
        #             pose_extraction(pose.clone(), views_subset[0].smpl_param, smpl_model, self.device)).to(
        #             self.device)

        #         mesh = color_mesh.with_colors(vertex_colors)

        #         if self.tpose:
        #             mesh = mesh.detach().to('cpu')

        #             # deform canonical mesh to posed mesh
        #             posed_vert = diff_forward_skinning(mesh.vertices.unsqueeze(0).float().to(self.device),
        #                                                             A=A_pose, lbs=lbs.unsqueeze(0),
        #                                                             homogen_coord=homogen_coord,
        #                                                             device='cuda')  # A=pose
        #             # scaling
        #             posed_vert = ((posed_vert.squeeze() + transl_smplx) * scale_smplx).detach().cpu()

        #             posed_mesh = mesh.with_vertices(posed_vert)

        #         # find vertices visible from image views
        #         if self.optim_only_visible:
        #             if self.tpose:
        #                 posed_mesh = color_mesh.with_vertices(posed_mesh.vertices.to(self.device))
        #                 vis_mask = self.renderer.get_vert_visibility(views_subset, posed_mesh)
        #             else:
        #                 vis_mask = self.renderer.get_vert_visibility(views_subset, mesh)
        #             if self.not_hand_mask is not None and iteration < self.upsample_interval:
        #                 vis_mask *= torch.tensor(self.not_hand_mask, device=self.device)
        #             target_vertice_colors = nn.Parameter(vertex_colors[vis_mask].clone())
        #             detach_vertice_colors = vertex_colors[~vis_mask].detach()
        #             optimizer_colors = torch.optim.Adam([target_vertice_colors], lr=self.lr_vertices * 10)

        #             if self.tpose:
        #                 vertex_colors = torch.zeros_like(vertex_colors)
        #                 vertex_colors[vis_mask] = target_vertice_colors
        #                 vertex_colors[~vis_mask] = detach_vertice_colors

        #                 mesh = color_mesh.with_colors(vertex_colors)

        #                 # deform canonical mesh to posed mesh
        #                 posed_vert = diff_forward_skinning(canon_vert.unsqueeze(0).float(), A=A_pose,
        #                                                                 lbs=lbs.unsqueeze(0),
        #                                                                 homogen_coord=homogen_coord,
        #                                                                 device='cuda')  # A = pose
        #                 # scaling
        #                 posed_vert = (posed_vert.squeeze() + transl_smplx) * scale_smplx

        #                 posed_mesh = mesh.with_vertices(posed_vert)

        #         # Render the mesh from the views
        #         # Perform antialiasing here because we cannot antialias after shading if we only shade a some of the pixels
        #         if self.canonical:
        #             gbuffers = self.renderer.render(views_subset, posed_mesh, channels=['color'],
        #                                             with_antialiasing=True)
        #         else:
        #             gbuffers = self.renderer.render(views_subset, mesh, channels=['color'],
        #                                             with_antialiasing=True)

        #         # Combine losses and weights
        #         if loss_weights['color'] > 0:
        #             losses['color'] = color_loss(views_subset, gbuffers)

        #         loss = torch.tensor(0., device=self.device)
        #         for k, v in losses.items():
        #             loss += v * loss_weights[k]

        #         # Optimize
        #         optimizer_colors.zero_grad()
        #         loss.backward()  # retain_graph=True
        #         optimizer_colors.step()

        #         if self.optim_only_visible:
        #             vertex_colors = torch.zeros_like(vertex_colors)
        #             vertex_colors[vis_mask] = torch.clip(target_vertice_colors, 0, 1)
        #             vertex_colors[~vis_mask] = torch.clip(detach_vertice_colors, 0, 1)

        #         progress_bar.set_postfix({'loss': loss.detach().cpu()})

        #         # Visualizations
        #         if (self.visualization_frequency > 0) and (
        #                 (iteration == 0) or ((iteration + 1) % self.visualization_frequency == 0)):
        #             import matplotlib.pyplot as plt
        #             with torch.no_grad():
        #                 use_fixed_views = len(self.visualization_views) > 0
        #                 view_indices = self.visualization_views if use_fixed_views else [
        #                     np.random.choice(list(range(len(views_subset))))]
        #                 if self.canonical:
        #                     canon_color_mesh = color_mesh.with_colors(vertex_colors)
        #                 for vi in view_indices:
        #                     debug_view = self.views[vi] if use_fixed_views else views_subset[vi]
        #                     debug_gbuffer = \
        #                         self.renderer.render([debug_view], canon_color_mesh, channels=['color', 'mask'],
        #                                              with_antialiasing=True)[0]
        #                     color = debug_gbuffer["color"]

        #                     # Save a color map in camera space
        #                     # color_image = color * debug_gbuffer["mask"]
        #                     plt.imsave(
        #                         save_color_path / f'{num_views}views_{(iteration + 1):04d}_{debug_view.view_angle}.png',
        #                         torch.clamp(color, 0, 1).cpu().numpy())

        #         if (self.save_frequency > 0) and ((iteration == 0) or ((iteration + 1) % self.save_frequency == 0)):
        #             with torch.no_grad():
        #                 if self.canonical:
        #                     canon_color_mesh = color_mesh.with_colors(torch.clip(vertex_colors, 0, 1))
        #                 mesh_for_writing = self.space_normalization.denormalize_mesh(
        #                     canon_color_mesh.detach().to('cpu'))
        #                 if self.yaw_inverse_mat is not None:
        #                     mesh_for_writing.vertices = mesh_for_writing.vertices @ self.yaw_inverse_mat
        #                 mesh_for_writing.vertices /= self.scale
        #                 write_mesh(save_mesh_path / f"{num_views}views_{(iteration + 1):04d}.obj", mesh_for_writing)

        #     nds_result = self.space_normalization.denormalize_mesh(canon_color_mesh.detach().to('cpu'))
        #     if self.yaw_inverse_mat is not None:
        #         nds_result.vertices = nds_result.vertices @ self.yaw_inverse_mat
        #     nds_result.vertices /= self.scale

        #     nds_result.vertices /= self.body_scale.detach().cpu()
        #     nds_result.vertices += self.body_vmed.detach().cpu()




        # return nds_result, lbs