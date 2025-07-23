from __future__ import annotations
import torch.utils.data
import trimesh.remesh
from sklearn.neighbors import KDTree
from tqdm import tqdm
#from utils.loader_utils import *
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    point_mesh_distance,
    point_mesh_edge_distance,
    point_mesh_face_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.renderer import (
    TexturesVertex
)
import torch.nn as nn
from . import render_utils
import pdb
import smplx
import copy

def mesh_smoothness_custom(meshes, vertex):
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )
    l1_loss = nn.L1Loss()
    faces_packed = meshes.faces_packed()
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    v0 = verts_packed[faces_packed[:, 0]]
    v1 = verts_packed[faces_packed[:, 1]]
    v2 = verts_packed[faces_packed[:, 2]]
    v0_ori = vertex[faces_packed[:, 0]]
    v1_ori = vertex[faces_packed[:, 1]]
    v2_ori = vertex[faces_packed[:, 2]]
    e0 = (v0 - v0_ori).norm(dim=1, p=2)
    e1 = (v1 - v1_ori).norm(dim=1, p=2)
    e2 = (v2 - v2_ori).norm(dim=1, p=2)

    loss = l1_loss(e0, e1) + l1_loss(e1, e2) + l1_loss(e2, e0)
    return loss.sum()


def mesh_edge_loss_custom(meshes):
    """
    Computes mesh edge length regularization loss averaged across all meshes
    in a batch. Each mesh contributes equally to the final loss, regardless of
    the number of edges per mesh in the batch by weighting each mesh with the
    inverse number of edges. For example, if mesh 3 (out of N) has only E=4
    edges, then the loss for each edge in mesh 3 should be multiplied by 1/E to
    contribute to the final loss.

    Args:
        meshes: Meshes object with a batch of meshes.
        target_length: Resting value for the edge length.

    Returns:
        loss: Average loss across the batch. Returns 0 if meshes contains
        no meshes or all empty meshes.
    """
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )
    l1_loss = nn.L1Loss()
    faces_packed = meshes.faces_packed()
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    v0 = verts_packed[faces_packed[:, 0]]
    v1 = verts_packed[faces_packed[:, 1]]
    v2 = verts_packed[faces_packed[:, 2]]

    e0 = (v0 - v1).norm(dim=1, p=2)
    e1 = (v1 - v2).norm(dim=1, p=2)
    e2 = (v2 - v0).norm(dim=1, p=2)
    avg_len = torch.mean(e0 + e1 + e2)/3
    loss = l1_loss(e0, e1) + l1_loss(e1, e2) + l1_loss(e2, e0) \
           + l1_loss(e0, avg_len) + l1_loss(e1, avg_len)+ l1_loss(e2, avg_len)

    return loss.sum(), avg_len


class Mesh2PointOptimizer(nn.Module):
    """Implementation of single-stage mesh deformation."""
    def __init__(self, device='cuda:0'):
        super(Mesh2PointOptimizer, self).__init__()
        self.device = device

    def forward(self, src_vts, src_faces, trg_vts, trg_color, return_verts=False, exclude_idx=None):
        src_mesh = Meshes(verts=[src_vts],
                          faces=[torch.Tensor(src_faces)]).to(self.device)

        trg_pcd = Pointclouds(points=[trg_vts],
                              features=[trg_color]).to(self.device)

        deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=self.device, requires_grad=True)
        mask = torch.ones_like(deform_verts).to(self.device)
        if exclude_idx is not None and len(exclude_idx) > 0:
            mask[exclude_idx, :] = 0.0
        optimizer = torch.optim.SGD([deform_verts], lr=0.01, momentum=0.99)

        # Number of optimization steps
        iter = 500
        # Weight for the chamfer loss
        w_chamfer = 0.8
        # Weight for mesh edge loss
        w_edge = 1.0
        # Weight for mesh normal consistency
        w_normal = 0.1
        # Weight for mesh laplacian smoothing
        w_laplacian = 0.05

        new_src_mesh = src_mesh.offset_verts(deform_verts)
        # new_src_pcd = src_pcd.offset_verts(deform_verts)

        with tqdm(range(iter), position=0, ncols=100) as pbar:
            for i in pbar:
                # Initialize optimizer
                optimizer.zero_grad()

                # We sample 10k points from the surface of each mesh
                sample_src = sample_points_from_meshes(new_src_mesh, trg_vts.shape[0], return_normals=False)
                loss_chamfer, _ = chamfer_distance(sample_src, trg_pcd)

                # We compare the two sets of pointclouds by computing (a) the chamfer loss
                loss_dist1 = point_mesh_edge_distance(new_src_mesh, trg_pcd)
                loss_dist2 = point_mesh_face_distance(new_src_mesh, trg_pcd)

                # and (b) the edge length of the predicted mesh
                loss_edge = mesh_edge_loss(new_src_mesh)
                loss_edge2 = mesh_edge_loss_custom(new_src_mesh)

                # mesh normal consistency
                loss_normal = mesh_normal_consistency(new_src_mesh)

                # mesh laplacian smoothing
                loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")

                # Weighted sum of the losses
                loss = loss_chamfer + loss_dist1 + loss_dist2 \
                       + loss_edge * w_edge \
                       + loss_normal * w_normal \
                       + loss_laplacian * w_laplacian \

                # Print the losses
                pbar.set_description('total_loss = {0:.6f}'.format(loss))

                # Optimization step
                loss.backward()
                optimizer.step()

                # Deform the mesh
                new_src_mesh = src_mesh.offset_verts(deform_verts * mask)

        if return_verts:
            return new_src_mesh[0].verts_packed().detach().cpu().numpy()
        else:
            vis_mesh = trimesh.Trimesh(new_src_mesh[0].verts_packed().detach().cpu().numpy(),
                                       new_src_mesh[0].faces_packed().detach().cpu().numpy(), process=False)
            return vis_mesh

class LBSOptimizer(nn.Module):
    """Implementation of single-stage mesh deformation."""
    def __init__(self, device='cuda:0'):
        super(LBSOptimizer, self).__init__()
        self.device = device

    def forward(self, src_mesh_trimesh, init_lbs, src_params, src_transl, smpl_model, pose, trg_mesh_trimesh, lr=0.1, iter=500):
        deform_lbs = torch.full(init_lbs.shape, 0.0, device=self.device, requires_grad=True)

        optimizer = torch.optim.SGD([deform_lbs], lr=lr, momentum=0.99)

        # weights
        w = {'chamfer': 1.0, 'edge': 0.0, 'normal': 0.0, 'laplacian': 0.0, 'face': 0.0,
             'symmetry': 0.0, 'consistency': 0.0, 'penetration': 0.0, 'smooth': 0.0}

        # pre-defined variables (constants)
        init_lbs += deform_lbs
        l2_loss = nn.MSELoss()

        with tqdm(range(iter), position=0, ncols=100) as pbar:
            for _ in pbar:
                for i in range(len(pose)):
                    optimizer.zero_grad()
                    loss = 0.0

                    verts = (torch.Tensor(src_mesh_trimesh.vertices).to(self.device) - torch.Tensor(src_params[i]['centroid_real']).to(self.device))  \
                        * torch.Tensor([src_params[i]['scale_real']]).to(self.device) / torch.Tensor([src_params[i]['scale_smplx']]).to(self.device) + torch.Tensor(src_params[i]['centroid_smplx']).to(self.device)
                    pdb.set_trace()
                    verts = render_utils.deform_vertices(verts.unsqueeze(0) - torch.Tensor(src_transl['transl']).to(self.device),
                                                        smpl_model, init_lbs,
                                                        pose[i].to(self.device),
                                                        inverse=False,
                                                        return_vshape=False,
                                                        device=self.device)
                    
                    verts = (verts - torch.Tensor(src_params[i]['centroid_smplx']).to(self.device)) * torch.Tensor([src_params[i]['scale_smplx']]).to(self.device) \
                                            / torch.Tensor([src_params[i]['scale_real']]).to(self.device) + torch.Tensor(src_params[i]['centroid_real']).to(self.device)
                    pdb.set_trace()
                    tgt_verts = trg_mesh_trimesh.vertices.unsqueeze(0).to(self.device)

                    # (a) chamfer loss (basic loss)
                    loss_chamfer_dist, _ = chamfer_distance(verts, tgt_verts)
                    loss += loss_chamfer_dist * w['chamfer']

                pbar.set_description('total_loss = {0:.6f}'.format(loss))
                loss.backward()
                optimizer.step()

                # update the current lbs
                init_lbs += deform_lbs
        return init_lbs
    
class LBS2PoseOptimizer(nn.Module):
    """Implementation of single-stage mesh deformation."""
    def __init__(self, smpl_path, device='cuda:0'):
        super(LBS2PoseOptimizer, self).__init__()
        self.device = device

        self.smpl_model = smplx.create(model_path=smpl_path,
                                       model_type='smplx',
                                       gender='male',
                                       num_betas=10, ext='npz',
                                       use_face_contour=True,
                                       flat_hand_mean=True,
                                       use_pca=False,
                                       ).to(device)

    def forward(self, src_mesh_trimesh, smpl_params, smpl_model=None, use_gt=True, lr=0.1, iter=100):
        # SMPL SPACE!
        src_mesh = Meshes(verts=[torch.Tensor(src_mesh_trimesh.vertices)],
                          faces=[torch.Tensor(src_mesh_trimesh.faces)]).to(self.device)

        # verts = torch.Tensor(src_mesh_trimesh.vertices)
        # verts = (verts - smpl_params['centroid_real']) * smpl_params['scale_real'] / \
        #         smpl_params['scale_smplx'] + smpl_params['centroid_smplx']
        # verts = verts.unsqueeze(0).to(self.device)
        body_pose = torch.nn.Parameter(torch.full([1, 63], 0.0, device=self.device, requires_grad=True))

        if use_gt or smpl_model is not None:
            scale = smpl_params['scale']
            betas = smpl_model.betas
            optimizer = torch.optim.SGD([body_pose], lr=lr, momentum=0.99)
        else:
            scale = torch.full([1], smpl_params['scale'][0], device=self.device, requires_grad=True)
            betas = torch.full([1, 10], 0.0, device=self.device, requires_grad=True)
            optimizer = torch.optim.SGD([body_pose, betas, scale], lr=lr, momentum=0.99)

        #verts = torch.Tensor(src_mesh.vertices).unsqueeze(0).to(self.device)

        # Number of optimization steps
        with tqdm(range(iter), position=0, ncols=100) as pbar:
            for i in pbar:
                # Initialize optimizer
                optimizer.zero_grad()

                smpl_output = self.smpl_model(betas=betas, body_pose=body_pose)
                pcd = Pointclouds([torch.Tensor(smpl_output.vertices.squeeze() * scale.squeeze())]).to(self.device)
                #loss_chamfer, _ = chamfer_distance(smpl_output.vertices * scale, verts)
                loss_chamfer = point_mesh_face_distance(src_mesh, pcd)
                # Print the losses
                pbar.set_description('total_loss = {0:.6f}'.format(loss_chamfer))

                # Optimization step
                loss_chamfer.backward(retain_graph=True)
                optimizer.step()

        smpl_output.vertices *= scale
        new_mesh = trimesh.Trimesh(smpl_output.vertices.squeeze(0).detach().cpu().numpy(),
                                   self.smpl_model.faces)
        # scan_mesh = trimesh.Trimesh(verts.squeeze(0).detach().cpu().numpy(), src_mesh_trimesh.faces,
        #                             visual=src_mesh_trimesh.visual)
        # new_mesh.show()
        #show_meshes([new_mesh, scan_mesh])
        return new_mesh, body_pose
    
class SMPL2MeshOptimizer(nn.Module):
    """Implementation of single-stage mesh deformation."""
    def __init__(self, device='cuda:0'):
        super(SMPL2MeshOptimizer, self).__init__()
        self.device = device

    def forward(self, src_mesh_trimesh, trg_mesh_trimesh, lr=0.1, iter=500):
        src_mesh = Meshes(verts=[torch.Tensor(src_mesh_trimesh.vertices)],
                          faces=[torch.Tensor(src_mesh_trimesh.faces)]).to(self.device)
        trg_mesh = Meshes(verts=[torch.Tensor(trg_mesh_trimesh.vertices)],
                          faces=[torch.Tensor(trg_mesh_trimesh.faces)]).to(self.device)
        deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=self.device, requires_grad=True)
        optimizer = torch.optim.SGD([deform_verts], lr=lr, momentum=0.99)

        # weights
        w = {'chamfer': 1.0, 'edge': 0.1, 'normal': 0.1, 'laplacian': 0.1, 'face': 0.1,
             'symmetry': 0.1, 'consistency': 0.5, 'penetration': 5.0, 'smooth': 0.1}

        # pre-defined variables (constants)
        vertex_normal = torch.FloatTensor(src_mesh_trimesh.vertex_normals).to(self.device).requires_grad_(False)
        new_src_mesh = src_mesh.offset_verts(deform_verts)
        initial_vertex = src_mesh.verts_packed().detach().clone()
        x_flip_mask = torch.ones_like(vertex_normal).to(self.device)
        x_flip_mask[:, 0] *= -1.0
        hands_mask = torch.ones_like(vertex_normal).to(self.device)
        hands_mask[initial_vertex[:, 0] < -65.0, :] = 0.0
        hands_mask[initial_vertex[:, 0] > 65.0, :] = 0.0

        l2_loss = nn.MSELoss()

        with tqdm(range(iter), position=0, ncols=100) as pbar:
            for _ in pbar:
                optimizer.zero_grad()
                loss = 0.0

                # sampling points from meshes (src << trg works similarly with p2s distance)
                sample_src = sample_points_from_meshes(new_src_mesh, 100000)
                sample_trg = sample_points_from_meshes(trg_mesh, 200000)

                # (a) chamfer loss (basic loss)
                loss_chamfer_dist, _ = chamfer_distance(sample_src, sample_trg)
                loss += loss_chamfer_dist * w['chamfer']

                # (b) constraining the size and shape of faces and edges
                loss_face, cur_len = mesh_edge_loss_custom(new_src_mesh)
                loss_edge = mesh_edge_loss(new_src_mesh, target_length=cur_len)
                loss += loss_face * w['face']
                loss += loss_edge * w['edge']

                # (c) mesh normal consistency
                loss_normal = mesh_normal_consistency(new_src_mesh)
                loss += loss_normal * w['normal']

                # (d) mesh smoothness ( |d_x| ~ |d_x_n| )
                loss_smoothness = mesh_smoothness_custom(new_src_mesh, initial_vertex)
                loss += loss_smoothness * w['smooth']

                # (e) mesh laplacian smoothing
                loss_laplacian = mesh_laplacian_smoothing(new_src_mesh)
                loss += loss_laplacian * w['laplacian']

                # (f) keep the normal consistent with initial (smpl) normals
                loss_normal_consistency = l2_loss(new_src_mesh.verts_normals_packed(), vertex_normal)
                loss += loss_normal_consistency * w['consistency']

                # (g) keep the symmetry of canonical mesh
                loss_flip, _ = chamfer_distance(new_src_mesh.verts_packed().unsqueeze(0),
                                                new_src_mesh.verts_packed().unsqueeze(0)*x_flip_mask.unsqueeze(0))
                loss += loss_flip * w['symmetry']

                # (h) avoid penetration (encourage deformation outward skinned body)
                # @param delta_t: allowing [delta_t] amount of penetration (larger than 0 means outer of SMPL surface)
                delta_t = -0.1
                loss_penetration = torch.mean(torch.abs(torch.clamp(torch.sum(deform_verts * vertex_normal, dim=1),
                                                                    max=delta_t)))
                loss += loss_penetration * w['penetration']

                pbar.set_description('total_loss = {0:.6f}'.format(loss))
                loss.backward()
                optimizer.step()

                # update the current mesh
                new_src_mesh = src_mesh.offset_verts(deform_verts * hands_mask)

        vertices = new_src_mesh[0].verts_packed().detach().cpu().numpy()
        faces = new_src_mesh[0].faces_packed().detach().cpu().numpy()
        kdtree = KDTree(trg_mesh_trimesh.vertices, leaf_size=30, metric='euclidean')
        idx = kdtree.query(vertices, k=1, return_distance=False)
        vis_mesh = trimesh.Trimesh(vertices, faces,
                                   vertex_colors=trg_mesh_trimesh.visual.vertex_colors[idx[:, 0], :],
                                   process=False)
        vis_mesh.fix_normals()
        vis_mesh.fill_holes()
        # vis_mesh.show()
        return trimesh.smoothing.filter_laplacian(vis_mesh)


class Mesh2MeshOptimizer(nn.Module):
    """Implementation of single-stage mesh deformation."""
    def __init__(self, device='cuda:0'):
        super(Mesh2MeshOptimizer, self).__init__()
        self.device = device

    def forward(self, src_mesh_trimesh, trg_mesh_trimesh, return_verts=False, lr=0.1, iter=500):
        # src_texture = TexturesVertex(verts_features=torch.Tensor(src_mesh_trimesh.visual.vertex_colors).unsqueeze(0))
        src_mesh = Meshes(verts=[torch.Tensor(src_mesh_trimesh.vertices)],
                          faces=[torch.Tensor(src_mesh_trimesh.faces)]).to(self.device)
                            # ,textures=src_texture).to(self.device)
        # trg_texture = TexturesVertex(verts_features=torch.Tensor(trg_mesh_trimesh.visual.vertex_colors).unsqueeze(0))
        src_pcd = Pointclouds([torch.Tensor(src_mesh_trimesh.vertices)]).to(self.device)
        trg_mesh = Meshes(verts=[torch.Tensor(trg_mesh_trimesh.vertices)],
                          faces=[torch.Tensor(trg_mesh_trimesh.faces)]).to(self.device)
                          # textures=trg_texture).to(self.device)
        # deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=self.device, requires_grad=True)
        deform_verts = torch.full([src_mesh.verts_packed().shape[0], 1], 0.0, device=self.device, requires_grad=True)
        optimizer = torch.optim.SGD([deform_verts], lr=lr, momentum=0.99)

        # Number of optimization steps
        iter = iter
        # Weight for the chamfer loss
        w_chamfer = 0.8
        # Weight for mesh edge loss
        w_edge = 0.2
        # Weight for mesh normal consistency
        w_normal = 0.2
        # Weight for mesh laplacian smoothing
        w_laplacian = 0.5
        w_face = 0.1

        # initial_vts = src_mesh.verts_packed().clone().requires_grad_(True)
        vertex_normal = torch.FloatTensor(src_mesh_trimesh.vertex_normals).to(self.device).requires_grad_(False)
        new_src_mesh = src_mesh.offset_verts(vertex_normal * deform_verts)
        _, avg_len = mesh_edge_loss_custom(new_src_mesh)
        avg_len = avg_len.clone().detach()

        with tqdm(range(iter), position=0, ncols=100) as pbar:
            for i in pbar:
                # Initialize optimizer
                optimizer.zero_grad()

                # We sample 10k points from the surface of each mesh
                sample_src, sample_src_normal = sample_points_from_meshes(new_src_mesh, 100000, return_normals=True)
                sample_trg, sample_trg_normal = sample_points_from_meshes(trg_mesh, 100000, return_normals=True)

                loss_chamfer_dist, loss_chamfer_normal = chamfer_distance(sample_src,
                                                                          sample_trg,
                                                                          x_normals=sample_src_normal,
                                                                          y_normals=sample_trg_normal)
                loss_chamfer = loss_chamfer_dist * 0.8 + loss_chamfer_normal * 0.2

                # and (b) the edge length of the predicted mesh
                loss_face, cur_len = mesh_edge_loss_custom(new_src_mesh)
                loss_edge = mesh_edge_loss(new_src_mesh, target_length=avg_len)
                #
                # # mesh normal consistency
                loss_normal = mesh_normal_consistency(new_src_mesh)
                #
                # # mesh laplacian smoothing
                loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="cot")

                # loss_symmetry = sample_src - sample_src
                loss_normal2 = torch.mean(torch.abs(src_mesh.verts_normals_packed() - vertex_normal))
                #
                # Weighted sum of the losses
                loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + \
                       loss_normal2 + loss_laplacian * w_laplacian + loss_face * w_face + loss_normal * 0.1

                # Print the losses
                pbar.set_description('total_loss = {0:.6f}'.format(loss))

                # Optimization step
                loss.backward(retain_graph=True)
                optimizer.step()

                # Deform the mesh
                mask = torch.zeros_like(deform_verts).to(self.device)
                mask[deform_verts > 0] = 1.0
                deform_verts = deform_verts * vertex_normal * mask
                new_src_mesh = src_mesh.offset_verts(deform_verts)

        new_mesh = trimesh.Trimesh(new_src_mesh[0].verts_packed().detach().cpu().numpy(),
                                   new_src_mesh[0].faces_packed().detach().cpu().numpy())
        new_mesh.show()
        return trimesh.smoothing.filter_laplacian(new_mesh)

        # if return_verts:
        #     return new_src_mesh[0].verts_packed().detach().cpu().numpy()
        # else:
        #     # update color as well.
        #     disp = new_src_mesh[0].verts_packed() - initial_vts
        #     norm = torch.norm(disp, dim=1, keepdim=True)
        #     disp = disp / norm
        #     vertices = initial_vts + disp * torch.min(norm, torch.Tensor([2.0]).to(self.device))
        #     vertices = vertices.detach().cpu().numpy()
        #     faces = new_src_mesh[0].faces_packed().detach().cpu().numpy()
        #
        #     kdtree = KDTree(trg_mesh_trimesh.vertices, leaf_size=30, metric='euclidean')
        #     idx = kdtree.query(vertices, k=1, return_distance=False)
        #     vis_mesh = trimesh.Trimesh(vertices, faces,
        #                                vertex_colors=trg_mesh_trimesh.visual.vertex_colors[idx[:, 0], :], process=False)
        #     return trimesh.smoothing.filter_laplacian(vis_mesh)
        #     # vis_mesh.show()
        #     # return trimesh.smoothing.
        #     # return vis_mesh
        #     # return trimesh.smoothing.filter_humphrey(vis_mesh)

