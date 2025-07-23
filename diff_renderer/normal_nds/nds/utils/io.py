import numpy as np
from pathlib import Path
import torch
import json
import trimesh

from diff_renderer.normal_nds.nds.core import Mesh, View
from diff_renderer.normal_nds import utils as recon_utils
import smplx
import os
from sklearn.neighbors import KDTree

def read_mesh(path, device='cpu'):
    mesh_ = trimesh.load_mesh(str(path), process=False)

    vertices = np.array(mesh_.vertices, dtype=np.float32)
    indices = None
    if hasattr(mesh_, 'faces'):
        indices = np.array(mesh_.faces, dtype=np.int32)

    return Mesh(vertices, indices, device)

def write_mesh(path, mesh, flip=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vertices = mesh.vertices.numpy()
    indices = mesh.indices.numpy() if mesh.indices is not None else None

    mesh.compute_normals()
    vertex_normals = mesh.vertex_normals.numpy()
    face_normals = mesh.face_normals.numpy()
    colors = mesh.colors.numpy()
    mesh_ = trimesh.Trimesh(
        vertices=vertices, faces=indices, 
        face_normals=face_normals, 
        vertex_normals=vertex_normals,
        vertex_colors=colors,
        process=False)
    
    if flip:
        mesh_.apply_transform([[-1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    mesh_.export(path, include_normals=True)
    
    # initial_mesh_txt = recon_utils.export_obj(mesh, include_texture=True)

    # with open(os.path.join(save_path, 'remeshed.obj'), 'w') as f:
    #     for i in initial_mesh_txt.splitlines():
    #         f.write(i + '\n')
    # f.close()

def read_views(directory, mask_threshold, scale, device, data_type):
    if data_type == 'dvr':
        assert len(directory) == 1
        image_paths = sorted([path for path in directory[0].iterdir() if (path.is_file() and path.suffix == '.png')])
    
        views = []
        for image_path in image_paths:
            views.append(View.load_dvr(str(image_path), view_angle=(image_path.stem), device=device))
    elif data_type == 'co3d':
        assert len(directory) == 3 # directory[0] : images path, directory[1] : masks path, directory[2] : cameras path
        cam = np.load(directory[2])
        assert len(cam.keys()) % 3 == 0

        image_paths = sorted([path for path in directory[0].iterdir()])
        mask_paths = sorted([path for path in directory[1].iterdir()])

        cam_key = cam.keys()
        views = []
        '''
        for i in range(len(image_paths)):
            if image_paths[i].name in cam_key:
                #assert image_paths[i].name in mask_paths[i].name
                cam_id = str(cam[image_paths[i].name])
                pose = cam["pose_"+cam_id]
                intrinsic = cam["intrinsic_"+cam_id]
                views.append(View.load_co3d(image_paths[i], mask_paths[i], pose, intrinsic, device))
        '''
        mask_name_list = [path.name for path in mask_paths]
        for i in range(len(image_paths)):
            if image_paths[i].name in cam_key:
                if image_paths[i].name+".png" not in mask_name_list:
                    continue
                ind = mask_name_list.index(image_paths[i].name+".png")
                cam_id = str(cam[image_paths[i].name])
                pose = cam["pose_"+cam_id]
                intrinsic = cam["intrinsic_"+cam_id]
                view_co3d = View.load_co3d(image_paths[i], mask_paths[ind], pose, intrinsic, mask_threshold, device)
                if view_co3d is not None:
                    views.append(view_co3d)
    else:
        raise Exception("Invalid dataset type")
    
    print("Found {:d} views".format(len(views)))

    if scale > 1:
        for view in views:
            view.scale(scale)
        print("Scaled views to 1/{:d}th size".format(scale))

    return views

def load_smpl_info(smpl_params, smpl_type='smplx', resource_dir=None, tpose=True, device='cuda'):
    full_poses = []
    smpl_infos = []
    smpl_models = []
    eyeball_fid = np.load(os.path.join(resource_dir, 'eyeball_fid.npy'))
    fill_mouth_fid = np.load(os.path.join(resource_dir, 'fill_mouth_fid.npy'))
    
    for smpl_param in smpl_params:
        full_pose = torch.zeros((55, 3))
        full_pose[0:1, :] = smpl_param.global_orient
        full_pose[1:22, :] = smpl_param.body_pose.reshape(21, 3)
        full_poses.append(full_pose)
        num_pca_comps = len(smpl_param.right_hand_pose[0])
        gender = smpl_param.gender

    if num_pca_comps != 45:
        smplx_model = smplx.create(resource_dir,
                                    model_type='smplx',
                                    gender=gender,
                                    num_betas=10, ext='npz',
                                    use_face_contour=True,
                                    flat_hand_mean=False,
                                    use_pca=True,
                                    num_pca_comps=num_pca_comps).cuda()
    else:
        smplx_model = smplx.create(resource_dir,
                                    model_type='smplx',
                                    gender=gender,
                                    num_betas=10, ext='npz',
                                    use_face_contour=True,
                                    flat_hand_mean=False,
                                    use_pca=False).cuda()
            
    for smpl_param in smpl_params:
        smpl_info = {}

        posed_smpl, smpl_model, smpl_param = recon_utils.set_smpl_model(smplx_model, smpl_param, device=device)
        
        lbs = smpl_model.lbs_weights.detach().cpu().numpy()
        v_pose_smpl = trimesh.Trimesh(smpl_model.v_template.cpu(), smpl_model.faces)
        canon_smpl_vertices = smpl_model.v_template + smplx.lbs.blend_shapes(smpl_model.betas, smpl_model.shapedirs)
        canon_smpl = trimesh.Trimesh(canon_smpl_vertices.squeeze(0).detach().cpu().numpy(),
                                          smpl_model.faces, process=False)
        
        if tpose:
            smpl = canon_smpl
        else:
            smpl = posed_smpl
                        
        # remove the eyeball and fill the mouth
        smpl_faces_not_watertight = smpl.faces
        smpl_faces_no_eybeballs = smpl_faces_not_watertight[~eyeball_fid]
        faces = np.concatenate([smpl_faces_no_eybeballs, fill_mouth_fid], axis=0)
        smpl = trimesh.Trimesh(smpl.vertices, faces)
        
        joints = smplx.lbs.vertices2joints(smpl_model.J_regressor, canon_smpl_vertices)

        smpl_info['vertices'] = posed_smpl.vertices
        smpl_info['lbs'] = lbs
        smpl_info['transl'] = smpl_param.transl
        smpl_info['scale'] = smpl_param.scale
        smpl_info['joints'] = joints

        smpl_infos.append(smpl_info)
        smpl_models.append(smpl_model)
    return smpl, smpl_infos, full_poses, smpl_models