import torch.nn as nn
import torch
import smplx
import trimesh
import numpy as np
from model import LBSModel
from unwrapping.utils.utils import flatten_lbs
from glob import glob 
import utils
import os
import json
import pickle
from PIL import Image
from skimage.io import imread, imsave
import pdb

def load_real_mesh(dir_path):
    mesh_path = glob(dir_path + '/*.obj')[0]
    npy_path = glob(dir_path + '/*_RE.npy')[0]
    
    mesh = trimesh.load(mesh_path, process=False)


    lbs = np.load(npy_path, allow_pickle=True).item().get('lbs')
    
    return mesh, lbs

def lbs_decoder(lbs, pretrained_path):
    model = LBSModel().cuda()
    
    model.load_state_dict(torch.load(os.path.join(pretrained_path, 'best.tar'))['state_dict'])    
    
    decoder = model.decoder
    model.eval()
    with torch.no_grad():
        pred_lbs = decoder(torch.tensor(lbs, dtype=torch.float32).cuda())
    return pred_lbs

def set_smpl_model(smpl_model, smpl_params, device):
    smpl_model.betas = torch.nn.Parameter(torch.tensor([smpl_params['betas']], device=device))
    smpl_model.transl = torch.nn.Parameter(torch.tensor([smpl_params['transl']], device=device))
    smpl_model.expression = torch.nn.Parameter(torch.tensor([smpl_params['expression']], device=device))
    smpl_model.body_pose = torch.nn.Parameter(torch.tensor([smpl_params['body_pose']], device=device))
    smpl_model.global_orient = torch.nn.Parameter(torch.tensor([smpl_params['global_orient']], device=device))
    smpl_model.jaw_pose = torch.nn.Parameter(torch.tensor([smpl_params['jaw_pose']], device=device))
    smpl_model.left_hand_pose = torch.nn.Parameter(torch.tensor([smpl_params['left_hand_pose']], device=device))
    smpl_model.right_hand_pose = torch.nn.Parameter(torch.tensor([smpl_params['right_hand_pose']], device=device))
    smpl_mesh = smpl_model(return_verts=True, return_full_pose=True)
    smpl_mesh.joints = smpl_mesh.joints * torch.nn.Parameter(
        torch.tensor([smpl_params['scale']],
                     device=device))
    smpl_mesh.vertices = smpl_mesh.vertices * torch.nn.Parameter(
        torch.tensor([smpl_params['scale']],
                     device=device))
    return smpl_mesh, smpl_model

def main():
    pretrained_path = ''
    resource_path = ''
    mesh_path = ''
    pose_path = ''
    save_path = ''

    smpl_model = smplx.create(model_path=os.path.join(resource_path, "smpl_models/"),
                                    model_type='smplx',
                                    gender='male',
                                    num_betas=10, ext='npz',
                                    use_face_contour=True,
                                    flat_hand_mean=True,
                                    use_pca=True,
                                    num_pca_comp=12
                                    ).to('cuda')    

    mesh = trimesh.Trimesh(smpl_model(return_verts=True).vertices.detach().cpu().numpy().squeeze(), smpl_model.faces, process=False)
    lbs = smpl_model.lbs_weights.detach().cpu()

    
    real_mesh = trimesh.load(mesh_path, process=False)
    smpl_json = mesh_path.replace('obj', 'json')
    with open(smpl_json, 'r') as f:
        smpl_params = json.load(f)

    # smpl_output, smpl_model = set_smpl_model(smpl_model, smpl_params, device='cuda')
    canon_smpl_vertices = smpl_model.v_template + smplx.lbs.blend_shapes(smpl_model.betas, smpl_model.shapedirs)
    canon_smpl_mesh = trimesh.Trimesh(canon_smpl_vertices.detach().squeeze(0).detach().cpu().numpy(),
                                      smpl_model.faces, process=False)
    v_pose_smpl = trimesh.Trimesh(smpl_model.v_template.cpu(),
                                  smpl_model.faces)
    centroid_smpl = v_pose_smpl.bounding_box.centroid
    scale_smpl = 2.0 / np.max(v_pose_smpl.bounding_box.extents)

    real_mesh.vertices = (real_mesh.vertices - [0, 0, 0]) * 0.011111111112 / scale_smpl + centroid_smpl

    with open(os.path.join(resource_path, 'body_segmentation/smplx/smplx_vert_segmentation.json'), 'rb') as f:
        smplx_seg = json.load(f)

    lbs = flatten_lbs(mesh, mesh.vertices, lbs, smplx_seg)

    np.save(os.path.join(pretrained_path, 'lbs_flattened.npy'), lbs)

    real_mesh_lbs = np.zeros((real_mesh.vertices.shape[0], 55))
    nearest_distances, nearest_sources = mesh.kdtree.query(real_mesh.vertices)
    for dst, src in enumerate(nearest_sources):
        real_mesh_lbs[dst] = lbs[src]

    model = LBSModel().cuda()
    
    model.load_state_dict(torch.load(os.path.join(pretrained_path, 'best.tar'))['state_dict'])    
    
    encoder = model.encoder
    model.eval()

    with open('data/smplx_1024_UV_LBS.pickle', 'rb') as f:
        lbs_map = pickle.load(f)
    
    mask = Image.open('data/smplx_1024_UV_mask.png')
    lbs_idx = np.where(np.array(mask)>0)

    lbs = lbs_map[lbs_idx[0], lbs_idx[1]]

    lbs_uv_map = np.ones((1024, 1024, 3))

    pred_lbs = []
    with torch.no_grad():
        for idx, i in enumerate(real_mesh_lbs):
            temp = encoder(torch.tensor(i, dtype=torch.float32).cuda())
            pred_lbs.append(temp.detach().cpu().numpy())
            lbs_uv_map[lbs_idx[0][idx],lbs_idx[1][idx]] = temp.detach().cpu().numpy()

    # imsave(os.path.join(pretrained_path, 'lbs_encoded_uv.png'), (lbs_uv_map*255).astype(np.uint8))
    # np.save(os.path.join(pretrained_path, 'lbs_encoded.npy'), pred_lbs)

    # assign color to smplx
    # mesh.visual.vertex_colors = pred_lbs
    # mesh.export(os.path.join(pretrained_path, 'encoder.obj'))
    
    full_pose = torch.load(pose_path)
    
    decoder = model.decoder
    with torch.no_grad():
        reconstructed_lbs = decoder(torch.tensor(pred_lbs, dtype=torch.float32).cuda())


    pred_vertices = utils.deform_vertices(torch.Tensor(real_mesh.vertices).unsqueeze(0).cuda(), smpl_model,
                                          reconstructed_lbs, full_pose, inverse=False, return_vshape=False,
                                          device='cuda:0')

    pred_mesh = trimesh.Trimesh(vertices=pred_vertices.squeeze().detach().cpu().numpy(), faces=real_mesh.faces, vertex_colors=real_mesh.visual.vertex_colors,
                                process=False)
    pred_mesh.export(save_path)

    # pred_vertices = utils.deform_vertices(torch.Tensor(mesh.vertices).unsqueeze(0).cuda(), smpl_model, reconstructed_lbs, full_pose, inverse=False, return_vshape=False, device='cuda:0')
    #
    # pred_mesh = trimesh.Trimesh(vertices=pred_vertices.squeeze().detach().cpu().numpy(), faces=mesh.faces, process=False)
    # pred_mesh.export(os.path.join(pretrained_path, 'pred_realmesh.obj'))
if __name__ == '__main__':
    main()
    
