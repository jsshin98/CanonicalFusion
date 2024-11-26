import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import glob
import smplx, trimesh
import json
import pdb
import pickle
from PIL import Image

# smplx_partids = {
#     'body': [0,1,2,3,4,5,6,9,13,14,16,17,18,19],
#     'face': [12, 15, 22],
#     'eyeball': [23, 24],
#     'leg': [4, 5, 7, 8, 10, 11],
#     'arm': [18, 19, 20, 21],
#     'handl': [20] + list(range(25, 40)),
#     'handr': [21] + list(range(40, 55)),
#     'footl': [7,10],
#     'footr': [8,11],
#     'ftip': [27, 30, 33, 36, 39, 42, 45, 48, 51, 54]
# }

# def postprocess_mesh(mesh, num_faces=2000):
#     """Post processing mesh by removing small isolated pieces.

#     Args:
#         mesh (trimesh.Trimesh): input mesh to be processed
#         num_faces (int, optional): min face num threshold. Defaults to 4096.
#     """
#     total_num_faces = len(mesh.faces)
#     if num_faces is None:
#         num_faces = total_num_faces // 100
#     cc = trimesh.graph.connected_components(
#         mesh.face_adjacency, min_len=3)
#     mask = np.zeros(total_num_faces, dtype=np.bool)
#     cc = np.concatenate([
#         c for c in cc if len(c) > num_faces
#     ], axis=0)
#     mask[cc] = True
#     mesh.update_faces(mask)
#     return mesh


# def subdivide_concat(vertices, faces, smpl_data):
#     vertices, faces = trimesh.remesh.subdivide(
#         vertices = np.hstack((vertices, smpl_data)),
#         faces = faces,
#     )
#     return vertices[:, :3], faces, vertices[:, 3:]

# with open('/media/jisu/JISU_8T/ECCV2024/resource/body_segmentation/smplx/smplx_vert_segmentation.json', 'rb') as f:
#     smplx_seg = json.load(f)
    
# smplx_seg_keys = smplx_seg.keys()

# inverse_smplx_seg = dict()
# for key in smplx_seg_keys:
#     for idx in smplx_seg[key]:
#         inverse_smplx_seg[idx] = key

# def flatten_lbs(smpl_mesh, q_vts, lbs,):
#     part_seg = dict()
#     for key in smplx_seg.keys():
#         part_seg[key] = []

#     eyeremoved_mesh = postprocess_mesh(smpl_mesh, num_faces=2000)

#     nearest_distances, nearest_sources = eyeremoved_mesh.kdtree.query(q_vts[smplx_seg['leftEye']])
#     for dst, src in enumerate(nearest_sources):
#         lbs[src] = lbs[dst]

#     nearest_distances, nearest_sources = eyeremoved_mesh.kdtree.query(q_vts[smplx_seg['rightEye']])
#     for dst, src in enumerate(nearest_sources):
#         lbs[src] = lbs[dst]

#     nearest_distances, nearest_sources = smpl_mesh.kdtree.query(q_vts)
#     for dst, src in enumerate(nearest_sources):
#         try:
#             seg = inverse_smplx_seg[src]
#             part_seg[seg].append(dst)
#         except:
#             continue

#     for key in smplx_seg.keys():
#         lbs_part = lbs[part_seg[key]]
        
#         if key == 'rightHand' or key == 'rightHandIndex1':
#             ids = smplx_partids['handr']
#         elif key == 'leftHand' or key == 'leftHandIndex1':
#             ids = smplx_partids['handl']
#         # elif key == 'rightEye' or key == 'leftEye':
#         #     ids = smplx_partids['eyeball']
#         else:
#             continue
        
#         lbs_part_joints = lbs_part[:, ids]
#         lbs_part_joints_avg = torch.mean(lbs_part_joints, dim=1, keepdims=True)
#         lbs_part[:, ids] = lbs_part_joints_avg.repeat(1, lbs_part_joints.shape[1])
#         lbs[part_seg[key]] = lbs_part
    
#     return lbs

# smpl_model = smplx.create(model_path="/media/jisu/JISU_8T/ECCV2024/resource/smpl_models/",
#                                 model_type='smplx',
#                                 gender='male',
#                                 num_betas=10, ext='npz',
#                                 use_face_contour=True,
#                                 flat_hand_mean=True,
#                                 use_pca=False,
#                                 ).to('cuda')

# print(smpl_model(return_verts=True).vertices.detach().cpu().numpy().squeeze().shape)
# smplx_mesh = trimesh.Trimesh(smpl_model(return_verts=True).vertices.detach().cpu().numpy().squeeze(), smpl_model.faces, process=False)
# print(smplx_mesh.vertices.shape, smplx_mesh.faces.shape)
# lbs = smpl_model.lbs_weights.detach().cpu()
# lbs = flatten_lbs(smplx_mesh, smplx_mesh.vertices, lbs)

# vert, faces, lbs = subdivide_concat(smplx_mesh.vertices, smplx_mesh.faces, lbs)
# print(vert.shape, faces.shape, lbs.shape)


class LBSUnwrappingDataset(Dataset):
    def __init__(self, lbs_path):
        with open(lbs_path, 'rb') as f:
            lbs_map = pickle.load(f)
        
        mask = Image.open(lbs_path.replace('LBS.pickle', 'mask.png'))
        lbs_idx = np.where(np.array(mask)>0)

        self.lbs = lbs_map[lbs_idx[0], lbs_idx[1]]

    def __len__(self):
        return len(self.lbs)
    
    def __getitem__(self, idx):
        x = self.lbs[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)
