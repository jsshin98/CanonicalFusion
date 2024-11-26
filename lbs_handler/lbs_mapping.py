import sys
sys.path.append('./')
import smplx
import torch
import numpy as np
import pdb
import trimesh
import torch.nn as nn
import pickle
import argparse
import os
import json
import utils
import glob
import model
from pysdf import SDF
from sklearn.neighbors import KDTree

class LBS_MAPPING(nn.Module):
    def __init__(self, data_path='', mesh_path='/MESH/GT_ALIGNED', dataset='RP', save_path='', resource_path='',
                 pretrained_path='./result_uv_1024', type='flow', device='cuda'):
        super(LBS_MAPPING, self).__init__()
        self.data_path = data_path
        self.mesh_path = mesh_path
        self.dataset = dataset
        self.save_path = save_path
        self.resource_path = resource_path
        self.pretrained_path = pretrained_path
        self.device = device
        self.type = type
        self.data_list = glob.glob(os.path.join(self.data_path, self.dataset, self.mesh_path, '*'))

        self.encoded_lbs = np.load(os.path.join(self.pretrained_path, 'lbs_encoded_10475.npy'))
        self.full_lbs = np.load(os.path.join(self.pretrained_path, 'lbs_flattened_10475.npy'))
        # #
        # smpl_model = smplx.create(model_path=self.resource_path,
        #                           model_type='smplx',
        #                           gender='male',
        #                           num_betas=10, ext='npz',
        #                           use_face_contour=True,
        #                           flat_hand_mean=True,
        #                           use_pca=True,
        #                           num_pca_comp=12
        #                           ).to('cuda')
        # self.full_lbs = smpl_model.lbs_weights.detach().cpu().numpy()

        self.mlp = model.LBSModel().cuda()
        self.mlp.load_state_dict(torch.load(os.path.join(pretrained_path, 'best.tar'))['state_dict'])
        self.mlp.eval()

    def sdf_flow(self, vts_src, mesh_tgt, iter=20):
        d = np.array([[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]])
        sdf = SDF(mesh_tgt.vertices, mesh_tgt.faces)
        for _ in range(iter):
            w = sdf(vts_src)
            w = np.clip(w, a_min=-0.001, a_max=0.001)
            dir = np.zeros_like(vts_src)
            dir[:, 0] = sdf(vts_src + d[0]) - sdf(vts_src - d[0])
            dir[:, 1] = sdf(vts_src + d[1]) - sdf(vts_src - d[1])
            dir[:, 2] = sdf(vts_src + d[2]) - sdf(vts_src - d[2])
            norm = np.linalg.norm(dir, axis=1, keepdims=True)
            # in case of dir = [0, 0, 0] (boundary condition)
            dir = np.nan_to_num(dir, nan=1.0)
            norm = np.nan_to_num(norm, nan=1.0)
            dir = dir / norm
            dir = np.nan_to_num(dir, nan=1.0)

            vts_src = vts_src - dir * w.reshape(-1, 1)
            # mesh = trimesh.Trimesh(vts_src, self.custom_faces)
            # show_meshes([mesh, mesh_tgt])
        return vts_src

    def subdivide_lbs(self, vertices, faces, lbs, iter=1):
        stacked = np.hstack((vertices, lbs))
        for i in range(iter):
            stacked, faces = trimesh.remesh.subdivide(vertices=stacked, faces=faces)
        return stacked[:, 3:], stacked[:, :3]

    def mapping(self, mesh_file):
        self.mesh = trimesh.load(mesh_file, process=False)
        self.smplx = trimesh.load(mesh_file.replace('GT_ALIGNED_TEMP', 'SMPLX'), process=False)
        # self.smplx = trimesh.load(mesh_file.replace('.obj', '_smplx.obj'))

        mesh_color = np.zeros((self.mesh.vertices.shape))
        mesh_lbs = torch.zeros((self.mesh.vertices.shape[0], 55))

        if self.dataset == 'RP_ETC':
            if self.type == 'nearest':
                nearest_distances, nearest_sources = self.smplx.kdtree.query(self.mesh.vertices)
                mesh_color[:] = self.encoded_lbs[nearest_sources]
                mesh_lbs[:] = torch.Tensor(self.full_lbs[nearest_sources])
            elif self.type == 'flow':
                self.smplx.vertices = self.smplx.vertices / 200
                self.mesh.vertices = self.mesh.vertices / 200

                warped_vts = self.sdf_flow(self.mesh.vertices, self.smplx, iter=5)
                new_lbs, new_vertices = self.subdivide_lbs(self.smplx.vertices, self.smplx.faces, self.full_lbs, iter=1)
                # new_vertices = np.concatenate([self.smplx.vertices, new_vertices], axis=0)
                # new_lbs = torch.cat((torch.Tensor(self.full_lbs).to(self.device), torch.Tensor(new_lbs).to(self.device)), dim=0)
                kdtree = KDTree(new_vertices, leaf_size=30, metric='euclidean')
                kd_idx = kdtree.query(warped_vts, k=1, return_distance=False)
                custom_lbs = torch.Tensor(new_lbs[kd_idx.squeeze(), :]).to(self.device)
                mesh_color = self.mlp.encoder(custom_lbs).detach().cpu().numpy()

        elif self.dataset == 'TH_ETC':
            if self.type == 'nearest':
                nearest_distances, nearest_sources = self.smplx.kdtree.query(self.mesh.vertices)
                mesh_color[:] = self.encoded_lbs[nearest_sources]
                mesh_lbs[:] = torch.Tensor(self.full_lbs[nearest_sources])

            elif self.type == 'flow':
                self.smplx.vertices = self.smplx.vertices / 200
                self.mesh.vertices = self.mesh.vertices / 200

                warped_vts = self.sdf_flow(self.mesh.vertices, self.smplx, iter=5)
                new_lbs, new_vertices = self.subdivide_lbs(self.smplx.vertices, self.smplx.faces, self.full_lbs, iter=3)
                kdtree = KDTree(new_vertices, leaf_size=30, metric='euclidean')
                kd_idx = kdtree.query(warped_vts, k=1, return_distance=False)
                custom_lbs = torch.Tensor(new_lbs[kd_idx.squeeze(), :]).to(self.device)
                mesh_color = self.mlp.encoder(custom_lbs).detach().cpu().numpy()

        elif self.dataset == 'RP_T_DIFFUSION':
            if self.type == 'nearest':
                nearest_distances, nearest_sources = self.smplx.kdtree.query(self.mesh.vertices)
                mesh_color[:] = self.encoded_lbs[nearest_sources]
                mesh_lbs[:] = torch.Tensor(self.full_lbs[nearest_sources])

            elif self.type == 'flow':
                self.smplx.vertices = self.smplx.vertices / 200
                self.mesh.vertices = self.mesh.vertices / 200

                warped_vts = self.sdf_flow(self.mesh.vertices, self.smplx, iter=5)
                new_lbs, new_vertices = self.subdivide_lbs(self.smplx.vertices, self.smplx.faces, self.full_lbs, iter=1)
                kdtree = KDTree(new_vertices, leaf_size=30, metric='euclidean')
                kd_idx = kdtree.query(warped_vts, k=1, return_distance=False)
                custom_lbs = torch.Tensor(new_lbs[kd_idx.squeeze(), :]).to(self.device)
                mesh_color = self.mlp.encoder(custom_lbs).detach().cpu().numpy()


        result = {'lbs': custom_lbs.detach(), 'vertices': self.mesh.vertices, 'faces': self.mesh.faces}

        if not os.path.exists(os.path.join(self.data_path, self.dataset, self.save_path, (mesh_file.split('/')[-2]))):
            #os.makedirs(os.path.join(self.save_path, mesh_file.split('/')[-3]), exist_ok=True)
            os.makedirs(os.path.join(self.data_path, self.dataset, self.save_path, mesh_file.split('/')[-2]), exist_ok=True)
        # with open(os.path.join(self.data_path, self.dataset, self.save_path, '/'.join(mesh_file.split('/')[-2:])).replace('.obj', '.npy'),
        #           'wb') as f:
        np.save(os.path.join(self.data_path, self.dataset, self.save_path, '/'.join(mesh_file.split('/')[-2:])).replace('.obj', '.npy'), result)

        mesh_mlp = trimesh.Trimesh(self.mesh.vertices * 200, self.mesh.faces, vertex_colors=(mesh_color), process=False)
        mesh_mlp.export(os.path.join(self.data_path, self.dataset, self.save_path, '/'.join(mesh_file.split('/')[-2:])))

    def forward(self):
        for data in self.data_list:
            print(data.split('/')[-1])
            mesh_file = os.path.join(data, data.split('/')[-1]+'.obj')
            self.mapping(mesh_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LBS unwrapping Preprocessor')

    parser.add_argument('--data-path', type=str, default='/media/jisu/DATASET/ECCV2024/')
    parser.add_argument('--mesh-path', type=str, default='GT_ALIGNED_TEMP') #OBJ MESH/CANON_GT MESH/GT_ALIGNED
    parser.add_argument('--dataset', type=str, default='RP_T_DIFFUSION')
    parser.add_argument('--resource-path', type=str, default='/media/jisu/DATASET/ECCV2024/resource/smpl_models')
    parser.add_argument('--save_path', type=str, default='MLP_FLOW_TEMP')
    parser.add_argument('--type', type=str, default='flow')

    parser.add_argument('--pretrained_path', type=str, default='./result_uv_1024')


    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lbs_preprocessor = LBS_MAPPING(args.data_path, args.mesh_path, args.dataset, args.save_path, args.resource_path, args.pretrained_path, args.type, device)
    lbs_preprocessor()
