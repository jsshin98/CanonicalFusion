import os
import cv2
import json
import glob
import trimesh
import torch
from apps.misc import load_images
from argparse import Namespace
from smpl_optimizer.misc import keypoint_loader
###########################################################
# Directory hierarchies                                   #
# ------------------------------------------------------- #
# The following structure must be kept. Otherwise,        #
# we cannot guarantee the quality of output.              #
# Last update on March 21, 2023                           #
###########################################################
# train set must have the following information.
# ./DATASET/TRAIN (L: sub_dirs, >: files)
# ├─ OBJ
# │  └─ {DATASET_%s}
# │     └─ {DATA00} > {%s_%02d}.obj
# └─ {DATASET_%s} (RP, IOYS, THuman, ...)
#    ├─ COLOR
#    │  ├─ NR
#    │  │  └─ {DATA00} > {%d}_{%d}_{%02d}_front/back.png (uint8)
#    │  └─ ALIGNED
#    │     └─ {DATA00} > {%d}_{%d}_{%02d}_front/back.png (uint8)
#    ├─ DEPTH
#    │  ├─ ALIGNED (trimesh renderer)
#    │  │  └─ {DATA00} > {%d}_{%d}_{%02d}_front/back.png (uint16)
#    │  ├─ ALIGNED_UNBIASED
#    │  │  └─ {DATA00} > {%d}_{%d}_{%02d}_front/back.png (uint16)
#    │  ├─ ALIGNED_SMPLX
#    │  │  └─ {DATA00} > {%d}_{%d}_{%02d}_front/back.png (uint16)
#    │  └─ ALIGNED_UNBIASED_SMPLX
#    │     └─ {DATA00} > {%d}_{%d}_{%02d}_front/back.png (uint16)
#    ├─ MASK
#    │ ├─ NR
#    │ │  └─ {DATA00} > {%d}_{%d}_{%02d}.png (uint8)
#    │ └─ ALIGNED
#    │    └─ {DATA00} > {%d}_{%d}_{%02d}.png (uint8)
#    ├─ NORMAL
#    │ ├─ ALIGNED
#    │ │  └─ {DATA00} > {%d}_{%d}_{%02d}_front/back.png (uint8)
#    │ ├─ ALIGNED_SMPLX
#    │ │  └─ {DATA00} > {%d}_{%d}_{%02d}_front/back.png (uint8)
#    │ └─ ALIGNED_UNBIASED_SMPLX
#    │    └─ {DATA00} > {%d}_{%d}_{%02d}_front/back.png (uint8)
#    ├─ PARAM
#    │ └─ CAM_PARAMS > {%d}_{%d}_{%02d}_front/back.json
#    ├─ POSE
#    │ ├─ OPENPOSE
#    │ │  └─ {DATA00} > {%d}.json
#    │ └─ OPENPIFPAF
#    │    └─ {DATA00} > {%d}.json
#    └─ SMPLX
#       ├─ {DATA00} > SMPL_params
#       └─ {DATA00} > SMPL_mesh
# test/eval set must have the following information.
# ./DATASET/{EVAL} (L: sub_dirs, >: files)
# ├─ {CUSTOM dir} (containing images)
# │  └─ {DATA00} > {%d}.png (uint8)
# ├─ DEPTH
# │  ├─ ALIGNED_UNBIASED
# │  │  └─ {DATA00} > {%d}_front/back.png (uint16)
# │  ├─ ALIGNED_SMPLX
# │  │  └─ {DATA00} > {%d}_front/back.png (uint16)
# │  └─ ALIGNED_UNBIASED_SMPLX
# │     └─ {DATA00} > {%d}_front/back.png (uint16)
# ├─ NORMAL
# │  ├─ ALIGNED_UNBIASED
# │  │  └─ {DATA00} > {%d}_front/back.png (uint8)
# │  ├─ ALIGNED_SMPLX
# │  │  └─ {DATA00} > {%d}_front/back.png (uint8)
# │  └─ ALIGNED_UNBIASED_SMPLX
# │     └─ {DATA00} > {%d}_front/back.png (uint8)
# ├─ POSE
# │  ├─ OPENPOSE
# │  │  └─ {DATA00} > {%d}.json
# │  └─ OPENPIFPAF
# │     └─ {DATA00} > {%d}.json
# ├─ SMPLX
# │  ├─ {DATA00} > SMPL_params
# │  └─ {DATA00} > SMPL_mesh
# └─ MESH
#    ├─ {DATA00} > {%d}.obj
#    ├─ {DATA00} > {%d}.pkl
#    └─ {DATA00} > {%d}.fbx (not implemented)
###########################################


class DirectoryManager:
    def __init__(self, params, renderer='nr', config='smpl', device='cuda:0'):
        self.input_root = params.path2data
        self.params = params
        self.device = device

        # self.img_dirs = dict()
        # if renderer == 'nr':
        #     self.img_dirs['COLOR/NR'] = os.path.join(self.save_root, 'COLOR/NR')
        #     self.img_dirs['MASK/NR'] = os.path.join(self.save_root, 'MASK/NR')
        #     self.img_dirs['DEPTH/NR'] = os.path.join(self.save_root, 'DEPTH/NR')
        #     self.img_dirs['NORMAL/NR'] = os.path.join(self.save_root, 'NORMAL/NR')
        # if renderer == 'trimesh':
        #     self.img_dirs['COLOR/ALIGNED'] = os.path.join(self.save_root, 'COLOR/ALIGNED')
        #     self.img_dirs['DEPTH/ALIGNED'] = os.path.join(self.save_root, 'DEPTH/ALIGNED')
        #     self.img_dirs['DEPTH/ALIGNED_UNBIASED'] = os.path.join(self.save_root, 'DEPTH/ALIGNED_UNBIASED')
        #     self.img_dirs['NORMAL/ALIGNED'] = os.path.join(self.save_root, 'NORMAL/ALIGNED')
        #     self.img_dirs['NORMAL/ALIGNED_UNBIASED'] = os.path.join(self.save_root, 'NORMAL/ALIGNED_UNBIASED')
        #
        # if 'smpl' in rendering_config:
        #     self.img_dirs['DEPTH/ALIGNED_SMPLX'] = os.path.join(self.save_root, 'DEPTH/ALIGNED_SMPLX')
        #     self.img_dirs['DEPTH/ALIGNED_UNBIASED_SMPLX'] \
        #         = os.path.join(self.save_root, 'DEPTH/ALIGNED_UNBIASED_SMPLX')
        #     self.img_dirs['NORMAL/SMPLX_ALIGNED'] = os.path.join(self.save_root, 'NORMAL/ALIGNED_SMPLX')
        #     self.img_dirs['NORMAL/SMPLX_ALIGNED_UNBIASED'] \
        #         = os.path.join(self.save_root, 'NORMAL/ALIGNED_UNBIASED_SMPLX')
        #
        # self.param_dirs = dict()
        # self.param_dirs['PARAM'] = os.path.join(self.save_root, 'PARAM')
        # self.param_dirs['POSE'] = os.path.join(self.save_root, 'POSE')
        # self.param_dirs['SMPLX'] = os.path.join(self.save_root, 'SMPLX')

        self.path2cam = os.path.join(self.params.path2data, 'PARAM')
        self.path2depth = os.path.join(self.params.path2data, 'DEPTH/ALIGNED_UNBIASED_SMPLX')
        self.path2pose = os.path.join(self.params.path2data, 'POSE')
        self.path2mesh = os.path.join(self.params.path2data, 'MESH')
        self.path2smpl = os.path.join(self.params.path2data, 'SMPLX')
        self.data_list = []

    def set_data(self):
        self.data_list = glob.glob(os.path.join(self.params.path2image, '/*' if self.params.path2image[-1] != '/' else '*'))

    def fetch_data(self):
        data = self.data_list.pop(0)
        images, files = load_images(data, width=self.params.recon_res, height=self.params.recon_res)
        data_name = data.split('/')[-1]

        out = {'image': images, 'depth_front': [], 'depth_back': [], 'smpl_mesh': [], 'smpl_param': [],
               'keypoint': [], 'cam_param': [], 'mesh': []}
        for i, file in enumerate(files):
            file = file.split('/')[-1]
            ext = file.split('.')[-1]
            if '_front.png' not in files[i]:
                file = file.replace('.' + ext, '_front.' + ext)
            file_front = os.path.join(self.path2depth, data_name, file)
            file_back = file_front.replace('_front.', '_back.')
            if os.path.isfile(file_front) and os.path.isfile(file_back):
                out['depth_front'].append(cv2.imread(file_front, cv2.IMREAD_ANYDEPTH).astype(float))
                out['depth_back'].append(cv2.imread(file_back, cv2.IMREAD_ANYDEPTH).astype(float))

        mesh_files = glob.glob(os.path.join(self.path2smpl, data_name, '*.obj'))
        for i in range(len(mesh_files)):
            out['smpl_mesh'].append(trimesh.load_mesh(mesh_files[i]))

        mesh_files = glob.glob(os.path.join(self.path2mesh, data_name, '*.obj'))
        for i in range(len(mesh_files)):
            out['mesh'].append(trimesh.load_mesh(mesh_files[i]))

        param_files = glob.glob(os.path.join(self.path2smpl, data_name, '*.json'))
        for i in range(len(param_files)):
            with open(param_files[i], 'r') as f:
                params = json.load(f)
                for key in params.keys():
                    if isinstance(params[key], list):
                        params[key] = torch.Tensor(params[key]).reshape(1, -1).to(self.device)
                    else:
                        params[key] = torch.Tensor([params[key]]).reshape(1, -1).to(self.device)
                out['smpl_param'].append(Namespace(**params))

        for i, file in enumerate(files):
            file = file.split('/')[-1]
            ext = file.split('.')[-1]
            out['keypoint'].append(keypoint_loader(os.path.join(self.path2pose, data_name, file.replace(ext, 'json'))))

        param_files = glob.glob(os.path.join(self.path2cam, data_name, '*.json'))
        for i in range(len(param_files)):
            with open(param_files[i], 'r') as f:
                out['cam_param'] = json.load(f)

        return Namespace(**out)

    def set_dirs(self, data_name=None):
        for dir in self.img_dirs.items():
            if data_name is None:
                os.makedirs(self.img_dirs[dir], exist_ok=True)
            else:
                os.makedirs(os.path.join(self.img_dirs[dir], data_name), exist_ok=True)

    def get_filenames(self, data_name, vid, p):
        filenames = dict()
        filenames['image'] = os.path.join(self.directories['COLOR/NR'], data_name, '%d_%d_%02d.png' % (vid, p, 0))
        filenames['mask'] = os.path.join(self.directories['MASK/NR'], data_name, '%d_%d_%02d.png' % (vid, p, 0))
        filenames['depth'] = os.path.join(self.directories['DEPTH/NR'], data_name, '%d_%d_%02d.png' % (vid, p, 0))
        filenames['normal'] = os.path.join(self.directories['NORMAL/NR'], data_name, '%d_%d_%02d.png' % (vid, p, 0))
        filenames['param'] = os.path.join(self.directories['PARAM'], data_name, '%d_%d_%02d.json' % (vid, p, 0))

        return filenames

    def fetch_pose_data(self, path2pose):
        if not isinstance(path2pose, list):
            path2pose = [path2pose]

        out = {'pose': []}
        for path in path2pose:
            dir_list = path.split('/')
            dataname = dir_list[-2]
            filename = dir_list[-1]
            pose_path = os.path.join(self.input_root, 'POSE', dataname,
                                     filename.replace('.png', '.json'))
            if os.path.isfile(pose_path):
                keypoints = keypoint_loader(pose_path)
                out['pose'].append(keypoints)
        return out

    def fetch_smpl_data(self, path2img):
        if not isinstance(path2img, list):
            path2img = [path2img]
        out = {'smpl_params': [], 'normal_front': [], 'normal_back': [],
               'depth_front': [], 'depth_back': []}

        for path in path2img:
            dir_list = path.split('/')
            dataname = dir_list[-2]
            filename = dir_list[-1]
            smpl_path = os.path.join(self.input_root, 'SMPLX', dataname,
                                     filename.replace('.png', '.json'))
            normal_front = os.path.join(self.input_root, 'NORMAL/ALIGNED_UNBIASED_SMPLX',
                                        dataname, filename.replace('.png', '_front.png'))
            normal_back = normal_front.replace('_front.png', '_back.png')
            depth_front = os.path.join(self.input_root, 'DEPTH/ALIGNED_UNBIASED_SMPLX',
                                       dataname, filename.replace('.png', '_front.png'))
            depth_back = depth_front.replace('_front.png', '_back.png')

            if os.path.isfile(smpl_path):
                with open(filename, 'r') as f:
                    out['smpl_params'].append(json.load(f))
            if os.path.isfile(normal_front):
                out['normal_front'].append(cv2.imread(normal_front))
            if os.path.isfile(normal_back):
                out['normal_back'].append(cv2.imread(normal_back))
            if os.path.isfile(depth_front):
                out['depth_front'].append(cv2.imread(depth_front))
            if os.path.isfile(depth_back):
                out['depth_back'].append(cv2.imread(depth_back))
        return out


