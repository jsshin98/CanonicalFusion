import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import pickle as pkl
import json
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import os, sys
import glob
import sys
import yaml
import trimesh
import smplx
cur_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(cur_path + '/pytorch3d')
sys.path.append(cur_path + '/nvdiffrast')
sys.path.append(cur_path)
from misc import load_image, load_smpl, load_normal
from diff_renderer.canonfusion import CanonFusion


if __name__ == '__main__':
    # config_file = sys.argv[1]
    config_file = 'canonfusion.yaml'
    config = OmegaConf.load(config_file)
    config_cli = OmegaConf.from_cli()
    args = OmegaConf.merge(config, config_cli)  # update config from command line

    data_dir = args.dataset.data_dir
    data_type = args.dataset.dataset_type
    datasets = args.dataset.dataset_name
    color_path = args.dataset.color_path
    normal_path = args.dataset.normal_path
    mesh_path = args.dataset.recon_path
    smpl_path = args.dataset.smpl_path
    result_path = args.dataset.diffrend_path
    
    cam_path = args.dataset.cam_path
    cam_config = args.dataset.cam_config

    with open(os.path.join(data_dir, data_type, cam_path)) as f:
        cam_params = yaml.load(f, Loader=yaml.FullLoader)
        cam_params = cam_params[cam_config]
        
    canonfusion = CanonFusion(args, cam_params)

    '''
    dataset tree:
    ├── d (SET1)
    │   └── (COLOR/DIFFUSE/ or MESH/INIT/) 
    |       └── data_name (0464)
    |          if use_multi_frame:
    │          └── frame1/
    │               └── 000_000_000_front.(png/obj)
    │               └── ...
    │           └── frame2/
    │               └── 000_000_000_front.(png/obj)
    │               └── ...
    |          else:
    │           └── 000_000_000_front.(png/obj)
    │           └── ...
    │       └── ...
    '''
    for d in datasets:
        data_path = os.path.join(data_dir,  data_type, d, color_path)
        data_names = sorted(os.listdir(data_path)) # avatar lists ex) [0464, 0478, 0490, ...]
        
        for data_name in data_names:
            input_frames, input_masks, input_normals = [], [], []
            smpl_params, smpl_meshes = [], []
            reconstructed_meshes, canonicalized_meshes = [], [] # posed meshes, canonicalized meshes
            if args.canonfusion.use_multi_frame:
                frame_lists = os.listdir(os.path.join(data_path, data_name))# frame lists ex) [frame1, frame2, frame3, ...]
                if args.canonfusion.frame_numbers is not None:
                    frame_lists = frame_lists[args.canonfusion.frame_numbers]
                for frame_list in frame_lists:
                    tmp_img_list, tmp_mask_list, tmp_normal_list = [], [], []
                    for view_angle in args.canonfusion.view_angle:
                        img, mask, _ = load_image(os.path.join(data_path, data_name, frame_list, '000_' + str(view_angle) + '_000_front.png'), pred_res=args.canonfusion.res)
                        if args.canonfusion.use_normal:
                            normal = load_normal(os.path.join(data_path.replace(color_path, normal_path), data_name, frame_list, '000_' + str(view_angle) + '_000_front.png'), pred_res=args.canonfusion.res)
                        else:
                            normal = np.ones(img.shape)
                        tmp_img_list.append(img)
                        tmp_mask_list.append(mask)
                        tmp_normal_list.append(normal)
                    input_frames.append(tmp_img_list)
                    input_masks.append(tmp_mask_list)
                    input_normals.append(tmp_normal_list)
                    smpl_param, smpl_mesh = load_smpl(os.path.join(data_path.replace(color_path, smpl_path), data_name, frame_list, frame_list + '.json'))
                    smpl_params.append(smpl_param)
                    smpl_meshes.append(smpl_mesh)
                if data_path.replace(color_path, mesh_path) is not None and args.canonfusion.use_mesh:
                    for frame_list in frame_lists:
                        reconstructed_meshes.append(trimesh.load(os.path.join(data_path.replace(color_path, mesh_path) , data_name, frame_list, frame_list + '_color.obj')))
                        canonicalized_meshes.append(trimesh.load(os.path.join(data_path.replace(color_path, mesh_path), data_name, frame_list, frame_list + '_canon_from_smpl.obj')))
                output_dir = os.path.join(data_path.replace(color_path, result_path), data_name + '_'.join(args.canonfusion.frame_numbers))

            else:
                tmp_img_list, tmp_mask_list, tmp_normal_list = [], [], []
                for view_angle in args.canonfusion.view_angle:
                    img, mask, _ = load_image(os.path.join(data_path, data_name, '000_' + str(view_angle) + '_000_front.png'), pred_res=args.canonfusion.res)
                    if args.canonfusion.use_normal:
                        normal = load_normal(os.path.join(data_path.replace(color_path, normal_path), data_name, '000_' + str(view_angle) + '_000_front.png'), pred_res=args.canonfusion.res)
                    else:
                        normal = np.ones(img.shape)
                    tmp_img_list.append(img)
                    tmp_mask_list.append(mask)
                    tmp_normal_list.append(normal)
                input_frames.append(tmp_img_list)
                input_masks.append(tmp_mask_list)
                input_normals.append(tmp_normal_list)
                smpl_param, smpl_mesh = load_smpl(os.path.join(data_path.replace(color_path, smpl_path), data_name, data_name + '.json'))
                smpl_params.append(smpl_param)
                smpl_meshes.append(smpl_mesh)
                if data_path.replace(color_path, mesh_path) is not None and args.canonfusion.use_mesh:
                    reconstructed_meshes.append(trimesh.load(os.path.join(data_path.replace(color_path, mesh_path) , data_name, data_name + '_color.obj')))
                    canonicalized_meshes.append(trimesh.load(os.path.join(data_path.replace(color_path, mesh_path), data_name, data_name + '_canon_from_smpl.obj')))
                output_dir = os.path.join(data_path.replace(color_path, result_path), data_name)
            
            os.makedirs(output_dir, exist_ok=True)
            canonfusion.forward(input_frames=input_frames, input_masks=input_masks, input_normals=input_normals, smpl_params=smpl_params, smpl_meshes=smpl_meshes, posed_meshes=reconstructed_meshes, canon_meshes=canonicalized_meshes, output_dir=output_dir, cam_params=cam_params, config=args)