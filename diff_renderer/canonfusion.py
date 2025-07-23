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
import omegaconf
from omegaconf import OmegaConf
from datetime import datetime
import io
import time
import os, sys
import cv2
import glob
import sys
from pytorch3d.loss import chamfer_distance
from diff_renderer.normal_nds.reconstruct import NormalNDS
from diff_renderer.normal_nds.nds.utils import write_mesh, load_smpl_info
from diff_renderer.normal_nds import utils as recon_utils
from diff_renderer.img_utils import (to_cam, to_cam_B, to_pil, to_pil_color)
import yaml
import trimesh
from FlexiCubes.examples import optimize
from pytorch_lightning import seed_everything
import pickle

class CanonFusion():
    def __init__(self, config : omegaconf.dictconfig.DictConfig, cam_params):

        for key, value in config.canonfusion.items():
            setattr(self, key, value)
            
        self.normal_nds = NormalNDS(args=config.nds, cam_params=cam_params, device=self.device)
        self.smpl_data = os.path.join(config.dataset.data_dir, 'resource/smpl_models')
        self.replace_hands = config.nds.replace_hands
        self.view_angle = [int(i) for i in self.view_angle]
        self.device = 'cuda'

    def get_face_length(self, vts, faces):
        # check faces.
        areas = []
        for k in range(faces.shape[0]):
            x, y, z = faces[k]
            a = sum((vts[x, :] - vts[y, :]) ** 2) ** 2
            b = sum((vts[y, :] - vts[z, :]) ** 2) ** 2
            c = sum((vts[x, :] - vts[z, :]) ** 2) ** 2
            s = a + b + c
            if s < 0.001:
                areas.append(True)
            else:
                areas.append(False)
        return areas

    def nds_dual(self, body_dual_maps, output_dir):
        self.normal_nds.set_views_from_normal_maps(body_dual_maps)
        # Do optimization with dual normal maps
        nds_mesh = self.normal_nds(output_dir / 'nds_dual')
        return nds_mesh

    def select_canon(self, meshes, smpl):
        best_idx = 0
        best_chamfer = torch.tensor(float('inf'))
        for i in range(len(meshes)):
            temp = chamfer_distance(torch.Tensor(meshes[i].vertices).unsqueeze(0), torch.Tensor(smpl.vertices).unsqueeze(0))[0]
            if temp < best_chamfer:
                best_idx = i
                best_chamfer = temp
        return meshes[best_idx], best_idx
    
    def forward(self, input_frames, input_masks, input_normals, smpl_params, smpl_meshes, posed_meshes=None, canon_meshes=None, output_dir=None, cam_params=None, config=None):
        output_dir = Path(output_dir)
        if self.save_intermediate:
            (output_dir / "normal_F").mkdir(parents=True, exist_ok=True)
            (output_dir / "color_F").mkdir(parents=True, exist_ok=True)
            (output_dir / "nds_dual").mkdir(parents=True, exist_ok=True)
            
        self.normal_nds = NormalNDS(args=config.nds, cam_params=cam_params, device=self.device)

        # set initial mesh from smpl info for normal NDS
        # smpl_mesh, smpl_infos, full_poses, smpl_models = load_smpl_info(smpl_params, self.smpl_type, self.smpl_data, self.tpose) # smpl_mesh : canonical mesh for tpose / posed mesh for x tpose
        smpl_mesh, smpl_infos, full_poses, smpl_models = None, None, None, None
        if not self.tpose:
            smpl_mesh = smpl_meshes # smpl_mesh : posed mesh
        
        if self.use_initial_mesh and self.tpose: # refine initial canonicalized mesh
            initial_mesh, best_idx = self.select_canon(canon_meshes, smpl_mesh)
            if self.flexicubes:
                initial_mesh = optimize.optimize_flexicube(mesh=initial_mesh, iter=300) # voxel_grid_res=64, learning_rate=0.005, iter=500, batch=4, train_res=[512, 512]
        
        else: # refine smpl mesh
            initial_mesh, best_idx = None, None
            if self.flexicubes:
                # re-mesh by using flexicube optimization
                smpl_mesh = optimize.optimize_flexicube(mesh=smpl_mesh, iter=300)
        
        # prepare SMPL-X normal maps for normal NDS
        self.normal_nds.set_initial_mesh(smpl_mesh, smpl_infos, smpl_models, posed_meshes, initial_mesh, config=config)

        angle_to_idx = dict(zip(self.view_angle, range(len(self.view_angle))))
        
        if config.canonfusion.use_mesh:
            # render target normal & color maps from given posed meshes
            tgt_normal_scan_sets, tgt_color_scan_sets = self.normal_nds.render_mesh(smpl_infos, config=config)
        else:
            tgt_normal_scan_sets, tgt_color_scan_sets = self.normal_nds.image_to_render(input_frames, input_masks, input_normals, config=config)
                    
        tgt_scan_images_sets = []
        for (tgt_normal_scan_images, tgt_color_scan_images) in zip(tgt_normal_scan_sets, tgt_color_scan_sets):
            tgt_scan_images = [to_cam(tgt_normal_image, tgt_color_image, int(view_angle)) for (tgt_normal_image, tgt_color_image, view_angle)
                                                                    in zip(tgt_normal_scan_images, tgt_color_scan_images, self.view_angle)]
            tgt_scan_images_sets.append(tgt_scan_images) # (# poses, # view_angles) [normal, color]                
            
        # Dual normal map generation
        scan_dual_maps = {}
        for frame in range(len(input_frames)):
            for view in range(len(input_frames[0])):
                if int(self.view_angle[view]) in scan_dual_maps.keys():
                    scan_dual_maps[int(self.view_angle[view])].append([tgt_scan_images_sets[frame][angle_to_idx[int(self.view_angle[view])]][0], \
                        tgt_scan_images_sets[frame][angle_to_idx[int(self.view_angle[view])]][1], full_poses[frame], smpl_infos[frame], smpl_models[frame]])

                else:
                    # scan_dual_maps[int(self.view_angle[view])] = [[tgt_scan_images_sets[frame][angle_to_idx[int(self.view_angle[view])]][0], \
                    #     tgt_scan_images_sets[frame][angle_to_idx[int(self.view_angle[view])]][1], full_poses[frame], smpl_infos[frame], smpl_models[frame]]] # tgt_scan_images_sets[i] -- [normal, color]
                    scan_dual_maps[int(self.view_angle[view])] = [[tgt_scan_images_sets[frame][angle_to_idx[int(self.view_angle[view])]][0], \
                                                                   tgt_scan_images_sets[frame][angle_to_idx[int(self.view_angle[view])]][1], None, None, None]]
        if self.save_intermediate:
            for angle in scan_dual_maps.keys():
                for i, [normal_map, color_map, pose, _, _] in enumerate(scan_dual_maps[angle]):
                    to_pil(normal_map).save(os.path.join(output_dir, 'normal_F', f'{angle:03d}' + '_' + str(i) + '.png'))
                    to_pil_color(color_map).save(os.path.join(output_dir, 'color_F', f'{angle:03d}' + '_' + str(i) + '.png'))

        # NDS with dual normal map
        if config.canonfusion.tpose:
            nds_mesh_dual, lbs = self.nds_dual(scan_dual_maps, output_dir)
            with open(str(output_mesh_path).replace('.obj', '.pkl'), 'wb') as f:
                pickle.dump(lbs, f)
        else:
            nds_mesh_dual = self.nds_dual(scan_dual_maps, output_dir)
        output_mesh_path = output_dir / f'mesh_dual.obj'
        write_mesh(output_mesh_path, nds_mesh_dual)
        return output_mesh_path
