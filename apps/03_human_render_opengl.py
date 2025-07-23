
import os
import yaml
import torch
from apps.options import Configurator
from human_renderer.human_renderer import GLRenderer

####################################################
# Usage Guide (Jan. 16, 2024)
# ------------------------------------------------
# Features:
# > Can render depth maps, color images, normal maps from .obj files
# > .obj files can be scan model or smpl model (option: rendering_mode)
# > Can change renderer (option: renderer = 'trimesh', 'nr', 'opengl', etc)
# > To run the code, set dataset, paths to source(obj) and result directories
#####################################################

if __name__=='__main__':
    config = Configurator()
    params = config.parse()
    mode = 'GT_RENDER'  # select one of 'GT_RENDER', 'CES', 'VAE'
    is_train = True
    # data_name = ['RP', 'TH2.1'] # for research
    data_name = ['IOYS_T']
    # data_name = ['TH2.1', 'IOYS_T', 'IOYS_4090']
    # data_name = ['TH2.1']
    # data_name = ['RP', 'TH2.1', 'IOYS_T', 'IOYS_4090']
    for dataset  in data_name:
        if is_train:
            # dataset = 'TH2.1'
            root_dir = '/home/xxx/data/IOYS_Famoz/OBJ'
            save_dir = '/home/xxx/data/IOYS_Famoz'
        else:
            # dataset = 'STUDIO_SET6'
            root_dir = '/home/xxx/data2/IOYS_Famoz'
            save_dir = '/home/xxx/data2/IOYS_Famoz'

        config_file = '../config_render/config_keti2024.yaml'
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            cam_params = config['CAM']
            render_params = config[mode]

        if is_train:
            params.path2obj = os.path.join(root_dir, dataset, 'MESH')  # input OBJ data path
            params.path2smpl = os.path.join(root_dir, dataset, 'SMPLX')  # gt smpl params
        else:
            params.path2obj = os.path.join(root_dir, dataset, 'IMG')  # input OBJ data path
            params.path2smpl = os.path.join(root_dir, dataset, 'SMPLX/INIT')  # gt smpl params

        if mode == 'GT_RENDER':
            params.predict_gender = True
            params.path2save = os.path.join(save_dir, mode + '_' + dataset)
        else:
            params.predict_gender = False
            params.path2save = os.path.join(save_dir, 'DATASET_2024', dataset, mode)
        params.path2light = os.path.join(params.path2sh, 'natural_saito.npy')  # env. lights

        skip_exist = False
        gl_renderer = GLRenderer(params=params,
                                 cam_params=cam_params,
                                 render_params=render_params,
                                 skip_exist=skip_exist,
                                 device=torch.device("cuda:0"))

        # offset: the index of starting directory (0 means from the first directory)
        gl_renderer(a_min=0)
