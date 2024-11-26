import os
import torch
import json
import trimesh
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2

class DataLoader(Dataset):
    def __init__(self,
                 config,
                 smpl_root='resource/smpl_models',
                 res=512):

        use_semantic = True
        if use_semantic:
            path2semantic = os.path.join(smpl_root, config['segmentation'])
            if os.path.exists(path2semantic):
                self.v_label = init_semantic_labels(path2semantic)

    def load_image(path2image, pred_res=None):
        img = np.array(Image.open(path2image))
        if not img.shape[1] == pred_res:
            img = cv2.resize(img, (pred_res, pred_res), interpolation=cv2.INTER_NEAREST) #AREA
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # mask[mask > 0] = 1
        # mask = np.expand_dims(mask, axis=2)
        return img#, path2image
    
    def load_normal(self):
        pass

    def load_mask(path2mask, pred_res=None):
        mask = np.array(Image.open(path2mask))
        if not mask.shape[1] == pred_res:
            mask = cv2.resize(mask, (pred_res, pred_res), interpolation=cv2.INTER_NEAREST) # AREA
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask[mask > 0] = 1
        mask = np.expand_dims(mask, axis=2)
        return mask
    
    def load_depth(path2depth, pred_res=None):
        depth = np.array(Image.open(path2depth)).astype('float32')
        depth = cv2.medianBlur(depth, 3)
        if not depth.shape[1] == pred_res:
            depth = cv2.resize(depth, (pred_res, pred_res), interpolation=cv2.INTER_NEAREST) # AREA

        depth = (depth - 32767.0) / 128.0 / 128.0 + 0.5
        depth = np.clip(depth, a_min=0, a_max=1)
        depth_front = np.expand_dims(depth, axis=2)

        depth = np.array(Image.open(path2depth.replace('front', 'back'))).astype('float32')
        depth = cv2.medianBlur(depth, 3)
        if not depth.shape[1] == pred_res:
            depth = cv2.resize(depth, (pred_res, pred_res), interpolation=cv2.INTER_NEAREST) # AREA

        depth = (depth - 32767.0) / 128.0 / 128.0 + 0.5
        depth = np.clip(depth, a_min=0, a_max=1)
        depth_back = np.expand_dims(depth, axis=2)

        return depth_front, depth_back
    
    # @staticmethod
    # def load_gt_mesh(path2obj):
    #     vertices, faces, textures, texture_image = load_gt_data(path2obj)
    #     return vertices, faces, textures, texture_image

    @staticmethod
    def load_smpl_mesh(path2obj):
        m = trimesh.load(path2obj, process=False)
        # return m
        return m.vertices, m.faces, m.visual.uv

    @staticmethod
    def load_smpl_params(path2json):
        with open(path2json, "r") as f:
            smpl_params = json.load(f)
        for key in smpl_params.keys():
            if isinstance(smpl_params[key], list):
                smpl_params[key] = torch.FloatTensor(smpl_params[key]).reshape(1, -1)
            elif isinstance(smpl_params[key], float):
                smpl_params[key] = torch.FloatTensor([smpl_params[key]]).reshape(1, -1)

        return smpl_params

def load_smpl_info(path2param, path2mesh=None):
    with open(path2param, 'r') as f:
        smpl_params = json.load(f)
    for key in smpl_params.keys():
        smpl_params[key] = torch.FloatTensor(smpl_params[key]).reshape(1, -1)

    if path2mesh is not None:
        smpl_mesh = trimesh.load(path2mesh, process=False)
        return smpl_params, smpl_mesh
    else:
        return smpl_params

def init_semantic_labels(path2label):
    """
    Set semantic labels for SMPL(-X) vertices
    :param path2label: path to the semantic information (json file)
    :results are saved in instance variables
    """
    # semantic labels for smplx vertices.
    if os.path.isfile(path2label):
        left_wrist_idx, right_wrist_idx = [], []
        hand_idx, non_hand_idx = [], []
        with open(path2label, "r") as json_file:
            v_label = json.load(json_file)
            v_label['leftWrist'], v_label['rightWrist'] = [], []
            v_label['hand_idx'], v_label['non_hand_idx'] = [], []
            v_label['left_wrist_idx'], v_label['right_wrist_idx'] = [], []

            for k in v_label['leftHand']:
                if k in v_label['leftForeArm']:
                    v_label['left_wrist_idx'].append(k)
            for k in v_label['rightHand']:
                if k in v_label['rightForeArm']:
                    v_label['right_wrist_idx'].append(k)
            for key in v_label.keys():
                if 'leftHand' in key or 'rightHand' in key:
                    v_label['hand_idx'].append(v_label[key])
                else:
                    v_label['non_hand_idx'].append(v_label[key])

            nonbody_idx = v_label['head'] + v_label['eyeballs'] + \
                               v_label['leftToeBase'] + v_label['rightToeBase'] + \
                               v_label['leftEye'] + v_label['rightEye'] + \
                               hand_idx
            nonbody_idx = np.asarray(list(set(nonbody_idx)))  # unique idx
            body_idx = np.asarray([i for i in range(0, 10475) if i not in nonbody_idx])
            v_label['body_idx'] = body_idx
            v_label['non_body_idx'] = nonbody_idx
        return v_label