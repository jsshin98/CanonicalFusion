import random
import os
import re
import glob
import pickle
import numpy as np
import cv2
import collections
import trimesh
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from depth_predictor.utils.eval.evaluator_sample import *
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from depth_predictor.utils.loader_utils import *
from depth_predictor.utils.core import depth2volume

class HumanRecon(nn.Module):
    """Implementation of single-stage SMPLify."""
    def __init__(self,
                 result_path='',
                 ckpt_path='',
                 half_input=False,
                 half_output=False,
                 center_crop=True,
                 res=512,
                 voxel_size=512,
                 learning_rate=1e-3,
                 start_epoch=1,
                 model_name='',
                 eval_metrics=None,
                 device=torch.device('cuda')):
        super(HumanRecon, self).__init__()

        self.result_path = result_path
        self.half_input = half_input
        self.half_output = half_output
        self.center_crop = center_crop
        self.res = res
        self.voxel_size = voxel_size
        self.eval_metrics = eval_metrics
        self.device = device

        # load pre-trained model
        self.model = getattr(models, model_name)(half_input=self.half_input,
                                                 half_output=self.half_output,
                                                 split_last=True)
        self.model.to(self.device)
        self.model.eval()
        optimizer_G = torch.optim.Adam(self.model.parameters(), learning_rate)
        recon_model, optimizer_G, start_epoch = \
            self.load_checkpoint([ckpt_path],
                                  self.model, optimizer_G, start_epoch,
                                  is_evaluate=False, device=device)

        self.RGB_MEAN = [0.485, 0.456, 0.406]
        self.RGB_STD = [0.229, 0.224, 0.225]
        self.RGB_MAX = [255.0, 255.0, 255.0]
        self.RGB_MG = [10.0, 10.0, 10.0]
        os.makedirs(self.result_path, exist_ok=True)

    def load_checkpoint(self, model_paths, model,
                        optimizer, start_epoch,
                        is_evaluate=False, device=None):

        for model_path in model_paths:
            items = glob.glob(os.path.join(model_path, '*.pth.tar'))
            items.sort()

            if len(items) > 0:
                if is_evaluate is True:
                    model_path = os.path.join(model_path, 'model_best.pth.tar')
                else:
                    if len(items) == 1:
                        model_path = items[0]
                    else:
                        model_path = items[len(items) - 1]

                print(("=> loading checkpoint '{}'".format(model_path)))
                checkpoint = torch.load(model_path, map_location=device)
                start_epoch = checkpoint['epoch'] + 1

                if hasattr(model, 'module'):
                    model_state_dict = checkpoint['model_state_dict']
                else:
                    model_state_dict = collections.OrderedDict(
                        {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})

                model.load_state_dict(model_state_dict, strict=False)
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print('=> generator optimizer has been loaded')
                except:
                    print('=> optimizer(g) not loaded (trying to train a new network?)')

                print(("=> loaded checkpoint (resumed epoch is {})".format(start_epoch)))
                return model, optimizer, start_epoch

        print(("=> no checkpoint found at '{}'".format(model_path)))
        return model, optimizer, start_epoch

    def evaluate(self, input_var, model, data_idx=None, angle=None,
                 save_images=False, device=None):
        if save_images:
            path_pred = \
                os.path.join(self.result_path, 'render_mesh', 'data%d' % data_idx)
            path_pred_depth = \
                os.path.join(self.result_path, 'render_depth', 'data%d' % data_idx)
            path_pred_depth2normal = \
                os.path.join(self.result_path, 'render_depth2normal', 'data%d' % data_idx)
            path_pred_color = \
                os.path.join(self.result_path, 'render_color', 'data%d' % data_idx)
            path_pred_normal = \
                os.path.join(self.result_path, 'render_normal', 'data%d' % data_idx)
            os.makedirs(path_pred, exist_ok=True)
            os.makedirs(path_pred_depth, exist_ok=True)
            os.makedirs(path_pred_depth2normal, exist_ok=True)
            os.makedirs(path_pred_color, exist_ok=True)
            os.makedirs(path_pred_normal, exist_ok=True)
        model.eval()
        evaluator = HumanEvaluator()

        with torch.no_grad():
            pred_var = model(input_var.unsqueeze(0))
            if 'color' in self.eval_metrics and 'pred_color' in pred_var[0]:
                pred_color = pred_var[0]['pred_color']
                pred_color = torch.chunk(pred_color, chunks=(pred_color.shape[1] // 3), dim=1)
            else:
                pred_color = [None, None, None, None]

            if 'normal' in self.eval_metrics and 'pred_normal' in pred_var[0]:
                pred_normal = pred_var[0]['pred_normal']
                pred_normal = torch.chunk(pred_normal, chunks=(pred_normal.shape[1] // 3), dim=1)
            else:
                pred_normal = [None, None, None, None]

            if 'color' in self.eval_metrics:
                if 'color_visualize' in self.eval_metrics:
                    target_color = None
                    image = evaluator.visualize_color(pred_color, target_color,
                                                       save_img=save_images, pred_path=path_pred_color,
                                                       tgt_path=None, data_idx=data_idx, angle=angle)
            if 'normal' in self.eval_metrics:
                if 'normal_visualize' in self.eval_metrics:
                    target_normal = None
                    evaluator.visualize_normal(pred_normal, target_normal,
                                               save_img=save_images, pred_path=path_pred_normal,
                                               tgt_path=None, data_idx=data_idx, angle=angle)

            if 'depth' in self.eval_metrics:
                if 'depth_visualize' in self.eval_metrics:
                    target_depth = None
                    pred_depth = pred_var[1]['pred_depth']
                    pred_depth = torch.chunk(pred_depth, chunks=(pred_depth.shape[1]), dim=1)
                    evaluator.visualize_depth(pred_depth, target_depth,
                                              save_img=save_images, pred_path=path_pred_depth,
                                              pred_depth2normal_path=path_pred_depth2normal,
                                              tgt_depth2normal_path=None, tgt_path=None,
                                              data_idx=data_idx, angle=angle)

            if 'mesh' in self.eval_metrics:
                depth_pred = []
                for idx in range(len(pred_depth)):
                    if pred_depth[idx] is not None:
                        depth_pred.append(pred_depth[idx])

                if len(depth_pred) == 2:
                    pred_front_depth = depth_pred[0]
                    pred_back_depth = depth_pred[1]

                    pred_front_depth[pred_front_depth < 0] = 0
                    pred_back_depth[pred_back_depth < 0] = 0

                    pred_volume = depth2occ_2view_torch(
                        pred_front_depth, pred_back_depth, device=self.device,
                        binarize=False, voxel_size=self.voxel_size)

                    src_volume = pred_volume.squeeze(0).detach().cpu().numpy()
                    src_front = evaluator.tensor2np_color(pred_color[0], save_img=False, dir='front')
                    src_back = evaluator.tensor2np_color(pred_color[1], save_img=False, dir='back')

                    pred_mesh, src_model_color = colorize_model(src_volume, src_back, src_front)
                    pred_mesh.vertices -= pred_mesh.bounding_box.centroid
                    pred_mesh.vertices *= 2 / np.max(pred_mesh.bounding_box.extents)
                    pred_mesh = self.postprocess_mesh(pred_mesh)
                    pred_mesh.export(path_pred + '/mesh_%d_%d.obj' % (idx, angle))
                else:
                    pred_front_depth = depth_pred[0]
                    pred_back_depth = depth_pred[1]
                    pred_left_depth = depth_pred[2]
                    pred_right_depth = depth_pred[3]
                    pred_volume = depth2occ_4view_torch(
                        pred_front_depth, pred_back_depth,
                        pred_left_depth, pred_right_depth,
                        device=device, binarize=False, voxel_size=self.voxel_size)

            return pred_mesh, image

        # save results
        # evaluator.save_results(self.model_path, self.start_epoch)

    def postprocess_mesh(self, mesh, num_faces=None):
        """Post processing mesh by removing small isolated pieces.

        Args:
            mesh (trimesh.Trimesh): input mesh to be processed
            num_faces (int, optional): min face num threshold. Defaults to 4096.
        """
        total_num_faces = len(mesh.faces)
        if num_faces is None:
            num_faces = total_num_faces // 100
        cc = trimesh.graph.connected_components(
            mesh.face_adjacency, min_len=3)
        mask = np.zeros(total_num_faces, dtype=np.bool)
        cc = np.concatenate([
            c for c in cc if len(c) > num_faces
        ], axis=0)
        mask[cc] = True
        mesh.update_faces(mask)

        return mesh

    def forward(self, images, idx, angle):
        pred_meshes = []
        pred_images = []
        for i in range(len(images)):
            if not images[i].shape[0] == 512:
                image = cv2.resize(images[i], dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
            else:
                image = images[i]

            if self.center_crop is True:
                width = image.shape[1]
                offset = np.int(width / 4)
                image = image[:, offset:width - offset, :]

            image = torch.Tensor(image).permute(2, 0, 1).float()
            image = image + torch.Tensor(self.RGB_MG).view(3, 1, 1)
            image = image / torch.Tensor(self.RGB_MAX).view(3, 1, 1)
            image = (image - torch.Tensor(self.RGB_MEAN).view(3, 1, 1)) \
                    / torch.Tensor(self.RGB_STD).view(3, 1, 1)

            if self.device is not None:
                if image is not None:
                    input_var = image.to(self.device)
            if input_var is not None:
                input_var = torch.autograd.Variable(input_var)

            pred_mesh, pred_image = self.evaluate(input_var, self.model,
                                                  data_idx=idx,
                                                  angle=angle[i],
                                                  save_images=True,
                                                  device=self.device)
            pred_meshes.append(pred_mesh)
            pred_images.append(pred_image)

        return pred_meshes, pred_images


class Renderer(nn.Module):
    """Implementation of single-stage SMPLify."""
    def __init__(self,
                 result_path='',
                 fov=45,
                 res=512,
                 angle=None,
                 axis='x',
                 device=torch.device('cuda')):
        super(Renderer, self).__init__()

        self.result_path = result_path
        self.res = res
        self.fov = fov
        self.angle = angle
        self.axis = axis
        self.device = device

        self.RGB_MEAN = [0.485, 0.456, 0.406]
        self.RGB_STD = [0.229, 0.224, 0.225]
        self.RGB_MAX = [255.0, 255.0, 255.0]
        os.makedirs(self.result_path, exist_ok=True)

    def get_pers_imgs(self, mesh, scene, res, fov):
        scene.camera.resolution = [res, res]
        scene.camera.fov = fov * (scene.camera.resolution /
                                  scene.camera.resolution.max())
        # scene.camera_transform[0:3, 3] = 0.0
        # scene.camera_transform[2, 3] = 1.0
        pers_origins, pers_vectors, pers_pixels = scene.camera_rays()
        pers_points, pers_index_ray, pers_index_tri = mesh.ray.intersects_location(
            pers_origins, pers_vectors, multiple_hits=True)
        pers_depth = trimesh.util.diagonal_dot(pers_points - pers_origins[0],
                                               pers_vectors[pers_index_ray])
        pers_colors = mesh.visual.face_colors[pers_index_tri]

        pers_pixel_ray = pers_pixels[pers_index_ray]
        pers_depth_far = np.zeros(scene.camera.resolution, dtype=np.float32)
        pers_color_far = np.zeros((res, res, 3), dtype=np.float32)

        pers_depth_near = np.ones(scene.camera.resolution, dtype=np.float32) * res
        pers_color_near = np.zeros((res, res, 3), dtype=np.float32)

        denom = np.tan(np.radians(fov) / 2.0) * 5
        # pers_depth_int = (pers_depth - 3.5)*(res/denom) + res / 2
        pers_depth_int = (pers_depth - np.mean(pers_depth)) * (res / denom) + res / 2

        for k in range(pers_pixel_ray.shape[0]):
            u, v = pers_pixel_ray[k, 0], pers_pixel_ray[k, 1]
            if pers_depth_int[k] > pers_depth_far[v, u]:
                pers_color_far[v, u, ::-1] = pers_colors[k, 0:3] / 255.0
                pers_depth_far[v, u] = pers_depth_int[k]
            if pers_depth_int[k] < pers_depth_near[v, u]:
                pers_depth_near[v, u] = pers_depth_int[k]
                pers_color_near[v, u, ::-1] = pers_colors[k, 0:3] / 255.0

        pers_depth_near = pers_depth_near * (pers_depth_near != res)
        pers_color_near = np.flip(pers_color_near, 0)
        pers_depth_near = np.flip(pers_depth_near, 0)
        pers_color_far = np.flip(pers_color_far, 0)
        pers_depth_far = np.flip(pers_depth_far, 0)

        return pers_color_near, pers_depth_near, pers_color_far, pers_depth_far

    def rotate_mesh(self, mesh, angle, axis='x'):
        vertices = mesh.vertices
        vertices_re = (np.zeros_like(vertices))
        if axis == 'y':  # pitch
            rotation_axis = np.array([1, 0, 0])
        elif axis == 'x':  # yaw
            rotation_axis = np.array([0, 1, 0])
        elif axis == 'z':  # roll
            rotation_axis = np.array([0, 0, 1])
        else:  # default is x (yaw)
            rotation_axis = np.array([0, 1, 0])

        for i in range(vertices.shape[0]):
            vec = vertices[i, :]
            rotation_degrees = angle
            rotation_radians = np.radians(rotation_degrees)

            rotation_vector = rotation_radians * rotation_axis
            rotation = R.from_rotvec(rotation_vector)
            rotated_vec = rotation.apply(vec)
            vertices_re[i, :] = rotated_vec
        rot_mesh = trimesh.Trimesh(vertices=vertices_re, faces=mesh.faces, vertex_colors=mesh.visual.vertex_colors)
        return rot_mesh

    def forward(self, mesh, idx):
        mesh_list = []
        image_list = []

        mesh.vertices -= mesh.bounding_box.centroid
        mesh.vertices *= 2 / np.max(mesh.bounding_box.extents)

        result_img_path = os.path.join(self.result_path, 'render_color', 'data%d' % idx)
        result_mesh_path = os.path.join(self.result_path, 'render_mesh', 'data%d' % idx)
        os.makedirs(result_img_path, exist_ok=True)
        os.makedirs(result_mesh_path, exist_ok=True)

        for x in self.angle:
            rot_mesh = self.rotate_mesh(mesh, x, axis=self.axis)
            mesh_list.append(rot_mesh)

            scene = rot_mesh.scene()
            scene.camera.resolution = [self.res, self.res]
            pers_color_front, pers_depth_front, pers_color_back, pers_depth_back = \
                self.get_pers_imgs(rot_mesh, scene, self.res, self.fov)
            image_list.append(pers_color_front)
            cv2.imwrite(result_img_path + '/color_%d_%d.png' % (idx, x), (pers_color_front * 255).astype(np.int))
            # rot_mesh.vertices -= rot_mesh.bounding_box.centroid
            # rot_mesh.vertices *= 2 / np.max(rot_mesh.bounding_box.extents)
            rot_mesh.export(result_mesh_path + '/mesh_%d_%d.obj' % (idx, x))
        return mesh_list, image_list

def set_save_dirs(dict_path):
    for key in dict_path.keys():
        os.makedirs(dict_path[key], exist_ok=True)

def save_results(data, path2save=None, type='image'):
    # type = (image, mesh, dictionary, npz)
    if 'image' in type:
        for k, image in enumerate(data):
            f_name = os.path.join(path2save, f'%s_%d.png' % (type, k))
            cv2.imwrite(f_name, (image * 255.0).astype(np.int) )
    elif 'mesh' in type:
        for k, mesh in enumerate(data):
            f_name = os.path.join(path2save, f'%s_%d.obj' % (type, k))
            mesh.export(f_name)
    elif type == 'opt':
        for k, params in enumerate(data):
            f_name = os.path.join(path2save, f'opt_params_%d.npz' % k)
            np.savez_compressed(f_name, **params)
    elif type == 'pickle':
        with open(os.path.join(path2save, f'avatars.pkl'), "wb") as pkl_file:
            pickle.dump(data, pkl_file)

def data_loader(path2image, path2mesh):
    image_list_front = glob.glob(path2image + '/image_front_*.png')
    image_list_back = glob.glob(path2image + '/image_back_*.png')
    mesh_list = glob.glob(path2mesh + '/*.obj')

    image_list_front = sorted(image_list_front)
    image_list_back = sorted(image_list_back)
    mesh_list = sorted(mesh_list)

    meshes = []
    images_front = []
    images_back = []

    for image in image_list_front:
        img = cv2.imread(image, 1)
        img = img.astype(np.float)/255.0
        images_front.append(img)

    for image in image_list_back:
        img = cv2.imread(image, 1)
        img = img.astype(np.float)/255.0
        images_back.append(img)

    for mesh in mesh_list:
        meshes.append(trimesh.load(mesh, processing=False, maintain_order=True))  # scanned model.

    return meshes, images_front, images_back
