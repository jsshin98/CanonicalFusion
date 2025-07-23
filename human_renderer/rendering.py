import os
import cv2
import torch.utils.data
import torch
import torch.nn as nn
import numpy as np
import random
import math
import nr_renderer as nr
import trimesh

from renderer.mesh import load_obj_mesh, load_obj_mesh2
from renderer.camera import Camera
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor


class NeuralRenderer(nn.Module):
    def __init__(self,
                 save_root='workspace/save_root',
                 data_name='ioys',
                 cam_param=None,
                 view_idx=[0, 180],
                 pitch=[0],
                 device=torch.device("cuda:0"),
                 **kwargs):
        super(NeuralRenderer, self).__init__()

        self.save_image = False
        self.save_root = save_root
        self.data_name = data_name
        self.view_idx = view_idx
        self.pitch = pitch
        self.eps = 1e-9
        self.cam_param = cam_param
        self.cam = Camera(width=self.cam_param['width'],
                          height=self.cam_param['height'],
                          projection='perspective')
        self.cam.near = self.cam_param['near']
        self.cam.far = self.cam_param['far']
        self.cam.focal_x = self.cam_param['fx']
        self.cam.focal_y = self.cam_param['fy']
        self.cam.principal_x = self.cam_param['px']
        self.cam.principal_y = self.cam_param['py']
        random_dist = np.random.randint(self.cam_param['cmin'],
                                        self.cam_param['cmax'], 1)
        self.cam.center = np.array([0, 0, random_dist[0]])

        os.makedirs(os.path.join(self.save_root, 'GEO', 'OBJ', self.data_name), exist_ok=True)
        os.makedirs(os.path.join(self.save_root, 'RENDER', self.data_name), exist_ok=True)
        os.makedirs(os.path.join(self.save_root, 'MASK', self.data_name), exist_ok=True)
        os.makedirs(os.path.join(self.save_root, 'DEPTH', self.data_name), exist_ok=True)
        os.makedirs(os.path.join(self.save_root, 'NORMAL', self.data_name), exist_ok=True)
        os.makedirs(os.path.join(self.save_root, 'PARAM', self.data_name), exist_ok=True)
        os.makedirs(self.save_root, exist_ok=True)
        if device is not None:
            self.device = device
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

    def make_rotate(self, rx, ry, rz):
        sinX = np.sin(rx)
        sinY = np.sin(ry)
        sinZ = np.sin(rz)

        cosX = np.cos(rx)
        cosY = np.cos(ry)
        cosZ = np.cos(rz)

        Rx = np.zeros((3, 3))
        Rx[0, 0] = 1.0
        Rx[1, 1] = cosX
        Rx[1, 2] = -sinX
        Rx[2, 1] = sinX
        Rx[2, 2] = cosX

        Ry = np.zeros((3, 3))
        Ry[0, 0] = cosY
        Ry[0, 2] = sinY
        Ry[1, 1] = 1.0
        Ry[2, 0] = -sinY
        Ry[2, 2] = cosY

        Rz = np.zeros((3, 3))
        Rz[0, 0] = cosZ
        Rz[0, 1] = -sinZ
        Rz[1, 0] = sinZ
        Rz[1, 1] = cosZ
        Rz[2, 2] = 1.0

        R = np.matmul(np.matmul(Rz, Ry), Rx)
        return R

    def get_normal(self, x, normalize=True, cut_off=0.2):
        def gradient_x(img):
            img = torch.nn.functional.pad(img, (0, 0, 1, 0), mode="replicate")  # pad a column to the end
            gx = img[:, :, :-1, :] - img[:, :, 1:, :]
            return gx

        def gradient_y(img):
            img = torch.nn.functional.pad(img, (0, 1, 0, 0), mode="replicate")  # pad a row on the bottom
            gy = img[:, :, :, :-1] - img[:, :, :, 1:]
            return gy

        def normal_from_grad(grad_x, grad_y, depth):
            grad_z = torch.ones_like(grad_x) / 255.0
            n = torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2) + torch.pow(grad_z, 2))
            normal = torch.cat((grad_y / n, grad_x / n, grad_z / n), dim=1)
            normal += 1
            normal /= 2
            if normalize is False:  # false gives 0~255, otherwise 0~1.
                normal *= 255

            # remove normals along the object discontinuities and outside the object.
            normal[depth.repeat(1, 3, 1, 1) < cut_off] = 0
            return normal

        if x is None:
            return None

        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        x = x.float()
        grad_x = gradient_x(x)
        grad_y = gradient_y(x)

        if x.shape[1] == 1:
            return normal_from_grad(grad_x, grad_y, x)
        else:
            normal = [normal_from_grad
                      (grad_x[:, k, :, :].unsqueeze(1), grad_y[:, k, :, :].unsqueeze(1), x[:, k, :, :].unsqueeze(1)) for
                      k in range(x.shape[1])]
            return torch.cat(normal, dim=1)

    def forward(self, mesh_file, text_file):
        if not os.path.isfile(os.path.join(self.save_root, 'GEO', 'OBJ', self.data_name, self.data_name + '.obj')):
            texture_image = cv2.imread(text_file)
            texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)
            texture = np.flip(texture_image, axis=0)
            mesh, _ = load_obj_mesh2(mesh_file, texture=texture, with_normal=True, with_texture=True)
            mesh.export(os.path.join(self.save_root, 'GEO', 'OBJ', self.data_name, self.data_name + '.obj'))

        m = trimesh.load_mesh(os.path.join(self.save_root, 'GEO', 'OBJ',
                                           self.data_name, self.data_name + '.obj'), process=False)
        vertices = m.vertices
        vmin = vertices.min(0)
        vmax = vertices.max(0)

        up_axis = 1 if (vmax - vmin).argmax() == 1 else 2
        center = np.median(vertices, 0)
        center[up_axis] = 0.5 * (vmax[up_axis] + vmin[up_axis])
        scale = 180 / (vmax[up_axis] - vmin[up_axis])
        vertices -= center
        # vertices += self.eps
        vertices *= scale

        faces = torch.tensor(m.faces[None, :, :].copy()).float().cuda()
        textr = torch.tensor(m.visual.face_colors[None, :, -2:-5:-1].copy()).float().cuda() / 255.0
        textr = textr.unsqueeze(2).unsqueeze(2).unsqueeze(2)

        for p in self.pitch:
            for vid in self.view_idx:
                # angle = np.matmul(self.make_rotate(math.radians(p), 0, 0),
                #                   self.make_rotate(0, math.radians(vid), 0))
                R_delta = self.make_rotate(0, math.radians(vid), 0)
                R_0, K, t, projection_matrix, model_view_matrix = self.cam.get_gl_matrix()
                R = np.matmul(R_0, R_delta)
                K = torch.tensor(K[None:, :].copy()).float().cuda().unsqueeze(0)
                R = torch.tensor(R[None:, :].copy()).float().cuda().unsqueeze(0)
                t = torch.tensor(t[None, :].copy()).float().cuda().unsqueeze(0)
                # R[0, 2, 2]*=(-1) # back view

                renderer = nr.Renderer(image_size=self.cam_param['width'],
                                       orig_size=self.cam_param['width'],
                                       K=K, R=R, t=t,
                                       dist_coeffs=torch.cuda.FloatTensor([[self.cam_param['distx'],
                                                                            self.cam_param['disty'],
                                                                            0.,
                                                                            0.,
                                                                            0.]]),
                                       anti_aliasing=False,
                                       camera_direction=[0, 0, -1],
                                       camera_mode='projection',
                                       viewing_angle=0,
                                       light_intensity_directional=0.9,
                                       light_intensity_ambient=0.8,
                                       near=self.cam.near, far=self.cam.far)

                verts = torch.Tensor(vertices).unsqueeze(0).cuda()
                images_out, depth_out, silhouette_out = renderer(verts, faces, textr)

                normal_out = self.get_normal(depth_out)
                image = images_out.squeeze().permute(2, 1, 0).detach().cpu().numpy()
                image = np.flip(np.rot90(image, -1), 1)
                depth = depth_out.squeeze().detach().cpu().numpy()
                normal = normal_out.squeeze().permute(1, 2, 0).detach().cpu().numpy()
                silhouette = silhouette_out.squeeze().detach().cpu().numpy()
                depth[silhouette == 0] = 0
                normal[silhouette==0, :] = 0

                dic = {'ortho_ratio': self.cam.ortho_ratio,
                       'scale': scale,
                       'center': center,
                       'R': R}

                image_file = os.path.join(self.save_root, 'RENDER', self.data_name, '%d_%d_%02d.png' % (vid, p, 0))
                silhouette_file = os.path.join(self.save_root, 'MASK', self.data_name, '%d_%d_%02d.png' % (vid, p, 0))
                depth_file = os.path.join(self.save_root, 'DEPTH', self.data_name, '%d_%d_%02d.png' % (vid, p, 0))
                normal_file = os.path.join(self.save_root, 'NORMAL', self.data_name, '%d_%d_%02d.png' % (vid, p, 0))

                np.save(os.path.join(self.save_root, 'PARAM', self.data_name, '%d_%d_%02d.npy' % (vid, p, 0)), dic)
                cv2.imwrite(image_file, (image * 255.0))
                cv2.imwrite(depth_file, (depth*64).astype(np.uint16))
                cv2.imwrite(silhouette_file, (silhouette * 255))
                cv2.imwrite(normal_file, (normal * 255.0))
                print(self.data_name, vid)