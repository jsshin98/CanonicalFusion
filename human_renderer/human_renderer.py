import os
import cv2
import math
import json
import random
import trimesh
import collections
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import time
import glob
import pickle
import open3d as o3d
from multiprocessing.pool import ThreadPool as Pool
from utils.im_utils import get_plane_params
from human_renderer.renderer.mesh import load_file2info, compute_tangent_from_normals
from human_renderer.renderer.camera import Camera, make_rotate, rotateSH
from human_renderer.renderer.light import computePRT
from human_renderer.utils import load_textured_mesh
# from utils.visualization import VisOpen3D
from PIL import Image
import os


class GLRenderer(nn.Module):
    def __init__(self,
                 params=None,
                 cam_params=None,
                 render_params=None,
                 skip_exist=True,
                 renderer='opengl',
                 device=torch.device("cuda:0")):
        super(GLRenderer, self).__init__()

        # set paths
        self.path2obj = params.path2obj
        self.path2smpl = params.path2smpl
        self.save_root = params.path2save
        # self.renderer = renderer
        if self.path2obj is not None and len(self.path2obj) > 0:
            self.data_list = sorted(os.listdir(self.path2obj))
        else:
            self.data_list = []

        # load camera parameters
        self.cam_params = cam_params
        self.render_params = render_params
        if self.render_params['uniform_sampling']:
            self.view_idx = [k for k in range(0, 360, self.render_params['interval'])]
        else:
            self.view_idx = self.render_params['view_idx']

        self.pitch = self.render_params['pitch']
        self.num_light = self.render_params['num_lights']
        self.rendering_method = self.render_params['renderer']
        self.h_min = self.render_params['h_range'][0]
        self.h_max = self.render_params['h_range'][-1]
        self.uniform_light = False
        if 'uniform_light' in self.render_params and self.render_params['uniform_light']:
            self.uniform_light = True

        # set spherical harmonics.
        self.shs = np.load(params.path2light)  # natural light by S. Saito.
        self.random_sh = 0.0  # set this to False, if env_sh.npy file exists
        if 'random' in self.render_params['lighting']:
            idx = self.render_params['lighting'].index('random')
            self.random_sh = self.render_params['lighting_prob'][idx]

        # set cam parameters.
        width, height, projection = (
            self.cam_params['width'], self.cam_params['height'], self.cam_params['projection'])
        self.avg_height = render_params['default_height']  # render_params['h_range']

        self.cam = Camera(width=width,
                          height=height,
                          projection=projection)
        self.cam.center = self.cam_params['cam_center']
        self.cam.dist_params = torch.FloatTensor([self.cam_params['distortion']])
        self.cam.focal_x = self.cam_params['fx']
        self.cam.focal_y = self.cam_params['fy']
        self.cam.principal_x = self.cam_params['px']
        self.cam.principal_y = self.cam_params['py']
        self.dist2cam = self.cam.center[-1]  # object is at the origin.
        self.scale = 1.0
        self.center = np.asarray([0.0, 0.0, 0.0])
        self.projection = self.cam_params['projection']
        self.INT15_MAX = 32767

        # set device.
        if device is not None:
            self.device = device
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # support four different types of rendering scenarios.
        self.rndr, self.rndr_uv = None, None
        self.smpl_rndr, self.smpl_rndr_uv = None, None
        self.set_renderer()
        if self.rendering_method == 'opengl':
            self.render_images = self.render
        elif self.rendering_method == 'nr':
            self.render_images = self.render_nr
        else:
            assert 'Undefined renderer, set opengl or nr'

        self.skip_exist = skip_exist

        # set gender predictor.
        self.gender_predictor = None
        self.set_gender_predictor(params.predict_gender)

        # rendering configurations (flags)
        self.xy = self.set_grid_coordinate(width=width, height=height)

    def set_gender_predictor(self, predict_gender=True):
        if predict_gender:
            from utils.gender_predictor import predict_gender
            self.gender_predictor = predict_gender

    def set_grid_coordinate(self, width=512, height=512):
        """
        Set grid coordinates for image
        :param width: width of image
        :param height: height of image
        :return: concatenated coordinates
        """
        x = np.reshape((np.linspace(0, width, width) - int(width/2)) / self.cam.principal_x,
                       [1, 1, -1, 1])
        y = np.reshape((np.linspace(0, height, height) - int(height/2)) / self.cam.principal_y,
                       [1, 1, 1, -1])
        x = np.tile(x, [1, 1, 1, width])
        y = np.tile(y, [1, 1, height, 1])
        return torch.Tensor(np.concatenate((x, y), axis=1))

    def set_dirs(self, data_name, zero123blender=False):
        path_dict = {}
        os.makedirs(self.save_root, exist_ok=True)
        if self.render_params['render_scan']:
            scan_dict = collections.defaultdict(str)
            if self.render_params['scan_diffuse']:
                if self.projection == 'orthographic':
                    scan_dict['diffuse'] = os.path.join(self.save_root, 'COLOR/DIFFUSE_ORTH', data_name)
                else:
                    if zero123blender:
                        scan_dict['diffuse'] = os.path.join(self.save_root, 'IMGS', data_name)
                    else:
                        scan_dict['diffuse'] = os.path.join(self.save_root, 'COLOR/DIFFUSE', data_name)

            if self.render_params['scan_albedo']:
                if self.projection == 'orthographic':
                    scan_dict['albedo'] = os.path.join(self.save_root, 'COLOR/ALBEDO_ORTH', data_name)
                else:
                    scan_dict['albedo'] = os.path.join(self.save_root, 'COLOR/ALBEDO', data_name)
            if self.render_params['scan_param']:
                if zero123blender:
                    scan_dict['param'] = os.path.join(self.save_root, 'PARAMS', data_name)
                else:
                    scan_dict['param'] = os.path.join(self.save_root, 'PARAM/RENDER', data_name)
            if self.render_params['scan_depth']:
                scan_dict['depth'] = os.path.join(self.save_root, 'DEPTH/GT', data_name)
            if self.render_params['scan_depth']:
                scan_dict['normal'] = os.path.join(self.save_root, 'NORMAL/GT', data_name)
            if self.render_params['scan_depth']:
                scan_dict['mask'] = os.path.join(self.save_root, 'MASK/GT', data_name)
            if self.render_params['scan_uv']:
                scan_dict['uv_mask'] = os.path.join(self.save_root, 'UV/MASK/GT', data_name)
                scan_dict['uv_normal'] = os.path.join(self.save_root, 'UV/NORMAL/GT', data_name)
                scan_dict['uv_texture'] = os.path.join(self.save_root, 'UV/TEXTURE/GT', data_name)
                scan_dict['uv_pos'] = os.path.join(self.save_root, 'UV/POS/GT', data_name)
                scan_dict['uv_pkl'] = os.path.join(self.save_root, 'UV/PKL', data_name)
            if not zero123blender:
                for key in scan_dict:
                    os.makedirs(scan_dict[key], exist_ok=True)
            path_dict['path4scan'] = scan_dict

        if self.render_params['render_smpl']:
            smpl_dict = collections.defaultdict(str)
            if self.render_params['smpl_depth']:
                smpl_dict['depth'] = os.path.join(self.save_root, 'DEPTH/GT_SMPLX', data_name)
            if self.render_params['smpl_depth']:
                smpl_dict['normal'] = os.path.join(self.save_root, 'NORMAL/GT_SMPLX', data_name)
            if self.render_params['smpl_depth']:
                smpl_dict['mask'] = os.path.join(self.save_root, 'MASK/GT_SMPLX', data_name)
            if self.render_params['smpl_uv']:
                smpl_dict['uv_mask'] = os.path.join(self.save_root, 'UV/MASK/GT_SMPLX', data_name)
                smpl_dict['uv_normal'] = os.path.join(self.save_root, 'UV/NORMAL/GT_SMPLX', data_name)
                smpl_dict['uv_texture'] = os.path.join(self.save_root, 'UV/TEXTURE/GT_SMPLX', data_name)
                smpl_dict['uv_pos'] = os.path.join(self.save_root, 'UV/POS/GT_SMPLX', data_name)
            for key in smpl_dict:
                os.makedirs(smpl_dict[key], exist_ok=True)
            path_dict['path4smpl'] = smpl_dict
        return path_dict

    @staticmethod
    def postprocess_mesh(mesh, num_faces=None):
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
        mask = np.zeros(total_num_faces, dtype=bool)
        cc = np.concatenate([
            c for c in cc if len(c) > num_faces
        ], axis=0)
        mask[cc] = True
        mesh.update_faces(mask)

        return mesh

    def load_data_trimesh(self, data, height=None):
        obj_path = os.path.join(self.path2obj, data, data + '.ply')
        assert os.path.exists(obj_path), 'file does not exist'

        m = trimesh.load(obj_path)
        rot = [[3.74939946e-33, 1.00000000e+00, -6.12323400e-17],
               [-6.12323400e-17, 6.12323400e-17, 1.00000000e+00],
                [1.00000000e+00, 0.00000000e+00, 6.12323400e-17]]
        vertices_ = m.vertices
        center = np.median(vertices_, 0)
        vertices_ = vertices_ - center
        vertices_ = np.matmul(np.asarray(vertices_), np.asarray(rot).transpose(1, 0))

        up_axis = 1
        vmin = vertices_.min(0)
        vmax = vertices_.max(0)
        center[up_axis] = 0.5 * (vmax[up_axis] + vmin[up_axis])
        if height is None:
            scale = self.avg_height / (vmax[up_axis] - vmin[up_axis])
        else:
            scale = height / (vmax[up_axis] - vmin[up_axis])

        # vertices_ = (vertices_ - center) * scale
        vertices_ = vertices_ * scale
        textr_vts = torch.tensor(m.visual.vertex_colors[None, :, 0:3].copy()).float() / 255.0
        return {'vertices': vertices_, 'faces': m.faces, 'textr_vts': textr_vts,
                'vertex_color': m.visual.vertex_colors, 'face_colors': m.visual.face_colors,
                'center': center, 'scale': scale}

    def load_gt_data(self, data, height=None):
        """
        NOT OPTIMIZED YET
        :param data:
        :param height:
        :return:
        """
        # input data loading
        obj_path = os.path.join(self.path2obj, data, data + '_xatlas.obj')

        if not os.path.exists(obj_path):
            obj_path = os.path.join(self.path2obj, data, data + '.obj')

        if not os.path.exists(obj_path):
            obj_path = glob.glob(os.path.join(self.path2obj, data, '*.obj'))
            if len(obj_path) > 4:  # what was this?
                obj_path = obj_path[3]
        # assert os.path.exists(obj_path), 'file does not exist'
        output = {}

        if len(obj_path) > 0:
            if isinstance(obj_path, list):
                obj_path = obj_path[0]

            m, texture_image = load_textured_mesh(obj_path, data)
            if texture_image is None:
                print('could not find texture map')
                texture_image = np.zeros((512, 512, 3))

            # preprocessing (agisoft)
            mesh_generation = 'legecy'
            if mesh_generation == 'agisoft':
                m = self.postprocess_mesh(m, num_faces=100)

            # for standing mesh (general)
            vertices = m.vertices
            vmin = vertices.min(0)
            vmax = vertices.max(0)
            up_axis = 1
            center = np.median(vertices, 0)
            center[up_axis] = 0.5 * (vmax[up_axis] + vmin[up_axis])
            if height is None:
                scale = self.avg_height / (vmax[up_axis] - vmin[up_axis])
            else:
                scale = height / (vmax[up_axis] - vmin[up_axis])

            # applied to the renderer (instead of vertices)
            self.scale = scale
            self.center = center

            # for rotated texture map. (e.g., 2K2K)
            # texture_image = np.rot90(texture_image, k=1)
            # texture_image = np.flip(texture_image, axis=1)
            # texture_image = np.rot90(texture_image, k=3)

            # for nerf results
            # theta1, theta2, theta3 = 90, 90, 0
            # R1 = make_rotate(np.deg2rad(theta1), 0, 0)
            # R2 = make_rotate(0, np.deg2rad(theta2), 0)
            # R3 = make_rotate(0, 0, np.deg2rad(theta3))
            # rot = np.matmul(np.matmul(R1, R2), R3)
            # vertices_ = m.vertices
            # vertices_ = vertices_ - center
            # vertices_ = np.matmul(np.asarray(vertices_), np.asarray(rot))
            # vertices = vertices_ + center
            #
            # up_axis = 1
            # center = np.median(vertices, 0)
            # vmin = vertices.min(0)
            # vmax = vertices.max(0)
            # center[up_axis] = 0.5 * (vmax[up_axis] + vmin[up_axis])
            # if height is None:
            #     scale = self.avg_height / (vmax[up_axis] - vmin[up_axis])
            # else:
            #     scale = height / (vmax[up_axis] - vmin[up_axis])
            #
            # self.scale = scale
            # self.center = center
            # vertices_, faces, normals, faces_normals, textures, face_textures = \
            #     load_file2info(obj_path, with_normal=True, with_texture=True)

            if mesh_generation == 'agisoft':
                # use open3d (use below for agisoft results)
                textures = None
                mesh = o3d.io.read_triangle_mesh(obj_path, enable_post_processing=False)
                vertices = np.asarray(mesh.vertices)
                normals = np.asarray(mesh.vertex_normals)
                face_textures = np.asarray(mesh.triangle_uvs)
                faces_normals = faces = np.asarray(mesh.triangles)
                textr_vts = np.asarray(mesh.vertex_colors)
            else:
                # without library (comment below lines for non-agisoft results)
                # textr_vts = torch.tensor(m.visual.vertex_colors[None, :, 0:3].copy()).float() / 255.0
                textr_vts = None
                vertices_, faces, normals, faces_normals, textures, face_textures = \
                    load_file2info(obj_path, with_normal=True, with_texture=True)

            if self.render_params['render_scan']:
                tan, bitan = compute_tangent_from_normals(normals)

                # 'bounce folder is not exit' -> calculate prt and face values
                prt_file = os.path.join(self.path2obj, data, 'bounce', 'bounce0.txt')
                face_prt_file = os.path.join(self.path2obj, data, 'bounce', 'face.npy')
                if not os.path.exists(face_prt_file) and not os.path.exists(prt_file):
                    prt, face_prt = computePRT(obj_path, 10, 2)
                else:
                    prt = np.loadtxt(prt_file)
                    face_prt = np.load(face_prt_file)

                # render images using texture maps
                self.rndr.set_norm_mat(scale, center)
                self.rndr_uv.set_norm_mat(scale, center)
                self.rndr.set_mesh(vertices, faces, normals, faces_normals,
                                   textures, face_textures, prt, face_prt, tan, bitan)
                self.rndr.set_albedo(texture_image)
                self.rndr_uv.set_mesh(vertices, faces, normals, faces_normals,
                                      textures, face_textures, prt, face_prt, tan, bitan)
                self.rndr_uv.set_albedo(texture_image)
                output = {'obj_path': obj_path, 'vertices': vertices, 'vertex_colors': textr_vts,
                          'faces': faces, 'center': center, 'scale': scale}
            else:
                # smpl is originally scaled to 180cm -> rescale to new height.
                center = np.asarray([0, 0, 0])
                scale = 1.0
                smpl_scale = height / self.avg_height
                texture_image = np.zeros((512, 512, 3))
                textures = np.zeros((10475, 3)).astype(np.uint8)
                face_textures = np.zeros((20908, 3)).astype(np.uint8)

        smpl_mesh, smpl_params = None, None
        if self.render_params['render_smpl']:
            smpl_path_mesh = os.path.join(self.path2smpl, data, data + '.obj')
            if not os.path.exists(smpl_path_mesh):
                files = [sorted(glob.glob(os.path.join(self.path2smpl, data, '*.obj')))]
                if len(files) == 0:
                    return output
                smpl_path_mesh = os.path.join(self.path2smpl, data, files[0])
            smpl_path_params = os.path.join(self.path2smpl, data, data + '.json')
            with open(smpl_path_params, 'r') as f:
                smpl_params = json.load(f)

            smpl_mesh = trimesh.load(smpl_path_mesh, process=False, maintain_order=True)
            # smpl_mesh = trimesh.smoothing.filter_laplacian(smpl_mesh.subdivide().subdivide())

            (smpl_vertices, smpl_faces, smpl_normals, smpl_faces_normals, smpl_textures, smpl_face_textures) = \
                load_file2info(smpl_path_mesh, with_normal=True, with_texture=True)
            smpl_tan, smpl_bitan = compute_tangent_from_normals(smpl_normals)
            smpl_prt, smpl_face_prt = computePRT(smpl_path_mesh,10, 2)

            self.smpl_rndr.set_norm_mat(smpl_scale, np.zeros_like(center))
            self.smpl_rndr.set_mesh(smpl_vertices, smpl_faces, smpl_normals, smpl_faces_normals,
                                    textures, face_textures, smpl_prt, smpl_face_prt, smpl_tan, smpl_bitan)
            self.smpl_rndr.set_albedo(texture_image)
            self.smpl_rndr_uv.set_norm_mat(smpl_scale, np.zeros_like(center))
            self.smpl_rndr_uv.set_mesh(smpl_vertices, smpl_faces, smpl_normals, smpl_faces_normals,
                                       textures, face_textures, smpl_prt, smpl_face_prt, smpl_tan, smpl_bitan)
            self.smpl_rndr_uv.set_albedo(texture_image)

            output['scale'] = scale
            output['smpl_mesh'] = smpl_mesh
            output['smpl_params'] = smpl_params
            output['center'] = center
        return output

    def set_renderer(self):
        if self.rendering_method == 'opengl':
            # NOTE: GL context has to be created before any other OpenGL function loads.
            from .renderer.gl_render.init_gl import initialize_GL_context
            initialize_GL_context(width=self.cam.width, height=self.cam.height, egl=True)
            from .renderer.gl_render.prt_render import PRTRender

            if self.render_params['render_scan']:
                self.rndr = PRTRender(width=self.cam.width, height=self.cam.height, ms_rate=4, egl=True)
                self.rndr_uv = PRTRender(width=self.cam.width, height=self.cam.height, uv_mode=True, egl=True)
            if self.render_params['render_smpl']:
                self.smpl_rndr = PRTRender(width=self.cam.width, height=self.cam.height, ms_rate=4, egl=True)
                self.smpl_rndr_uv = PRTRender(width=self.cam.width, height=self.cam.height, uv_mode=True, egl=True)
        elif self.rendering_method == 'open3d':  # neural renderer
            pass

        elif self.rendering_method == 'nr':  # neural renderer
            import neural_renderer as nr
            self.rndr = nr.Renderer(image_size=self.cam.width,
                                    orig_size=self.cam.width,
                                    dist_coeffs=self.cam.dist_params,
                                    anti_aliasing=True,
                                    camera_direction=[0, 0, -1],
                                    camera_mode='projection',
                                    viewing_angle=0,
                                    light_color_directional=[1, 1, 1],  # white light source
                                    light_intensity_directional=0.3,
                                    light_intensity_ambient=0.7,
                                    light_direction=[0, -0.7, -1],
                                    near=self.cam.near, far=self.cam.far)
        else:
            assert "Not defined renderer, only GL and neural renderer are supported"

    def render_diffuse_image(self, rot, yaw, pitch, path_dict, sh, idx,
                             reverse=False, render_smpl=False):
        """
        RENDER images in the image and uv spaces.
        If self.skip_exist is set to True, this function skip generating corresponding images.
        :param rot: rotation matrix (camera parameter) (3x3 numpy)
        :param yaw: yaw angle (int)
        :param pitch: pitch angle (int)
        :param path_dict: dictionary containing paths to save images
        :param sh: spherical harmonics (9x3)
        :param idx: index of images (indicating different lighting conditions) (int)
        :param reverse: render back-view images if this true (bool)
        :return: nothing but save rendered images.
        """
        if reverse:  # render front or back (hidden) views.
            img_png = '%03d_%03d_%03d_back.png' % (pitch, yaw, idx)
            self.cam.near = self.cam_params['far']
            self.cam.far = self.cam_params['near']
        else:
            img_png = '%03d_%03d_%03d_front.png' % (pitch, yaw, idx)
            self.cam.near = self.cam_params['near']
            self.cam.far = self.cam_params['far']

        if render_smpl:  # render smpl or scan model.
            rndr = self.smpl_rndr
            rndr_uv = self.smpl_rndr_uv
        else:
            rndr = self.rndr
            rndr_uv = self.rndr_uv

        if path_dict['diffuse'] != "":
            diffuse_image_name = os.path.join(path_dict['diffuse'], img_png)
            if not (self.skip_exist and os.path.exists(diffuse_image_name)):
                rndr.rot_matrix = rot
                rndr.set_camera(self.cam)
                rndr.set_sh(sh)
                rndr.analytic = False
                rndr.use_inverse_depth = False
                rndr.display()

                out_color_sh = rndr.get_color(0)
                # 4-channel rendering (diffusion network)
                out_color_sh = cv2.cvtColor(out_color_sh, cv2.COLOR_RGBA2BGRA)
                # 3-channel rendering (u-net or previous approaches)
                # out_color_sh = cv2.cvtColor(out_color_sh, cv2.COLOR_RGBA2BGR)
                cv2.imwrite(diffuse_image_name, (255.0 * out_color_sh).astype(np.int16))

        # if path_dict['uv_texture'] != "" and reverse is False:
        #     uv_image_name = os.path.join(path_dict['uv_texture'], img_png)
        #     uv_mask_name = os.path.join(path_dict['uv_mask'], img_png)
        #     if not (self.skip_exist and os.path.exists(uv_image_name)):
        #         rndr_uv.rot_matrix = rot
        #         rndr_uv.set_camera(self.cam)
        #         rndr_uv.set_sh(sh)
        #         rndr_uv.analytic = False
        #         rndr_uv.use_inverse_depth = False
        #         rndr_uv.display()
        #
        #         uv_color = rndr_uv.get_color(0)
        #         uv_mask = uv_color[:, :, -1]
        #         uv_color = cv2.cvtColor(uv_color, cv2.COLOR_RGBA2BGR)
        #         # rndr_uv; (0): color, (1): position map, (2): normal
        #         cv2.imwrite(uv_image_name, (255.0 * uv_color))
        #         cv2.imwrite(uv_mask_name, (255.0 * uv_mask))
        #
        #     if 'uv_pos' in path_dict:
        #         uv_pos_name = os.path.join(path_dict['uv_pos'], img_png)
        #         if not (self.skip_exist and os.path.exists(uv_image_name)):
        #             uv_pos = rndr_uv.get_color(1)
        #             uv_pos, uv_mask = uv_pos[:, :, :3], uv_pos[:, :, 3]
        #             # uv_pos is not scaled and translated, it is quite tricky
        #             uv_pos -= self.center
        #             uv_pos *= self.scale
        #
        #             # save zero-centered position map as a 16 bit image
        #             # uv_pos = uv_pos * 128.0 + self.INT15_MAX
        #             # uv_pos[uv_mask == 0, :] = 0
        #             # cv2.imwrite(uv_pos_name, uv_pos.astype(np.uint16))
        #
        #     # save a pkl file for training implicit function.
        #     uv_pos_ = np.transpose(uv_pos, (2, 0, 1)).reshape(3, -1)
        #     uv_color_ = np.transpose(uv_color, (2, 0, 1)).reshape(3, -1)
        #     mask = uv_mask.reshape(-1)
        #     implicit_data = {'uv_pos': uv_pos_[:, mask > 0], 'uv_color': uv_color_[:, mask > 0]}
        #     with open('test.pkl', 'wb') as handle:
        #         pickle.dump(implicit_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def render_diffuse_zero123(self, rot, path_dict, sh, idx, filename,
                               distance=1.4, render_smpl=False):
        """
        RENDER images in the image and uv spaces.
        If self.skip_exist is set to True, this function skip generating corresponding images.
        :param rot: rotation matrix (camera parameter) (3x3 numpy)
        :param yaw: yaw angle (int)
        :param pitch: pitch angle (int)
        :param path_dict: dictionary containing paths to save images
        :param sh: spherical harmonics (9x3)
        :param idx: index of images (indicating different lighting conditions) (int)
        :param reverse: render back-view images if this true (bool)
        :return: nothing but save rendered images.
        """
        img_png = os.path.join(filename, '%03d.png' % (idx))
        self.cam.near = self.cam_params['near']
        self.cam.far = self.cam_params['far']

        if render_smpl:  # render smpl or scan model.
            rndr = self.smpl_rndr
            rndr_uv = self.smpl_rndr_uv
        else:
            rndr = self.rndr
            rndr_uv = self.rndr_uv

        self.cam.center[2] = distance
        if path_dict['diffuse'] != "":
            diffuse_image_name = os.path.join(path_dict['diffuse'], img_png)
            if not (self.skip_exist and os.path.exists(diffuse_image_name)):
                rndr.rot_matrix = rot
                rndr.set_camera(self.cam)
                rndr.set_sh(sh)
                rndr.analytic = False
                rndr.use_inverse_depth = False
                rndr.display()

                out_color_sh = rndr.get_color(0)
                # print(np.min(np.max(out_color_sh, axis=2)))
                # print(np.max(np.max(out_color_sh, axis=2)))
                # out_color_sh = cv2.cvtColor(out_color_sh, cv2.COLOR_RGBA2BGR)
                out_color_sh = cv2.cvtColor(out_color_sh, cv2.COLOR_RGBA2BGRA)
                cv2.imwrite(diffuse_image_name, (255.0 * out_color_sh).astype(np.int16))

        if path_dict['uv_texture'] != "":
            uv_image_name = os.path.join(path_dict['uv_texture'], img_png)
            if not (self.skip_exist and os.path.exists(uv_image_name)):
                rndr_uv.rot_matrix = rot
                rndr_uv.set_camera(self.cam)
                rndr_uv.set_sh(sh)
                rndr_uv.analytic = False
                rndr_uv.use_inverse_depth = False
                rndr_uv.display()

                uv_color = rndr_uv.get_color(0)
                uv_color = cv2.cvtColor(uv_color, cv2.COLOR_RGBA2BGR)
                cv2.imwrite(uv_image_name, (255.0 * uv_color).astype(np.int16))

    def render_albedo_and_depth(self, rot, yaw, pitch, path_dict,
                                reverse=False, render_smpl=False, force_render=False):
        """
        Render albedo image, depth/normal maps, and mask
        > These are invariant to lighting, so we need to call this one time for
        > the same viewpoint
        :param rot: rotation matrix (camera parameter) (3x3 numpy)
        :param yaw: yaw angle (int)
        :param pitch: pitch angle (int)
        :param path_dict: dictionary containing paths to save images
        :param reverse: render back-view images if this true (bool)
        :param render_smpl: render images for smpl mesh
        :return: nothing but save rendered depth maps and images.
        """
        if reverse:
            img_png = '%03d_%03d_%03d_back.png' % (pitch, yaw, 0)
            self.cam.near = self.cam_params['far']
            self.cam.far = self.cam_params['near']
        else:
            img_png = '%03d_%03d_%03d_front.png' % (pitch, yaw, 0)
            self.cam.near = self.cam_params['near']
            self.cam.far = self.cam_params['far']

        skip_albedo_render, skip_mask_render = False, False
        skip_depth_render, skip_normal_render = False, False
        if path_dict['albedo'] != 0:
            albedo_image_name = os.path.join(path_dict['albedo'], img_png)
            if self.skip_exist and os.path.exists(albedo_image_name):
                skip_albedo_render = True
        if path_dict['mask'] != 0:
            mask_name = os.path.join(path_dict['mask'], img_png)
            if self.skip_exist and os.path.exists(mask_name):
                skip_mask_render = True
        if path_dict['depth'] != 0:
            depth_name = os.path.join(path_dict['depth'], img_png)
            if self.skip_exist and os.path.exists(depth_name):
                skip_depth_render = True
        if path_dict['normal'] != 0:
            normal_name = os.path.join(path_dict['normal'], img_png)
            if self.skip_exist and os.path.exists(normal_name):
                skip_normal_render = True

        if skip_albedo_render and skip_mask_render and skip_depth_render and skip_normal_render:
            return None

        nsh = np.zeros((9, 3))
        nsh[0, :] = 1.0

        if render_smpl:
            rndr = self.smpl_rndr
            rndr_uv = self.smpl_rndr_uv
        else:
            rndr = self.rndr
            rndr_uv = self.rndr_uv

        rndr.set_camera(self.cam)
        rndr.rot_matrix = rot
        rndr.set_sh(nsh)
        rndr.analytic = False
        rndr.use_inverse_depth = False
        rndr.display()

        out_color_f = rndr.get_color(0)
        out_mask = out_color_f[:, :, 3] * 255.0
        out_mask[out_mask < 255.0] = 0
        out_color_f = cv2.cvtColor(out_color_f, cv2.COLOR_RGBA2BGR)

        if path_dict['mask'] and not skip_mask_render:
            cv2.imwrite(mask_name, out_mask)

        if path_dict['depth'] or path_dict['normal']:
            out_depth_f = rndr.get_color(2)
            out_depth_f = cv2.cvtColor(out_depth_f, cv2.COLOR_RGBA2BGRA)
            linear_depth = ((2.0 * self.cam.near * self.cam.far)
                           / (self.cam.far + self.cam.near - out_depth_f[:, :, -1]
                              * (self.cam.far - self.cam.near)))

            if path_dict['depth'] and not skip_depth_render:
                out_depth_f = np.uint16((linear_depth - self.dist2cam) * 128 + self.INT15_MAX)
                out_depth_f[out_mask == 0] = 0
                cv2.imwrite(depth_name, out_depth_f)

            if path_dict['normal'] and not skip_normal_render:
                out_normal = get_plane_params(z=torch.Tensor(linear_depth).unsqueeze(0),
                                              pred_res=self.cam.width, xy=self.xy,
                                              real_dist=self.cam.center[2], z_real=False)
                out_normal[0, 0:3, :, :] = (out_normal[0, 0:3, :, :] + 1) / 2 * 255
                out_normal[:, :, out_mask == 0] = 0
                pers_normal = np.uint8(
                    out_normal[0, 0:3, :, :].permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(normal_name, pers_normal)

        if path_dict['albedo'] and not skip_albedo_render:
            cv2.imwrite(albedo_image_name, (255.0 * out_color_f).astype(np.int16))
            return out_color_f

        if path_dict['uv_mask'] or path_dict['uv_normal'] or path_dict['uv_texture'] or path_dict['uv_pkl']:
            uv_image_name = os.path.join(path_dict['uv_texture'], img_png)
            uv_mask_name = os.path.join(path_dict['uv_mask'], img_png)
            uv_pkl_name = os.path.join(path_dict['uv_pkl'], img_png.replace('.png', '.pkl'))
            if not (self.skip_exist and os.path.exists(uv_image_name)):
                rndr_uv.set_camera(self.cam)
                # test.
                # rot = self.update_rotation_matrix(yaw, -pitch)
                # rndr_uv.rot_matrix = rot  # rot doesn't work for rndr_uv
                rndr_uv.set_sh(nsh)
                rndr_uv.analytic = False
                rndr_uv.use_inverse_depth = False
                rndr_uv.display()

                uv_color = rndr_uv.get_color(0)
                uv_mask = uv_color[:, :, -1]
                uv_color = cv2.cvtColor(uv_color, cv2.COLOR_RGBA2BGR)

                # uv_normal = rndr_uv.get_color(2)
                # rndr_uv; (0): color, (1): position map, (2): normal
                cv2.imwrite(uv_image_name, (255.0 * uv_color))
                cv2.imwrite(uv_mask_name, (255.0 * uv_mask))
                # cv2.imwrite(uv_normal_name, (255.0 * uv_mask))

            if 'uv_pos' in path_dict:
                uv_pos_name = os.path.join(path_dict['uv_pos'], img_png)
                if not (self.skip_exist and os.path.exists(uv_image_name)):
                    uv_pos = rndr_uv.get_color(1)
                    uv_pos, uv_mask = uv_pos[:, :, :3], uv_pos[:, :, 3]
                    # uv_pos is not scaled and translated, it is quite tricky
                    # uv_pos -= self.center
                    # uv_pos *= self.scale

                    # save zero-centered position map as a 16 bit image
                    # uv_pos = uv_pos * 128.0 + self.INT15_MAX
                    # uv_pos[uv_mask == 0, :] = 0
                    # cv2.imwrite(uv_pos_name, uv_pos.astype(np.uint16))

                    # save a pkl file for training implicit function.
                    uv_pos_ = np.transpose(uv_pos, (2, 0, 1)).reshape(3, -1)
                    uv_color_ = np.transpose(uv_color, (2, 0, 1)).reshape(3, -1)
                    # uv_normal_ = np.transpose(uv_normal[:, :, :3], (2, 0, 1)).reshape(3, -1)
                    mask = uv_mask.reshape(-1)

                    uv_pos_ = uv_pos_[:, mask > 0]
                    uv_color_ = uv_color_[:, mask > 0] * 255.0
                    # uv_normal_ = uv_normal_[:, mask > 0]

                    uv_pos_ = uv_pos_ - self.center[:, None]
                    uv_pos_ = uv_pos_ * self.scale
                    # do not understand why pitch should be negative.
                    rot = self.update_rotation_matrix(yaw, -pitch)
                    # uv_pos_[-1, :] += 300.0
                    # uv_pos_ = np.matmul(rot.T, uv_pos_) - np.matmul(rot.T, np.asarray([[0], [0], [300]]))
                    uv_pos_ = np.matmul(rot, uv_pos_)
                    # uv_normal_ = np.matmul(rot, uv_normal_)

                    # save minimum abo
                    implicit_data = {'uv_pos': uv_pos_.astype(np.float16),
                                     'uv_color': uv_color_.astype(np.uint8)}
                                     # 'uv_normal': uv_normal_}
                    with open(uv_pkl_name, 'wb') as handle:
                        pickle.dump(implicit_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_spherical_harmonics(self, num_light=None):
        """
        Generate a list of sh(spherical harmonics)
        to render diffuse images (specular reflection will not be applied)
        :param num_light: the number of lighting conditions
        :return: generates spherical harmonics (9x3) matrix for each lighting condition
        """
        def get_random_sh():
            shcoeffs = np.zeros((9, 3))
            shcoeffs[0, :] = 0.8
            shcoeffs[1:, :] = 1.0 * np.random.rand(8, 3)
            return shcoeffs

        sh = []
        if num_light is None:
            num_light = self.num_light

        for _ in range(num_light):
            random_sh = False
            if self.random_sh > 0:
                prob = random.random()
                if prob < self.random_sh:
                    random_sh = True

            if random_sh:
                sh_ = get_random_sh()
            else:
                # set sh values from pre-defined 'env_sh.npy' file
                # generate natural lighting sources
                sh_id = random.randint(0, self.shs.shape[0] - 1)
                sh_ = np.copy(self.shs[sh_id])
                # add gaussian noise to further diversify illuminations.
                sh_ += np.random.rand(9, 3) * 0.01
                if np.sum(sh_[0, :]) > 3.0:
                    sh_[0, :] *= 0.8

            sh_angle = 0.5 * np.pi * (random.random() - 0.5)  # -45~+45
            sh_ = rotateSH(sh_, make_rotate(0, sh_angle, 0).T)
            sh_angle = 0.5 * np.pi * (random.random() - 0.5)  # -45~+45
            sh_ = rotateSH(sh_, make_rotate(0, 0, sh_angle).T)
            sh_angle = 0.5 * np.pi * (random.random() - 0.5)  # -45~+45
            sh_ = rotateSH(sh_, make_rotate(sh_angle, 0, 0).T)  # roll?
            sh.append(sh_)
        return sh

    @staticmethod
    def update_rotation_matrix(yaw, pitch):
        """
        Misc. function for generating rotation matrix with yaw/pitch angles
        :param rot: identity rotation matrix (3x3, numpy)
        :param yaw: yaw angle (int)
        :param pitch: pitch angle (int)
        :return: updated rotation matrix.
        """
        R_delta = np.matmul(make_rotate(np.deg2rad(pitch), 0, 0),
                            make_rotate(0, np.deg2rad(yaw), 0))
        R_np = np.matmul(np.eye(3), R_delta)
        return R_np

    @staticmethod
    def save_params(path2save, mesh_params, yaw, pitch, light_idx):
        json_name = '%03d_%03d_%03d.json' % (pitch, yaw, light_idx)
        param_name = os.path.join(path2save, json_name)
        with open(param_name, 'w') as fp:
            json.dump(mesh_params, fp, indent="\t")

    @staticmethod
    def save_params_zero123(path2save, mesh_params, num_data):
        json_name = '%03d.json' % (num_data)
        param_name = os.path.join(path2save, json_name)
        with open(param_name, 'w') as fp:
            json.dump(mesh_params, fp, indent="\t")

    def render_zero123(self, data=None):
        if data is None:
            data = self.data_list.pop(0)
        path_dict = self.set_dirs(data, zero123blender=True)

        # load mesh data
        if self.h_min != self.h_max:
            height = random.randint(self.h_min, self.h_max)
        else:
            height = self.avg_height

        mesh_data = self.load_gt_data(data, height)
        if mesh_data is None:
            return

        # calculate intrinsic and extrinsic matrix
        rot, intrinsic, translation, _, _ = self.cam.get_gl_matrix()
        distance = np.linalg.norm(translation)

        print('generating images for {0}'.format(data))
        sh = self.get_spherical_harmonics()

        def az_el_to_points(yaw, pitch):
            x = np.cos(yaw) * np.cos(pitch)
            y = np.sin(yaw) * np.cos(pitch)
            z = np.sin(pitch)
            return np.stack([x, y, z], -1)  #

        # save the camera parameters in the camera coordinate.
        for i in range(self.num_light):
            num_images_src = 5
            num_images_target = 10
            sample_num = 0
            while num_images_src > 0 or num_images_target > 0:
                if num_images_src > 0:
                    num_images_src -= 1
                    pitch = random.randint(-5, 5) % 360
                    yaw = random.randint(-10, 10) % 360
                elif num_images_target > 0:
                    num_images_target -= 1
                    pitch = random.randint(-10, 30) % 360
                    yaw = random.randint(10, 350)

                rot_cur = self.update_rotation_matrix(yaw, pitch)
                if self.rendering_method == 'opengl':
                    f_name = path_dict['path4scan']['diffuse'] + '_%03d' % i
                    os.makedirs(f_name, exist_ok=True)
                    self.render_diffuse_zero123(rot_cur, path_dict['path4scan'],
                                                sh[i], sample_num, f_name,
                                                distance=distance)

                    # save params
                    if self.render_params['scan_param']:
                        rot1 = self.update_rotation_matrix(yaw, 0)
                        rot2 = self.update_rotation_matrix(0, -pitch)
                        rot_cur = np.matmul(rot1, rot2)

                        # blender coordinate
                        R = np.asarray([rot_cur[k] for k in [2, 0, 1]])
                        t = az_el_to_points(np.deg2rad(yaw), np.deg2rad(pitch)) * distance

                        R = R.T
                        t = -R @ t

                        rot_world = rot @ R
                        t_world = rot @ t[:, None]

                        RT = np.concatenate((rot_world, t_world), axis=1)
                        mesh_params = {
                            'RT': RT.tolist()
                        }
                        f_name = path_dict['path4scan']['param'] + '_%03d' % i
                        os.makedirs(f_name, exist_ok=True)
                        self.save_params_zero123(f_name, mesh_params, sample_num)
                    sample_num += 1

    # render point cloud or mesh without texture map
    def render_zoom123(self, data=None):
        if data is None:
            data = self.data_list.pop(0)
        path_dict = self.set_dirs(data, zero123blender=True)

        # load mesh data
        if self.h_min != self.h_max:
            height = random.randint(self.h_min, self.h_max)
        else:
            height = self.avg_height

        mesh_data = self.load_gt_data(data, height)
        if mesh_data is None:
            return

        # calculate intrinsic and extrinsic matrix
        rot, intrinsic, translation, _, _ = self.cam.get_gl_matrix()
        distance = np.linalg.norm(translation)

        print('generating images for {0}'.format(data))
        sh = self.get_spherical_harmonics()

        def az_el_to_points(yaw, height, distance):
            x = np.cos(yaw) * distance
            y = np.sin(yaw) * distance
            z = height
            return np.stack([x, y, z], -1)

        # save the camera parameters in the camera coordinate.
        for i in range(self.num_light):
            num_images_src = 5
            num_images_target = 10
            sample_num = 0
            while num_images_src > 0 or num_images_target > 0:
                if num_images_src > 0:
                    num_images_src -= 1
                    height = random.randint(-1, 1) / 2
                    yaw = random.randint(-10, 10) % 360
                    cur_distance = distance - random.randint(0, 3) / 1
                elif num_images_target > 0:
                    num_images_target -= 1
                    pitch = random.randint(-10, 30) % 360
                    yaw = random.randint(10, 350)

                rot_cur = self.update_rotation_matrix(yaw, pitch)
                if self.rendering_method == 'opengl':
                    f_name = path_dict['path4scan']['diffuse'] + '_%03d' % i
                    os.makedirs(f_name, exist_ok=True)
                    self.render_diffuse_zero123(rot_cur, path_dict['path4scan'],
                                                sh[i], sample_num, f_name,
                                                distance=distance)

                    # save params
                    if self.render_params['scan_param']:
                        rot1 = self.update_rotation_matrix(yaw, 0)
                        rot2 = self.update_rotation_matrix(0, -pitch)
                        rot_cur = np.matmul(rot1, rot2)

                        # blender coordinate
                        R = np.asarray([rot_cur[k] for k in [2, 0, 1]])
                        t = az_el_to_points(np.deg2rad(yaw), np.deg2rad(pitch)) * distance

                        R = R.T
                        t = -R @ t

                        rot_world = rot @ R
                        t_world = rot @ t[:, None]

                        RT = np.concatenate((rot_world, t_world), axis=1)
                        mesh_params = {
                            'RT': RT.tolist()
                        }
                        f_name = path_dict['path4scan']['param'] + '_%03d' % i
                        os.makedirs(f_name, exist_ok=True)
                        self.save_params_zero123(f_name, mesh_params, sample_num)
                    sample_num += 1

    def get_spherical_harmonics_from_params(self, path2params,
                                            yaw, pitch):
        sh = []
        for idx in range(self.num_light):
            filename = '/%03d_%03d_%03d.json' % (pitch, yaw, idx)
            with open(path2params + filename, 'r') as f:
                cam_params = json.load(f)
                sh.append(np.asarray(cam_params['sh']))
        return np.stack(sh, axis=0)

    def render_nr(self, data):
        if data is None:
            data = self.data_list.pop(0)
        path_dict = self.set_dirs(data)

        # load mesh data
        if self.h_min != self.h_max:
            height = random.randint(self.h_min, self.h_max)
        else:
            height = self.avg_height

        mesh_data = self.load_data_trimesh(data, height)
        mesh_data['vertices'] = torch.Tensor(mesh_data['vertices'][None, :, :]).to(self.device)
        mesh_data['faces'] = torch.Tensor(mesh_data['faces'][None, :, :]).to(self.device)
        textr_face = torch.tensor(mesh_data['face_colors'][None, :, -2:-5:-1].copy()).float() / 255.0
        mesh_data['face_colors'] = textr_face.unsqueeze(2).unsqueeze(2).unsqueeze(2).to(self.device)

        if mesh_data is None:
            return

        # calculate intrinsic and extrinsic matrix
        def render_diffuse_nr(mesh_data, yaw, pitch, path2save):
            img_png = '%03d_%03d_%03d.png' % (pitch, yaw, 0)

            rot, intrinsic, translation, _, _ = self.cam.get_gl_matrix()
            rot_np = self.update_rotation_matrix(yaw, pitch)
            rot_new = np.matmul(rot, rot_np)

            K = torch.Tensor(intrinsic[None, :, :]).to(self.device)
            R = torch.Tensor(rot_new[None, :, :]).to(self.device)
            t = torch.Tensor(translation[None, :]).to(self.device)
            dist = torch.Tensor(self.cam.dist_params).to(self.device)
            image_out, depth_out, mask_out = self.rndr(mesh_data['vertices'],
                                                       mesh_data['faces'],
                                                       mesh_data['face_colors'],
                                                       K=K, R=R, t=t,
                                                       dist_coeffs=dist)

            image_out = image_out.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
            # mask_out = mask_out.detach().cpu().squeeze(0).numpy()
            cv2.imwrite(os.path.join(path2save['diffuse'], img_png), image_out * 255.0)
            # cv2.imwrite(os.path.join(path2save['mask'], img_png), mask_out * 255.0)

            return image_out, intrinsic, rot_np, translation

        gender = 'neutral'
        if self.gender_predictor:
            front_image, _, _, _ = \
                render_diffuse_nr(mesh_data, 0, 0, path_dict['path4scan'])
            if isinstance(front_image, np.ndarray):
                front_image = Image.fromarray((front_image * 255).astype(np.uint8))
                gender = self.gender_predictor(front_image)

        for pitch in self.pitch:
            for yaw in self.view_idx:
                if self.render_params['render_scan']:
                    _, intrinsic, rot_cur, translation = \
                        render_diffuse_nr(mesh_data, yaw, pitch, path_dict['path4scan'])

                    if self.render_params['scan_param']:
                        mesh_params = {'K': intrinsic.tolist(),
                                       'R': rot_cur.tolist(),
                                       't': translation.tolist(),
                                       'scale': mesh_data['scale'].tolist(),
                                       'center': mesh_data['center'].tolist(),
                                       'height': height,  # the height of the model
                                       'gender': gender
                                       }
                        self.save_params(path_dict['path4scan']['param'],
                                         mesh_params, yaw, pitch, 0)

    def render(self, data=None):
        if data is None:
            data = self.data_list.pop(0)
        path_dict = self.set_dirs(data)

        # load mesh data
        if self.projection == 'orthographic':
            filename = '/%03d_%03d_%03d.json' % (0, 0, 0)
            with open(path_dict['path4scan']['param'] + filename, 'r') as f:
                cam_params = json.load(f)
                height = cam_params['height']
        elif self.h_min != self.h_max:
            height = random.randint(self.h_min, self.h_max)
        else:
            height = self.avg_height

        # to render uv pkl files only.
        # filename = '/%03d_%03d_%03d.json' % (0, 0, 0)
        # with open(path_dict['path4scan']['param'].replace('IMPLICIT', 'TRAIN') + filename, 'r') as f:
        #     cam_params = json.load(f)
        #     height = cam_params['height']

        if self.cam.ortho_ratio is not None:
            # set new ortho ratio
            self.cam.ortho_ratio = 0.4 * (512 / self.cam_params['width'])
            self.cam.ortho_ratio /= height / self.avg_height

        mesh_data = self.load_gt_data(data, height)
        self.scale = mesh_data['scale']
        self.center = mesh_data['center']
        # if mesh_data is None:
        #     return

        # calculate intrinsic and extrinsic matrix
        rot, intrinsic, translation, _, _ = self.cam.get_gl_matrix()

        print('generating images for {0}'.format(data))
        # uniform lighting.
        sh = self.get_spherical_harmonics()
        rot = np.eye(3)  # i don't know why...

        gender = 'neutral'
        if self.gender_predictor:
            front_image = self.render_albedo_and_depth(rot, 0, 0,
                                                       path_dict['path4scan'], force_render=True)
            if isinstance(front_image, np.ndarray):
                front_image = Image.fromarray((front_image[:, :, [2, 1, 0]] * 255).astype(np.uint8))
                gender = self.gender_predictor(front_image)

        for pitch in self.pitch:
            for yaw in self.view_idx:
                rot_cur = self.update_rotation_matrix(yaw, pitch)
                rot_cur = np.matmul(rot, rot_cur)
                if self.cam.ortho_ratio is not None:
                    self.ortho_ratio = 0.4 * (512 / 512)
                    # mesh_params['ortho_ratio'] = self.ortho_ratio
                    # mesh_params['ortho_ratio'] = self.cam.ortho_ratio

                if self.rendering_method == 'opengl':
                    if self.render_params['render_smpl']:
                        self.render_albedo_and_depth(rot_cur, yaw, pitch, path_dict['path4smpl'],
                                                     render_smpl=True)
                        if self.render_params['smpl_back']:
                            self.render_albedo_and_depth(rot_cur, yaw, pitch, path_dict['path4smpl'],
                                                         render_smpl=True, reverse=True)
                    if self.render_params['render_scan']:
                        self.render_albedo_and_depth(rot_cur, yaw, pitch, path_dict['path4scan'])
                        if self.render_params['scan_back']:
                            self.render_albedo_and_depth(rot_cur, yaw, pitch, path_dict['path4scan'],
                                                         reverse=True)

                    if self.projection == 'orthographic':
                        # load sh from existing files
                        # sh = self.get_spherical_harmonics_from_params(path_dict['path4scan']['param'],
                        #                                               yaw, pitch)
                        sh = self.get_spherical_harmonics()
                    elif not self.uniform_light:
                        # change illumination every time.
                        sh = self.get_spherical_harmonics()

                    if self.render_params['render_scan']:
                        for i in range(self.num_light):
                            # save params
                            self.render_diffuse_image(rot_cur, yaw, pitch,
                                                      path_dict['path4scan'], sh[i], i)

                            if self.render_params['scan_param']:
                                mesh_params = {'K': intrinsic.tolist(),
                                               'R': rot_cur.tolist(),
                                               't': translation.tolist(),
                                               'scale': mesh_data['scale'].tolist(),
                                               'center': mesh_data['center'].tolist(),
                                               'height': height,  # the height of the model
                                               'gender': gender,
                                               'sh': sh[i].tolist()
                                               }
                                self.save_params(path_dict['path4scan']['param'],
                                                 mesh_params, yaw, pitch, i)

    def forward(self, a_min=0, a_max=-1, multiprocess=False, pool_size=4, verbose=True):
        if verbose:
            start_t = time.time()
        if multiprocess:
            pool = Pool(pool_size)
            for data in self.data_list[a_min:a_max]:
                pool.apply_async(self.render, (data, ))
            pool.close()
            pool.join()
        else:
            if a_max == -1:
                self.data_list = self.data_list[a_min:a_max] + self.data_list[-1:]
            else:
                self.data_list = self.data_list[a_min:a_max]
            for data in self.data_list:
                # self.render_zero123(data)
                # self.render_nr(data)
                self.render_images(data)
        if verbose:
            end_t = time.time()
            print(f"{end_t - start_t:.2f} sec")
