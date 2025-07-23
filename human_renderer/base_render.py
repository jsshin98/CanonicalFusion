import os
import cv2
import torch.utils.data
import torch
import torch.nn as nn
import numpy as np
import random
import math
import json
import trimesh
# import pyexr
from utils.im_utils import get_normal

# from .renderer.mesh import load_obj_mesh2, load_obj_mesh, compute_tangent, rotate_mesh
from .renderer.camera import Camera
from .renderer.camera import rotateSH, make_rotate
from smpl_optimizer.misc import keypoint_loader


class HumanRenderer(nn.Module):
    def __init__(self,
                 input_path=None,
                 save_root='workspace/save_root',
                 path2obj=None,
                 cam_ext=None,
                 view_idx=[0, 180],
                 pitch=[0],
                 light_num=30,
                 pose_detector=None,
                 path2smpl=None,
                 renderer='nr',
                 rendering_mode='dual',
                 skip_exist=True,
                 device=torch.device("cuda:0"),
                 **kwargs):
        super(HumanRenderer, self).__init__()

        if input_path is not None and len(input_path) > 0:
            self.data_list = sorted(os.listdir(input_path))
            self.input_path = input_path
        else:
            self.data_list = []
        self.path2obj = path2obj
        # self.save_image = False
        # 'nr', 'pytorch3d', 'opengl', 'trimesh'
        self.rendering_method = renderer
        self.rendering_mode = rendering_mode  # 'dual' depth maps, 'single' single depth map
        self.save_root = save_root
        self.view_idx = view_idx
        self.pitch = pitch
        self.light_num = light_num
        self.eps = 1e-9
        self.path2smpl = path2smpl

        if cam_ext:
            width, height, projection = cam_ext['width'], cam_ext['height'], cam_ext['projection']
        else:
            width, height, projection = 512, 512, 'perspective'

        self.cam = Camera(width=width,
                          height=height,
                          projection=projection)

        self.cam.center = cam_ext['cmin']
        self.projection = cam_ext['projection']
        self.cam.dist_params = torch.cuda.FloatTensor([[cam_ext['distx'], cam_ext['disty'], 0., 0., 0.]])
        self.cam.focal_x = cam_ext['fx']
        self.cam.focal_y = cam_ext['fy']
        self.cam.principal_x = cam_ext['px']
        self.cam.principal_y = cam_ext['py']
        self.cam.cmax = cam_ext['cmax']
        self.cam.cmin = cam_ext['cmin']

        if device is not None:
            self.device = device
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # support four different types of rendering methods.
        self.renderer, self.render_model = None, None
        self.set_renderer(renderer)
        self.skip_exist = skip_exist

        # support openpose, mediapipe, openpiapef
        self.openpose_flag = False
        if pose_detector is not None:
            if pose_detector == 'openpose':
                self.openpose_flag = True
                from apps.human_pose_estimator import OpenposeFacade
                self.pose_predictor = OpenposeFacade(width=width, height=height)
            else:
                print('Cannot initialize pose detector (support openpose only)')

    def set_dirs(self, data_name):
        os.makedirs(self.save_root, exist_ok=True)
        if self.rendering_method == 'nr':
            os.makedirs(os.path.join(self.save_root, 'COLOR/NR', data_name), exist_ok=True)
            os.makedirs(os.path.join(self.save_root, 'MASK/NR', data_name), exist_ok=True)
            os.makedirs(os.path.join(self.save_root, 'DEPTH/NR', data_name), exist_ok=True)
            os.makedirs(os.path.join(self.save_root, 'NORMAL/NR', data_name), exist_ok=True)
            os.makedirs(os.path.join(self.save_root, 'PARAM/NR', data_name), exist_ok=True)
        elif self.rendering_method == 'trimesh':
            os.makedirs(os.path.join(self.save_root, 'COLOR/ALIGNED', data_name), exist_ok=True)
            os.makedirs(os.path.join(self.save_root, 'MASK/ALIGNED', data_name), exist_ok=True)
            os.makedirs(os.path.join(self.save_root, 'DEPTH/ALIGNED', data_name), exist_ok=True)
            os.makedirs(os.path.join(self.save_root, 'DEPTH/ALIGNED_UNBIASED', data_name), exist_ok=True)
            # os.makedirs(os.path.join(self.save_root, 'NORMAL/ALIGNED', data_name), exist_ok=True)
            # os.makedirs(os.path.join(self.save_root, 'NORMAL/ALIGNED_UNBIASED', data_name), exist_ok=True)
            os.makedirs(os.path.join(self.save_root, 'PARAM/TRIMESH', data_name), exist_ok=True)
        elif self.rendering_method == 'opengl':
            os.makedirs(os.path.join(self.save_root, 'COLOR/OPENGL', data_name), exist_ok=True)
            os.makedirs(os.path.join(self.save_root, 'MASK/OPENGL', data_name), exist_ok=True)
            # os.makedirs(os.path.join(self.save_root, 'DEPTH/OPENGL', data_name), exist_ok=True)
            # os.makedirs(os.path.join(self.save_root, 'NORMAL/OPENGL', data_name), exist_ok=True)
            os.makedirs(os.path.join(self.save_root, 'UV_MASK/OPENGL', data_name), exist_ok=True)
            os.makedirs(os.path.join(self.save_root, 'UV_NORMAL/OPENGL', data_name), exist_ok=True)
            os.makedirs(os.path.join(self.save_root, 'UV_POS/OPENGL', data_name), exist_ok=True)
            os.makedirs(os.path.join(self.save_root, 'UV_RENDER/OPENGL', data_name), exist_ok=True)
            os.makedirs(os.path.join(self.save_root, 'PARAM/OPENGL', data_name), exist_ok=True)
        else:
            os.makedirs(os.path.join(self.save_root, 'POSE', data_name), exist_ok=True)
            os.makedirs(os.path.join(self.save_root, 'SMPLX', data_name), exist_ok=True)

    def set_renderer(self, renderer):
        if renderer == 'nr':
            self.render_model = 'neural_renderer'
            import nr_renderer as nr
            self.renderer = nr.Renderer(image_size=self.cam.width,
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
        elif self.renderer == 'opengl':
            # not implemented
            self.render_model = 'opengl'
            self.renderer = None
            pass
        elif self.renderer == 'trimesh':
            # not implemented
            self.render_model = 'trimesh'
            self.renderer = None
            pass
        elif self.renderer == 'pyrender':
            # not implemented
            self.render_model = 'pyrender'
            self.renderer = None
            pass
        elif renderer == 'pytorch3d':
            self.render_model = 'pytorch3d'
            self.renderer = None
            # not implemented
            pass

    def generate_gt_all(self, input_path):
        while self.data_list:
            self.generate_gt_single(input_path)

    def load_gt_data(self, data, normalize=True, avg_height=180.0):
        if self.path2obj is None:
            obj_path = os.path.join(self.input_path, data, data + '.obj')
        else:
            obj_path = os.path.join(self.path2obj, data, data + '.obj')

        # normalize mesh
        m = trimesh.load_mesh(obj_path, process=False)
        vertices = m.vertices

        if normalize:
            vmin = vertices.min(0)
            vmax = vertices.max(0)
            up_axis = 1 if (vmax - vmin).argmax() == 1 else 2

            center = np.median(vertices, 0)
            center[up_axis] = 0.5 * (vmax[up_axis] + vmin[up_axis])
            scale = avg_height / (vmax[up_axis] - vmin[up_axis])

            # normalization
            vertices -= center
            vertices *= scale

        smpl_path_mesh = os.path.join(self.path2smpl, data, data+'.obj')
        smpl_path_params = os.path.join(self.path2smpl, data, data+'.json')

        if os.path.isfile(smpl_path_params):
            with open(smpl_path_params, "r") as f:
                smpl_params = json.load(f)
        else:
            smpl_params = None

        if os.path.isfile(smpl_path_mesh):
            smpl_mesh = trimesh.load_mesh(smpl_path_mesh)
            # smpl_mesh.vertices = (smpl_mesh.vertices - center) * scale
        else:
            smpl_mesh = None

        textr_vts = []
        textr_face = []
        rndr = []
        rndr_uv = []
        faces = torch.tensor(m.faces[None, :, :].copy()).float().to(self.device)

        if 'smpl' not in self.rendering_mode:
            # if neural renderer is used for rendering
            if not hasattr(m.visual, 'vertex_colors'):
                m, texture_image = self.load_textured_mesh(obj_path, data)
            textr_vts = torch.tensor(m.visual.vertex_colors[None, :, 0:3].copy()).float() / 255.0
            if self.rendering_method == 'nr':
                # it requires face colors.
                textr_face = torch.tensor(m.visual.face_colors[None, :, -2:-5:-1].copy()).float().to(self.device) / 255.0
                textr_face = textr_face.unsqueeze(2).unsqueeze(2).unsqueeze(2)
            elif self.rendering_method == 'opengl':
                prt_file = os.path.join(self.input_path, data, 'bounce', 'bounce0.txt')
                if not os.path.exists(prt_file):
                    print('ERROR: prt file does not exist!!!', prt_file)
                    return
                face_prt_file = os.path.join(self.input_path, data, 'bounce', 'face.npy')
                if not os.path.exists(face_prt_file):
                    print('ERROR: face prt file does not exist!!!', prt_file)
                    return

                vertices, faces, normals, faces_normals, textures, face_textures = \
                    load_obj_mesh(obj_path, with_normal=True, with_texture=True)
                vmin = vertices.min(0)
                vmax = vertices.max(0)
                up_axis = 1 if (vmax - vmin).argmax() == 1 else 2

                center = np.median(vertices, 0)
                center[up_axis] = 0.5 * (vmax[up_axis] + vmin[up_axis])
                scale = 180 / (vmax[up_axis] - vmin[up_axis])

                tan, bitan = compute_tangent(vertices, faces, normals, textures, face_textures)
                prt = np.loadtxt(prt_file)
                face_prt = np.load(face_prt_file)

                # NOTE: GL context has to be created before any other OpenGL function loads.
                from .renderer.gl.init_gl import initialize_GL_context
                initialize_GL_context(width=self.cam.width, height=self.cam.height, egl=False)

                from .renderer.gl.prt_render import PRTRender
                rndr = PRTRender(width=self.cam.width, height=self.cam.height, ms_rate=4, egl=False)
                rndr_uv = PRTRender(width=self.cam.width, height=self.cam.height, uv_mode=True, egl=False)
                rndr.set_norm_mat(scale, center)
                rndr_uv.set_norm_mat(scale, center)

                rndr.set_mesh(vertices, faces, normals, faces_normals,
                              textures, face_textures, prt, face_prt, tan, bitan)
                rndr.set_albedo(texture_image)

                rndr_uv.set_mesh(vertices, faces, normals, faces_normals,
                                 textures, face_textures, prt, face_prt, tan, bitan)
                rndr_uv.set_albedo(texture_image)
                textr_face = torch.tensor(m.visual.face_colors[None, :, 0:3].copy()).float().to(self.device) / 255.0
            elif self.rendering_method == 'trimesh':
                textr_face = torch.tensor(m.visual.face_colors[None, :, 0:3].copy()).float().to(self.device) / 255.0

        return {'obj_path': obj_path, 'vertices': vertices, 'vertex_colors': textr_vts,
                'faces': faces, 'face_colors': textr_face, 'center': center, 'scale': scale,
                'smpl_mesh': smpl_mesh, 'smpl_params': smpl_params, 'rndr': rndr, 'rndr_uv': rndr_uv}

    def load_textured_mesh(self, mesh, data):
        exts = ['.tif', '.bmp', '.jpg']
        if self.path2obj is None:
            text_file = os.path.join(self.input_path, data, data)
        else:
            text_file = os.path.join(self.path2obj, data, data)
        text_file = [text_file+ext for ext in exts if os.path.isfile(text_file+ext)]

        if len(text_file) > 0:
            texture_image = cv2.imread(text_file[0])
            texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)
            texture = np.flip(texture_image, axis=0)
            mesh = load_obj_mesh2(mesh, texture=texture)
            # mesh.export(os.path.join(self.save_root, 'GEO', 'OBJ', data, data + '.obj'))
            return mesh, texture_image
        else:
            return None

    def fetch_generated_data(self, idx=0):
        data = self.data_list.pop(idx)

        dataset = dict()
        p = 0
        mesh_data = self.load_gt_data(data)

        for vid in self.view_idx:
            image_file = os.path.join(self.save_root, 'COLOR/ALIGNED', data, '%d_%d_%02d.png' % (vid, p, 0))
            silhouette_file = os.path.join(self.save_root, 'MASK/ALIGNED', data, '%d_%d_%02d.png' % (vid, p, 0))
            depth_file = os.path.join(self.save_root, 'DEPTH/ALIGNED', data, '%d_%d_%02d_front.png' % (vid, p, 0))
            normal_file = os.path.join(self.save_root, 'NORMAL/ALIGNED', data, '%d_%d_%02d_front.png' % (vid, p, 0))
            param_file = os.path.join(self.save_root, 'PARAM', data, '%d_%d_%02d.json' % (vid, p, 0))
            pose_file = os.path.join(self.save_root, 'POSE', data, '%d_%d_%02d.json' % (vid, p, 0))

            dataset[vid] = {'image': [], 'mask': [], 'depth': [],
                            'normal': [], 'cam_params': [], 'pose': []}

            if os.path.isfile(image_file):
                dataset[vid]['image'] = cv2.imread(image_file)
            elif os.path.isfile(image_file.replace('.png', '_front.png')):
                dataset[vid]['image'] = cv2.imread(image_file.replace('.png', '_front.png'))

            if os.path.isfile(silhouette_file):
                dataset[vid]['mask'] = cv2.imread(silhouette_file)
            if os.path.isfile(depth_file):
                dataset[vid]['depth'] = cv2.imread(depth_file)
            if os.path.isfile(normal_file):
                dataset[vid]['normal'] = cv2.imread(normal_file)
            if os.path.isfile(param_file):
                with open(param_file, 'r') as file:
                    dataset[vid]['cam_params'] = json.load(file)

            if os.path.isfile(pose_file):
                dataset[vid]['pose'] = keypoint_loader(pose_file)
            elif os.path.isfile(pose_file.replace('.json', '_front.json')):
                dataset[vid]['pose'] = keypoint_loader(pose_file.replace('.json', '_front.json'))

        return dataset, mesh_data, data

    def generate_smpl_proj(self, smpl_mesh, file_dir, file_name, flip_y=False, save_results=True):
        scene = smpl_mesh.scene()
        scene.camera.focal = [self.cam.focal_x, self.cam.focal_y]
        scene.camera.resolution = [self.cam.width, self.cam.height]

        # temporal code (valid only if the cam_center is fixed at cmin)
        cam_trans = np.copy(scene.camera_transform)
        cam_trans[:3, 3] = self.cam.center
        if flip_y:
            cam_trans[1, 3] = 0.0

        scene.camera_transform = cam_trans
        scene.camera.z_far = self.cam.far
        scene.camera.z_near = self.cam.near

        pers_color_front, pers_depth_front, pers_depth_front_unbiased, \
        pers_color_back, pers_depth_back, pers_depth_back_unbiased = \
            pers_get_depth_maps(smpl_mesh, scene, self.cam.width)

        pers_depth_front_unbiased = np.flip(pers_depth_front_unbiased, 1)
        pers_depth_back_unbiased = np.flip(pers_depth_back_unbiased, 1)

        if save_results:
            ext = '.' + file_name.split('.')[-1]
            filename = file_name.split('/')[-1].replace(ext, '_front' + '.png')
            path2depth = os.path.join(self.save_root, 'DEPTH/ALIGNED_UNBIASED_SMPLX_PRED', file_dir)
            os.makedirs(os.path.join(self.save_root, 'DEPTH/ALIGNED_UNBIASED_SMPLX_PRED', file_dir), exist_ok=True)

            front_depth_unbiased_name = os.path.join(path2depth, filename)
            back_depth_unbiased_name = front_depth_unbiased_name.replace('_front', '_back')
            cv2.imwrite(front_depth_unbiased_name, pers_depth_front_unbiased.astype(np.uint16))
            cv2.imwrite(back_depth_unbiased_name, pers_depth_back_unbiased.astype(np.uint16))

        pers_depth_front_unbiased = (pers_depth_front_unbiased - 32767.0) / 128 / 128 + 0.5
        pers_depth_back_unbiased = (pers_depth_back_unbiased - 32767.0) / 128 / 128 + 0.5

        return pers_depth_front_unbiased, pers_depth_back_unbiased

    def generate_gt_single(self, input_path):
        data = self.data_list.pop(0)
        self.set_dirs(data)
        mesh_data = self.load_gt_data(data)
        if mesh_data is None:
            return

        R_np, K_np, t_np, _, _ = self.cam.get_gl_matrix()
        K = torch.tensor(K_np[None, :, :]).float().cuda()
        t = torch.tensor(t_np[None, :]).float().cuda()
        verts = torch.Tensor(mesh_data['vertices']).unsqueeze(0).cuda()

        if self.rendering_method == 'opengl':
            uv_mask_name = os.path.join(self.save_root, 'UV_MASK/OPENGL', data, '00.png')
            uv_pos_name = os.path.join(self.save_root, 'UV_POS/OPENGL', data, '00.exr')
            uv_normal_name = os.path.join(self.save_root, 'UV_NORMAL/OPENGL', data, '00.png')
            shs = np.load('./env_sh.npy')

        print('generating images for {0}'.format(data))
        for p in self.pitch:
            for vid in self.view_idx:
                R_delta = np.matmul(make_rotate(math.radians(p), 0, 0), make_rotate(0, math.radians(vid), 0))
                R = np.matmul(R_np, R_delta)
                R = torch.tensor(R[None, :, :]).float().cuda()
                if self.rendering_method == 'nr':
                    image_file = os.path.join(self.save_root, 'COLOR/NR', data, '%d_%d_%02d.png' % (vid, p, 0))
                    silhouette_file = os.path.join(self.save_root, 'MASK/NR', data, '%d_%d_%02d.png' % (vid, p, 0))
                    depth_file = os.path.join(self.save_root, 'DEPTH/NR', data, '%d_%d_%02d.png' % (vid, p, 0))
                    normal_file = os.path.join(self.save_root, 'NORMAL/NR', data, '%d_%d_%02d.png' % (vid, p, 0))
                    param_file = os.path.join(self.save_root, 'PARAM/NR', data, '%d_%d_%02d.json' % (vid, p, 0))

                    # skip if exists
                    if self.skip_exist and os.path.isfile(image_file) and os.path.isfile(silhouette_file) and \
                            os.path.isfile(depth_file):
                        continue

                    images_out, depth_out, silhouette_out = self.renderer(verts, mesh_data['faces'],
                                                                          mesh_data['face_colors'],
                                                                          K=K, R=R, t=t,
                                                                          dist_coeffs=self.cam.dist_params)
                    # save data for training
                    # normal_out = get_normal(depth_out)
                    normal_out = get_normal(z=depth_out, pred_res=self.cam.width,
                                            real_dist=self.cam.center, z_real=False)
                    image = images_out.squeeze().permute(2, 1, 0).detach().cpu().numpy()
                    image = np.flip(np.rot90(image, -1), 1)
                    depth = depth_out.squeeze().detach().cpu().numpy()
                    normal = normal_out.squeeze().permute(1, 2, 0).detach().cpu().numpy()
                    silhouette = silhouette_out.squeeze().detach().cpu().numpy()
                    depth[silhouette == 0] = 0
                    normal[silhouette == 0, :] = 0

                    cam_params = {'scale': mesh_data['scale'].tolist(),
                                  'center': mesh_data['center'].tolist(),
                                  'K': K_np.tolist(), 'R': R.detach().cpu().numpy().squeeze(0).tolist(),
                                  't': t_np.tolist()}

                    with open(param_file, 'w') as fp:
                        json.dump(cam_params, fp, indent="\t")

                    image_out = (image / np.max(image) * 255.0).astype(np.uint8)
                    if self.openpose_flag:
                        datum, _ = self.pose_predictor(image_out)
                        json_file = os.path.join(self.save_root, 'POSE', data, '%d_%d_%02d.json' % (vid, p, 0))
                        self.pose_predictor.save2json(datum, save_path=json_file)

                    cv2.imwrite(image_file, image_out)
                    cv2.imwrite(depth_file, (depth * 64).astype(np.uint16))
                    cv2.imwrite(silhouette_file, (silhouette * 255))
                    cv2.imwrite(normal_file, (normal * 255.0))
                    print(data, vid)
                elif self.rendering_method.lower() == 'trimesh':
                    # to use Trimesh for rendering, use Python 3.7.x environment
                    # (pyembree does not work for 3.8+)
                    mask_name = os.path.join(self.save_root, 'MASK/ALIGNED', data, '%d_%d_%02d.png' % (vid, p, 0))
                    front_color_name = os.path.join(self.save_root, 'COLOR/ALIGNED', data,
                                                    '%d_%d_%02d_front.png' % (vid, p, 0))
                    front_depth_name = os.path.join(self.save_root, 'DEPTH/ALIGNED', data,
                                                    '%d_%d_%02d_front.png' % (vid, p, 0))
                    front_normal_name = os.path.join(self.save_root, 'NORMAL/ALIGNED', data,
                                                     '%d_%d_%02d_front.png' % (vid, p, 0))
                    front_depth_unbiased_name = os.path.join(self.save_root, 'DEPTH/ALIGNED_UNBIASED', data,
                                                             '%d_%d_%02d_front.png' % (vid, p, 0))
                    front_normal_unbiased_name = os.path.join(self.save_root, 'NORMAL/ALIGNED_UNBIASED', data,
                                                              '%d_%d_%02d_front.png' % (vid, p, 0))

                    if 'dual' in self.rendering_mode:
                        back_color_name = os.path.join(self.save_root, 'COLOR/ALIGNED', data,
                                                       '%d_%d_%02d_back.png' % (vid, p, 0))
                        back_depth_name = os.path.join(self.save_root, 'DEPTH/ALIGNED', data,
                                                       '%d_%d_%02d_back.png' % (vid, p, 0))
                        back_normal_name = os.path.join(self.save_root, 'NORMAL/ALIGNED', data,
                                                        '%d_%d_%02d_back.png' % (vid, p, 0))
                        back_depth_unbiased_name = os.path.join(self.save_root, 'DEPTH/ALIGNED_UNBIASED', data,
                                                                '%d_%d_%02d_back.png' % (vid, p, 0))
                        back_normal_unbiased_name = os.path.join(self.save_root, 'NORMAL/ALIGNED_UNBIASED', data,
                                                                 '%d_%d_%02d_back.png' % (vid, p, 0))

                    if 'smpl' in self.rendering_mode:
                        os.makedirs(os.path.join(self.save_root, 'MASK/ALIGNED_SMPLX', data), exist_ok=True)
                        os.makedirs(os.path.join(self.save_root, 'DEPTH/ALIGNED_SMPLX', data), exist_ok=True)
                        os.makedirs(os.path.join(self.save_root, 'DEPTH/ALIGNED_UNBIASED_SMPLX', data), exist_ok=True)
                        os.makedirs(os.path.join(self.save_root, 'NORMAL/ALIGNED_SMPLX', data), exist_ok=True)
                        os.makedirs(os.path.join(self.save_root, 'NORMAL/ALIGNED_UNBIASED_SMPLX', data), exist_ok=True)
                        mask_name = mask_name.replace('/ALIGNED', '/ALIGNED_SMPLX')

                        front_depth_name = front_depth_name.replace('/ALIGNED', '/ALIGNED_SMPLX')
                        front_normal_name = front_normal_name.replace('/ALIGNED', '/ALIGNED_SMPLX')
                        front_depth_unbiased_name \
                            = front_depth_unbiased_name.replace('/ALIGNED_UNBIASED', '/ALIGNED_UNBIASED_SMPLX')
                        front_normal_unbiased_name \
                            = front_normal_unbiased_name.replace('/ALIGNED_UNBIASED', '/ALIGNED_UNBIASED_SMPLX')
                        if 'dual' in self.rendering_mode:
                            back_color_name = back_color_name.replace('/ALIGNED', '/ALIGNED_SMPLX')
                            back_depth_name = back_depth_name.replace('/ALIGNED', '/ALIGNED_SMPLX')
                            back_normal_name = back_normal_name.replace('/ALIGNED', '/ALIGNED_SMPLX')
                            back_depth_unbiased_name \
                                = back_depth_unbiased_name.replace('/ALIGNED_UNBIASED', '/ALIGNED_UNBIASED_SMPLX')
                            back_normal_unbiased_name \
                                = back_normal_unbiased_name.replace('/ALIGNED_UNBIASED', '/ALIGNED_UNBIASED_SMPLX')

                    os.makedirs(os.path.join(self.save_root, 'PARAM/ALIGNED', data), exist_ok=True)
                    param_file = os.path.join(self.save_root, 'PARAM/ALIGNED', data, '%d_%d_%02d.json' % (vid, p, 0))

                    if not os.path.isfile(param_file):
                        cam_params = {'scale': mesh_data['scale'].tolist(),
                                      'center': mesh_data['center'].tolist(),
                                      'K': K_np.tolist(), 'R': R.detach().cpu().numpy().squeeze(0).tolist(),
                                      't': t_np.tolist()}

                        with open(param_file, 'w') as fp:
                            json.dump(cam_params, fp, indent="\t")

                    if 'dual' in self.rendering_mode:
                        if self.skip_exist and os.path.isfile(back_color_name) and \
                                os.path.isfile(front_depth_name) and os.path.isfile(back_depth_name) and \
                                os.path.isfile(front_depth_unbiased_name) and os.path.isfile(back_depth_unbiased_name):
                            continue
                            # os.path.isfile(front_color_name)
                    else:
                        if self.skip_exist and \
                                os.path.isfile(front_depth_name) and os.path.isfile(front_depth_unbiased_name):
                            continue
                            # os.path.isfile(front_color_name) is missed

                    # check why R.transpose() results in different results.
                    R_trans = -R.cpu().squeeze(0).numpy()
                    if 'smpl' in self.rendering_mode:
                        vertices = np.matmul(mesh_data['smpl_mesh'].vertices, R_trans)
                        mesh = trimesh.Trimesh(
                            vertices=vertices, faces=mesh_data['smpl_mesh'].faces, use_embree=True)
                        # 10475 x 4 x 4 (for each subdivide() call)
                        mesh = trimesh.smoothing.filter_laplacian(mesh.subdivide().subdivide())
                    else:
                        vertices = np.matmul(mesh_data['vertices'], R_trans)
                        mesh = trimesh.Trimesh(
                            vertices=vertices, faces=mesh_data['faces'].squeeze().cpu().numpy(),
                            face_colors=mesh_data['face_colors'].squeeze().cpu().numpy(), use_embree=True)

                    scene = mesh.scene()
                    scene.camera.focal = [self.cam.focal_x, self.cam.focal_y]
                    scene.camera.resolution = [self.cam.width, self.cam.height]
                    # if self.cam.cmax - self.cam.cmin > 0:
                    #     random_dist = np.random.randint(self.cam.cmin,
                    #                                     self.cam.cmax, 1)
                    # else:
                    random_dist = self.cam.cmin
                    cam_trans = np.copy(scene.camera_transform)
                    cam_trans[2, 3], cam_trans[0:2, 3] = random_dist[-1], 0.0

                    scene.camera_transform = cam_trans
                    scene.camera.z_far = self.cam.far
                    scene.camera.z_near = self.cam.near

                    if 'dual' in self.rendering_mode:
                        pers_color_front, pers_depth_front, pers_depth_front_unbiased, \
                        pers_color_back, pers_depth_back, pers_depth_back_unbiased = \
                            pers_get_depth_maps(mesh, scene, self.cam.width)
                    else:
                        pers_color_front, pers_depth_front, pers_depth_front_unbiased \
                            = pers_get_depth_map(mesh, scene, self.cam.width)

                    INT15_MAX = 32767.0
                    depth_front = torch.Tensor((pers_depth_front - INT15_MAX)/128 + random_dist[-1]).unsqueeze(0)
                    # normal_front = get_normal(depth_front, normalize=False)
                    # normal_front = get_normal(depth_front, # pred_res=self.cam.width,
                    #                           real_dist=self.cam.center[-1]) #, z_real=False)
                    # pers_normal_front = normal_front.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    # pers_normal_front = cv2.medianBlur(pers_normal_front.astype(np.uint8), 3)
                    depth_front_unbiased = torch.Tensor((pers_depth_front_unbiased - INT15_MAX)/128
                                                        + random_dist[-1]).unsqueeze(0)
                    # normal_front_unbiased = get_normal(depth_front_unbiased, normalize=False)
                    # normal_front_unbiased = get_normal(depth_front_unbiased, # pred_res=self.cam.width,
                    #                                    real_dist=self.cam.center[-1]) #, z_real=False)
                    # normal_front_unbiased = get_normal(z=depth_front_unbiased, pred_res=self.cam.width,
                    #                                    real_dist=self.cam.center, z_real=False)
                    # normal_front_unbiased = normal_front_unbiased.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    # normal_front_unbiased = cv2.medianBlur(normal_front_unbiased.astype(np.uint8), 3)
                    silhouette = np.zeros_like(pers_depth_front)
                    silhouette[pers_depth_front > 0] = 255

                    cv2.imwrite(mask_name, silhouette.astype(np.uint8))
                    if 'smpl' not in self.rendering_mode:
                        cv2.imwrite(front_color_name, (pers_color_front * 255).astype(np.uint8))
                    cv2.imwrite(front_depth_name, pers_depth_front.astype(np.uint16))
                    cv2.imwrite(front_depth_unbiased_name, pers_depth_front_unbiased.astype(np.uint16))
                    # cv2.imwrite(front_normal_name, pers_normal_front.astype(np.uint8))
                    # cv2.imwrite(front_normal_unbiased_name, normal_front_unbiased.astype(np.uint8))

                    if 'dual' in self.rendering_mode:
                        pers_depth_back = cv2.medianBlur(pers_depth_back, 3)
                        pers_depth_back_unbiased = cv2.medianBlur(pers_depth_back_unbiased, 3)
                        depth_back = torch.Tensor((pers_depth_back - INT15_MAX)/128 + random_dist[-1]).unsqueeze(0)
                        # normal_back = get_normal(depth_back, normalize=False)
                        # normal_back = get_normal(z=depth_back, pred_res=self.cam.width,
                        #                          real_dist=self.cam.center, z_real=False)
                        # pers_normal_back = normal_back.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                        # pers_normal_back = cv2.medianBlur(pers_normal_back.astype(np.uint8), 3)
                        depth_back_unbiased = torch.Tensor((pers_depth_back_unbiased - INT15_MAX)/128 + random_dist[-1]).unsqueeze(0)
                        # normal_back_unbiased = get_normal(depth_back_unbiased, normalize=False)
                        # normal_back_unbiased = get_normal(z=depth_back_unbiased, pred_res=self.cam.width,
                        #                                   real_dist=self.cam.center, z_real=False)
                        # normal_back_unbiased = normal_back_unbiased.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                        # normal_back_unbiased = cv2.medianBlur(normal_back_unbiased.astype(np.uint8), 3)
                        if 'smpl' not in self.rendering_mode:
                            cv2.imwrite(back_color_name, (pers_color_back * 255).astype(np.uint8))
                        cv2.imwrite(back_depth_name, pers_depth_back.astype(np.uint16))
                        cv2.imwrite(back_depth_unbiased_name, pers_depth_back_unbiased.astype(np.uint16))
                        # cv2.imwrite(back_normal_name, pers_normal_back.astype(np.uint8))
                        # cv2.imwrite(back_normal_unbiased_name, normal_back_unbiased.astype(np.uint8))
                elif self.rendering_method.lower() == 'opengl':
                    mask_name = os.path.join(self.save_root, 'MASK/OPENGL', data, '%d_%d_%02d.png' % (vid, p, 0))
                    rndr = mesh_data['rndr']
                    rndr_uv = mesh_data['rndr_uv']
                    R[0, 1, 1] *= (-1)
                    R[0, 2, 2] *= (-1)
                    R[0, 2, 0] *= (-1)

                    for j in range(self.light_num):
                        sh_id = random.randint(0, shs.shape[0] - 1)
                        sh = shs[sh_id]
                        sh_angle = 0.2 * np.pi * (random.random() - 0.5)
                        sh = rotateSH(sh, make_rotate(0, sh_angle, 0).T)

                        rndr.rot_matrix = R.cpu().numpy()
                        rndr_uv.rot_matrix = R.cpu().numpy()
                        rndr.set_camera(self.cam)
                        rndr_uv.set_camera(self.cam)

                        dic = {'scale': mesh_data['scale'].tolist(),
                               'center': mesh_data['center'].tolist(),
                               'K': np.asarray(K.detach().cpu()).tolist(),
                               'R': np.asarray(R.transpose(1, 0).cpu()).tolist(),
                               't': np.asarray(t.detach().cpu()).tolist(),
                               'sh': np.asarray(sh).tolist()}
                        os.makedirs(os.path.join(self.save_root, 'PARAM/OPENGL', data), exist_ok=True)
                        with open(os.path.join(self.save_root, 'PARAM/OPENGL', data, '%d_%d_%02d.json' % (vid, p, j)), 'w') as fp:
                            json.dump(dic, fp, indent="\t")

                        rndr.set_camera(self.cam)
                        rndr.set_sh(sh)
                        rndr.analytic = False
                        rndr.use_inverse_depth = False
                        rndr.display()

                        out_all_f = rndr.get_color(0)
                        out_mask = out_all_f[:, :, 3]
                        out_all_f = cv2.cvtColor(out_all_f, cv2.COLOR_RGBA2BGR)
                        # out_normal_f = rndr.get_color(1)
                        # out_normal_f = cv2.cvtColor(out_normal_f, cv2.COLOR_RGBA2BGRA)
                        color_name = os.path.join(self.save_root, 'COLOR/OPENGL', data,
                                                  '%d_%d_%02d.jpg' % (vid, p, j))
                        uv_render_name = os.path.join(self.save_root, 'UV_RENDER/OPENGL', data,
                                                      '%d_%d_%02d.jpg' % (vid, p, j))
                        cv2.imwrite(color_name, (255.0 * out_all_f).astype(np.int64))
                        if j == 0:
                            cv2.imwrite(mask_name, (255.0 * out_mask))
                            # cv2.imwrite(os.path.join(self.save_root, 'NORMAL/OPENGL', data, '%d_%d_%02d.png' % (vid, p, j)),
                            #             (255.0 * out_normal_f).astype(np.uint8))

                        rndr_uv.set_sh(sh)
                        rndr_uv.analytic = False
                        rndr_uv.use_inverse_depth = False
                        rndr_uv.display()

                        uv_color = rndr_uv.get_color(0)
                        uv_color = cv2.cvtColor(uv_color, cv2.COLOR_RGBA2BGR)
                        cv2.imwrite(uv_render_name, (255.0 * uv_color).astype(np.int64))

                        if vid == 0 and j == 0 and p == 0:
                            uv_pos = rndr_uv.get_color(1)
                            uv_mask = uv_pos[:, :, 3]
                            cv2.imwrite(uv_mask_name, (255.0 * uv_mask).astype(np.int64))

                            pos_data = {'default': uv_pos[:, :, :3]}  # default is a reserved name
                            pyexr.write(uv_pos_name, pos_data)

                            uv_nml = rndr_uv.get_color(2)
                            uv_nml = cv2.cvtColor(uv_nml, cv2.COLOR_RGBA2BGR)
                            cv2.imwrite(uv_normal_name, (255.0 * uv_nml).astype(np.int64))


    @staticmethod
    def parse_calib(calib_params):
        return calib_params['K'], calib_params['R'], calib_params['t']

    def forward(self, verts, faces, K, R, t, textr=None):
        if textr is None:
            textr = torch.ones_like(faces).to(self.device)
            textr = textr.unsqueeze(2).unsqueeze(2).unsqueeze(2).float()

        image, depth, mask = self.renderer(verts, faces, textr,
                                           K=K, R=R, t=t,
                                           dist_coeffs=self.cam.dist_params)
        # cv2.imwrite('render_test.jpg', mask.detach().cpu().squeeze().numpy() * 255)
        return image, depth, mask


def pers_get_depth_map(mesh, scene, res):
    mesh.scene = scene
    pers_origins, pers_vectors, pers_pixels = mesh.scene.camera_rays()
    pers_points, pers_index_ray, pers_index_tri = mesh.ray.intersects_location(pers_origins,
                                                                               pers_vectors,
                                                                               multiple_hits=False)
    # (A: pers_points)  ----------> (pers_origins[0] -> surface)
    # (B: pers_vectors) ->          (same vector with unit norm)
    # A dot B = distance (cos(theta)|A||B| = cos(theta) = 1, A = depth * B)
    pers_depth = trimesh.util.diagonal_dot(pers_points - pers_origins[0],
                                           pers_vectors[pers_index_ray])
    unbiased_depth = (pers_points - pers_origins[0])[:, 2]
    pers_colors = mesh.visual.face_colors[pers_index_tri] / 255.0

    # 128. retains 7bit sub-pixel precision, 32767.0 to centering for unsigned 16 data type.
    pers_depth = (pers_depth - pers_origins[0][2]) * 128.0 + 32767.0
    unbiased_depth = (unbiased_depth - pers_origins[0][2]) * 128.0 + 32767.0

    pers_pixel_ray = pers_pixels[pers_index_ray]

    INT16_MAX = 65536
    pers_depth_near = np.ones(mesh.scene.camera.resolution, dtype=np.float32) * INT16_MAX
    pers_depth_near_unbiased = np.ones(mesh.scene.camera.resolution, dtype=np.float32) * INT16_MAX
    pers_color_near = np.zeros((res, res, 3), dtype=np.float32)

    for k in range(pers_pixel_ray.shape[0]):
        u, v = pers_pixel_ray[k, 0], pers_pixel_ray[k, 1]
        if pers_depth[k] < pers_depth_near[v, u]:
            pers_depth_near[v, u] = pers_depth[k]
            pers_depth_near_unbiased[v, u] = unbiased_depth[k]
            pers_color_near[v, u, ::-1] = pers_colors[k, 0:3]

    pers_depth_near = pers_depth_near * (pers_depth_near != INT16_MAX)
    pers_depth_near_unbiased = pers_depth_near_unbiased * (pers_depth_near_unbiased != INT16_MAX)
    pers_color_near = np.flip(pers_color_near, 0)
    pers_depth_near = np.flip(pers_depth_near, 0)
    pers_depth_near_unbiased = np.flip(pers_depth_near_unbiased, 0)
    pers_color_near = np.flip(pers_color_near, 1)
    pers_depth_near = np.flip(pers_depth_near, 1)
    pers_depth_near_unbiased = np.flip(pers_depth_near_unbiased, 1)

    return pers_color_near, pers_depth_near, pers_depth_near_unbiased


def pers_get_depth_maps(mesh, scene, res, scaling_factor=128.0):
    mesh.scene = scene
    pers_origins, pers_vectors, pers_pixels = mesh.scene.camera_rays()
    pers_points, pers_index_ray, pers_index_tri = mesh.ray.intersects_location(pers_origins,
                                                                               pers_vectors,
                                                                               multiple_hits=True)
    # (A: pers_points_origin)  ----------> (pers_origins[0] -> surface)
    # (B: pers_unit_vector)    ->          (same vector with unit norm)
    # A dot B = cos(theta)|A||B| = |A||B| = distance
    # s.t. cos(theta) = 1, |B| = 1, |A|=distance x |B|
    pers_depth = trimesh.util.diagonal_dot(pers_points - pers_origins[0],
                                           pers_vectors[pers_index_ray])
    unbiased_depth = np.abs(pers_points - pers_origins[0])[:, 2]
    pers_colors = mesh.visual.face_colors[pers_index_tri] / 255.0

    # 128. retains 7bit sub-pixel precision, 32767.0 to centering for unsigned 16 data type.
    # align depth maps to the plane (z=0)
    pers_depth = (pers_depth - pers_origins[0][2]) * scaling_factor + 32767.0
    unbiased_depth = (unbiased_depth - pers_origins[0][2]) * scaling_factor + 32767.0

    pers_pixel_ray = pers_pixels[pers_index_ray]
    pers_depth_far = np.zeros(mesh.scene.camera.resolution, dtype=np.float32)
    pers_depth_far_unbiased = np.zeros(mesh.scene.camera.resolution, dtype=np.float32)
    pers_color_far = np.zeros((res, res, 3), dtype=np.float32)

    INT16_MAX = 65536
    pers_depth_near = np.ones(mesh.scene.camera.resolution, dtype=np.float32) * INT16_MAX
    pers_depth_near_unbiased = np.ones(mesh.scene.camera.resolution, dtype=np.float32) * INT16_MAX
    pers_color_near = np.zeros((res, res, 3), dtype=np.float32)

    for k in range(pers_pixel_ray.shape[0]):
        u, v = pers_pixel_ray[k, 0], pers_pixel_ray[k, 1]
        if pers_depth[k] > pers_depth_far[v, u]:
            pers_color_far[v, u, ::-1] = pers_colors[k, 0:3]
            pers_depth_far[v, u] = pers_depth[k]
            pers_depth_far_unbiased[v, u] = unbiased_depth[k]
        if pers_depth[k] < pers_depth_near[v, u]:
            pers_depth_near[v, u] = pers_depth[k]
            pers_depth_near_unbiased[v, u] = unbiased_depth[k]
            pers_color_near[v, u, ::-1] = pers_colors[k, 0:3]

    pers_depth_near = pers_depth_near * (pers_depth_near != INT16_MAX)
    pers_depth_near_unbiased = pers_depth_near_unbiased * (pers_depth_near_unbiased != INT16_MAX)
    pers_color_near = np.flip(pers_color_near, 0)
    pers_depth_near = np.flip(pers_depth_near, 0)
    pers_depth_near_unbiased = np.flip(pers_depth_near_unbiased, 0)
    pers_color_far = np.flip(pers_color_far, 0)
    pers_depth_far = np.flip(pers_depth_far, 0)
    pers_depth_far_unbiased = np.flip(pers_depth_far_unbiased, 0)

    pers_color_near = np.flip(pers_color_near, 1)
    pers_depth_near = np.flip(pers_depth_near, 1)
    pers_depth_near_unbiased = np.flip(pers_depth_near_unbiased, 1)
    pers_color_far = np.flip(pers_color_far, 1)
    pers_depth_far = np.flip(pers_depth_far, 1)
    pers_depth_far_unbiased = np.flip(pers_depth_far_unbiased, 1)

    return pers_color_near, pers_depth_near, pers_depth_near_unbiased, \
           pers_color_far, pers_depth_far, pers_depth_far_unbiased
