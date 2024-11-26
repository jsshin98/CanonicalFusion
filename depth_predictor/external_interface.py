import math
import os.path
import cv2
import trimesh
import torch.utils.data
import math
import time
from basicsr.archs.rrdbnet_arch import RRDBNet
from depth_predictor.utils.loader_utils import *
from depth_predictor.utils.core.im_utils import get_plane_params
from torchmcubes import grid_interp
from reconstructor import models
# import reconstructor.recon_utils as recon_utils
from lbs_handler.model import LBSModel
import smplx
from pysdf import SDF
from torchmcubes import marching_cubes, grid_interp
from typing import Tuple
import open3d as o3d
from sklearn.neighbors import KDTree

class HumanRecon(nn.Module):
    """Implementation of single-stage SMPLify."""
    def __init__(self,
                 result_path='',
                 ckpt_path='',
                 color_ckpt_path='',
                 model_name='',
                 model_C_name='',
                 esr_path='',
                 lbs_ckpt='',
                 v_label=None,
                 cam_params=None,
                 params=None,
                 device=torch.device('cuda:0')):
        super(HumanRecon, self).__init__()
        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)
        
        self.res = cam_params['width'] # or 'height'
        self.voxel_size = cam_params['voxel_size']
        self.z_min = cam_params['real_dist'] - 64
        self.z_max = cam_params['real_dist'] + 64
        self.px = cam_params['px']
        self.py = cam_params['py']
        self.fx = cam_params['fx']
        self.fy = cam_params['fy']
        self.real_dist = cam_params['real_dist']
        self.camera_height = cam_params['cam_center'][1]
        self.v_label = v_label
        self.device = torch.device(device)

        # load pre-trained model
        self.model = getattr(models, model_name)(split_last=True)
        self.model_C = getattr(models, model_C_name)(split_last=True)

        self.model.to(self.device)
        self.model_C.to(self.device)

        self.load_checkpoint([ckpt_path],
                             self.model,
                             is_evaluate=True,
                             device=device)

        self.load_checkpoint([color_ckpt_path],
                             self.model_C,
                             is_evaluate=True,
                             device=device)
        # Real-esrGAN
        esr_model_path = esr_path
        self.esr_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        loadnet = torch.load(esr_model_path, map_location=device)
        # prefer to use params_ema
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        self.esr_model.load_state_dict(loadnet[keyname], strict=True)
        self.esr_model.eval()
        self.esr_model = self.esr_model.to(device)

        self.RGB_MEAN = [0.485, 0.456, 0.406]
        self.RGB_STD = [0.229, 0.224, 0.225]
        self.RGB_MAX = [255.0, 255.0, 255.0]
        self.RGB_MG = [10.0, 10.0, 10.0]
        self.DEPTH_SCALE = 128.0
        self.DEPTH_MAX = 32767
        self.DEPTH_EPS = 0.5
        self.scale_factor = 1.0
        self.offset = 0
        x = np.reshape((np.linspace(0, self.res, self.res) - int(self.px)) / self.fx,
                       [1, 1, 1, -1])
        y = np.reshape((np.linspace(0, self.res, self.res) - int(self.res - self.py)) / self.fy,
                       [1, 1, -1, 1])

        x = np.tile(x, [1, 1, self.res, 1])
        y = np.tile(y, [1, 1, 1, self.res])
        self.xy = torch.Tensor(np.concatenate((y, x), axis=1)).cuda()
        self.coord = self.gen_volume_coordinate(xy=self.xy[0],
                                                z_min=self.z_min,
                                                z_max=self.z_max,
                                                voxel_size=self.voxel_size)
        # self.cosine_loss = nn.CosineSimilarity(dim=2)
        self.lbs_model = LBSModel().cuda()
        self.lbs_model.load_state_dict(torch.load(os.path.join(lbs_ckpt, 'best.tar'))['state_dict'])
        self.lbs_model.eval()

    @staticmethod
    def cosine_loss(x, y):
        return torch.mean(1.0 - torch.sum((x * y), dim=1))

    @staticmethod
    def load_checkpoint(model_paths, model,
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
                start_epoch = checkpoint['epoch'] #+ 1

                if hasattr(model, 'module'):
                    model_state_dict = checkpoint['model_state_dict']
                else:
                    model_state_dict = collections.OrderedDict(
                        {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})

                model.load_state_dict(model_state_dict, strict=False)

                print(("=> loaded checkpoint (resumed epoch is {})".format(start_epoch)))
                return model

        print(("=> no checkpoint found at '{}'".format(model_path)))
        return model

    @staticmethod
    def gen_volume_coordinate(xy, z_min=120, z_max=320, voxel_size=512):
        grid = torch.ones((3, xy.shape[1], xy.shape[2], voxel_size))
        z_range = z_max - z_min
        slope = z_range / voxel_size
        ones = torch.ones_like(xy[0:1, :, :])
        for k in range(voxel_size):
            z = z_min + slope * k
            grid[:, :, :, k] = torch.cat((xy * z, ones * z), dim=0)
        return grid

    def cube_sdf(self, x_nx3):
        sdf_values = 0.5 - torch.abs(x_nx3)
        sdf_values = torch.clamp(sdf_values, min=0.0)
        sdf_values = sdf_values[:, 0] * sdf_values[:, 1] * sdf_values[:, 2]
        sdf_values = -1.0 * sdf_values

        return sdf_values

    def cube_sdf_gradient(self, x_nx3):
        gradients = []
        for i in range(x_nx3.shape[0]):
            x, y, z = x_nx3[i]
            grad_x, grad_y, grad_z = 0, 0, 0

            max_val = max(abs(x) - 0.5, abs(y) - 0.5, abs(z) - 0.5)

            if max_val == abs(x) - 0.5:
                grad_x = 1.0 if x > 0 else -1.0
            if max_val == abs(y) - 0.5:
                grad_y = 1.0 if y > 0 else -1.0
            if max_val == abs(z) - 0.5:
                grad_z = 1.0 if z > 0 else -1.0

            gradients.append(torch.tensor([grad_x, grad_y, grad_z]))

        return torch.stack(gradients).to(x_nx3.device)

    @staticmethod
    def get_mesh(volume, grid_coord, scale_factor=1.0):
        # mesh generation.
        if isinstance(volume, np.ndarray):
            volume = torch.Tensor(volume)
        vertices, faces = marching_cubes(volume, 0.0)
        new_vertices = torch.Tensor(vertices.detach().cpu().numpy()[:, ::-1].copy())
        new_vertices = grid_interp(grid_coord.contiguous(), new_vertices)
        new_mesh = trimesh.Trimesh(new_vertices / scale_factor, faces)
        return new_mesh

    def _get_grid_coord_(self, v_min, v_max, res):
        x_ind, y_ind, z_ind = torch.meshgrid(torch.linspace(v_min[0], v_max[0], res[0]),
                                             torch.linspace(v_min[1], v_max[1], res[1]),
                                             torch.linspace(v_min[2], v_max[2], res[2]), indexing='ij')
        grid = torch.stack((x_ind, y_ind, z_ind), dim=0)
        grid = grid.float()
        pt = np.concatenate((np.asarray(x_ind).reshape(-1, 1),
                             np.asarray(y_ind).reshape(-1, 1),
                             np.asarray(z_ind).reshape(-1, 1)), axis=1)
        pt = pt.astype(float)
        return pt, grid.permute(0, 3, 2, 1).contiguous()

    def psnr(self, img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        # print("mse : ", mse)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    def euler_to_rot_mat(self, r_x, r_y, r_z):
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(r_x), -math.sin(r_x)],
                        [0, math.sin(r_x), math.cos(r_x)]
                        ])

        R_y = np.array([[math.cos(r_y), 0, math.sin(r_y)],
                        [0, 1, 0],
                        [-math.sin(r_y), 0, math.cos(r_y)]
                        ])

        R_z = np.array([[math.cos(r_z), -math.sin(r_z), 0],
                        [math.sin(r_z), math.cos(r_z), 0],
                        [0, 0, 1]
                        ])

        R = np.dot(R_z, np.dot(R_y, R_x))

        return R
    def make_rotation_matrix(self, rx, ry, rz):
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

    def volume_filter(self, volume, k_size=3, iter=3):
        # k_size must be an odd number
        if isinstance(volume, np.ndarray):
            volume = torch.Tensor(volume)
        filters = torch.ones(1, 1, k_size, k_size, k_size) / (k_size*k_size*k_size)  # average filter
        volume = volume.unsqueeze(0)
        for _ in range(iter):
            volume = F.conv3d(volume, filters, padding=k_size//2)
        return volume.squeeze().detach().cpu().numpy()
    
    def init_variables(self, datum, device=None):
        input_color, input_mask, input_depth = datum['input']

        if 'label' in datum:
            target_color, target_depth = datum['label']
        else:
            target_color, target_depth = None, None

        if device is not None:
            if input_color is not None:
                input_color = input_color.unsqueeze(0).to(device)
            if input_depth is not None:
                input_depth = input_depth.unsqueeze(0).to(device)
            if input_mask is not None:
                input_mask = input_mask.unsqueeze(0).to(device)
            if target_color is not None:
                target_color = target_color.to(device)
            if target_depth is not None:
                target_depth = target_depth.to(device)

        if input_color is not None:
            input_color = torch.autograd.Variable(input_color)
        if input_depth is not None:
            input_depth = torch.autograd.Variable(input_depth)
        if input_mask is not None:
            input_mask = torch.autograd.Variable(input_mask)
        if target_depth is not None:
            target_depth = torch.autograd.Variable(target_depth)
        if target_color is not None:
            target_color = torch.autograd.Variable(target_color)

        input_var = (input_color, input_mask, input_depth)
        if target_color is not None and target_depth is not None:
            target_var = (target_color, target_depth)
        else:
            target_var = None

        return input_var, target_var

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
        mask = np.zeros(total_num_faces, dtype=bool)
        cc = np.concatenate([
            c for c in cc if len(c) > num_faces
        ], axis=0)
        mask[cc] = True
        mesh.update_faces(mask)

        return mesh

    @torch.no_grad()
    def evaluate(self, input_var, model, model_C):
        model.eval()
        model_C.eval()
        
        start_full = time.time()
        start = time.time()
        pred_var = model(torch.cat([input_var[0], input_var[1], input_var[2]], dim=1))
    
        x = np.reshape((np.linspace(0, self.res, self.res) - int(self.res / 2)) / self.fx,
                       [1, 1, -1, 1])
        y = np.reshape((np.linspace(0, self.res, self.res) - int(self.res / 2)) / self.fx,
                       [1, 1, 1, -1])
        x = np.tile(x, [1, 1, 1, self.res])
        y = np.tile(y, [1, 1, self.res, 1])
        xy = torch.Tensor(np.concatenate((x, y), axis=1)).to(self.device)

        pred_df, pred_db = torch.chunk(pred_var['pred_depth'], chunks=pred_var['pred_depth'].shape[1], dim=1)
        
        predfd2n = get_plane_params(z=pred_df, xy=xy,
                                    pred_res=self.res, real_dist=self.real_dist,
                                    z_real=True, v_norm=True)
        predbd2n = get_plane_params(z=pred_db, xy=xy,
                                    pred_res=self.res, real_dist=self.real_dist,
                                    z_real=True, v_norm=True)
        pred_depth2normal = torch.cat([predfd2n[:, 0:3, :, :],
                                       predbd2n[:, 0:3, :, :]], dim=1)

        input = torch.cat([input_var[0], input_var[1], pred_depth2normal], dim=1)#.detach()
        
        pred_color = model_C(input)['pred_color']
        end = time.time()

        cf, cb = torch.chunk(pred_color, chunks=2, dim=1)
        cf = cf * input_var[1]
        cb = cb * input_var[1]
        cf_numpy = cf[0].permute(1, 2, 0).detach().cpu().numpy()
        cb_numpy = cb[0].permute(1, 2, 0).detach().cpu().numpy()
        cf_numpy = cf_numpy * self.RGB_STD + self.RGB_MEAN
        cb_numpy = cb_numpy * self.RGB_STD + self.RGB_MEAN
        cf_numpy[cf_numpy < 0] = 0
        cb_numpy[cb_numpy < 0] = 0

        img_f = torch.from_numpy(np.transpose(cf_numpy[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_b = torch.from_numpy(np.transpose(cb_numpy[:, :, [2, 1, 0]], (2, 0, 1))).float()
        imgf_LR = img_f.unsqueeze(0)
        imgf_LR = imgf_LR.to(self.device)
        imgb_LR = img_b.unsqueeze(0)
        imgb_LR = imgb_LR.to(self.device)

        with torch.no_grad():
            output_cf = self.esr_model(imgf_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output_cb = self.esr_model(imgb_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

        output_cf = np.transpose(output_cf[[2, 1, 0], :, :], (1, 2, 0))
        output_cf = (output_cf * 255.0).round()
        output_cb = np.transpose(output_cb[[2, 1, 0], :, :], (1, 2, 0))
        output_cb = (output_cb * 255.0).round()

        output_cf = np.array(output_cf, dtype=np.uint8)
        output_cb = np.array(output_cb, dtype=np.uint8)

        # generate volume from depth map
        pred_depth = pred_var['pred_depth']
        df, db = torch.chunk(pred_depth, chunks=2, dim=1)
        df = (df - 0.5) * 128 + self.real_dist
        db = (db - 0.5) * 128 + self.real_dist

        df = df * input_var[1]
        db = db * input_var[1]

        volume = depth2occ_2view_torch(df, db, binarize=False,
                                       z_min=self.z_min, z_max=self.z_max,
                                       voxel_size=self.voxel_size)
        volume = self.volume_filter(volume, iter=2)

        pred_lbs = pred_var['pred_lbs']
        lf, lb = torch.chunk(pred_lbs, chunks=2, dim=1)
        lf = lf * input_var[1]
        lb = lb * input_var[1]
        lf_numpy = lf[0].permute(1, 2, 0).detach().cpu().numpy()
        lb_numpy = lb[0].permute(1, 2, 0).detach().cpu().numpy()

        output_lf = np.transpose(np.transpose(lf_numpy[:, :, [2, 1, 0]], (2, 0, 1)), (1, 2, 0))
        output_lb = np.transpose(np.transpose(lb_numpy[:, :, [2, 1, 0]], (2, 0, 1)), (1, 2, 0))
        output_lf = np.clip(output_lf, a_min=0, a_max=1)
        output_lb = np.clip(output_lb, a_min=0, a_max=1)

        lf_1024_f = Image.fromarray((output_lf*255).astype(np.uint8))
        lf_1024_b = Image.fromarray((output_lb*255).astype(np.uint8))
        output_lf = np.array(lf_1024_f.resize((1024, 1024)))
        output_lb = np.array(lf_1024_b.resize((1024, 1024)))

        lbs_color_mesh = colorize_model(volume, output_lf / 255.0, output_lb / 255.0,
                                        mask=input_var[1].squeeze().detach().cpu().numpy(),
                                        texture_map=False)  # , volume_level=0.0)

        lbs_color_mesh = self.postprocess_mesh(lbs_color_mesh, num_faces=50000)

        new_vertices = grid_interp(self.coord, torch.Tensor(lbs_color_mesh.vertices))
        R = self.make_rotation_matrix(0, math.radians(0), math.radians(-90))
        vertices = np.matmul(np.asarray(new_vertices), R.transpose(1, 0))
        vertices[:, 2] *= (-1)
        vertices[:, 1] += self.camera_height  # 60.0 # 308.0/512.0/self.focal*220.0
        vertices[:, 2] += self.real_dist

        lbs_pred_mesh = trimesh.Trimesh(vertices=vertices,
                                        faces=lbs_color_mesh.faces,
                                        visual=lbs_color_mesh.visual)
        lbs_pred_mesh.fix_normals()


        color_mesh = colorize_model(volume, output_cf / 255.0, output_cb / 255.0,
                                        mask=input_var[1].squeeze().detach().cpu().numpy(),
                                        texture_map=False)  # , volume_level=0.0)
        color_mesh = self.postprocess_mesh(color_mesh, num_faces=50000)

        new_vertices = grid_interp(self.coord, torch.Tensor(color_mesh.vertices))
        R = self.make_rotation_matrix(0, math.radians(0), math.radians(-90))
        vertices = np.matmul(np.asarray(new_vertices), R.transpose(1, 0))
        vertices[:, 2] *= (-1)
        vertices[:, 1] += self.camera_height  # 60.0 # 308.0/512.0/self.focal*220.0
        vertices[:, 2] += self.real_dist
        color_pred_mesh = trimesh.Trimesh(vertices=vertices,
                                        faces=color_mesh.faces,
                                        visual=color_mesh.visual)

        color_pred_mesh.fix_normals()
        end_full = time.time()
        process_time = (end - start)
        process_time_full = (end_full - start_full)
        return lbs_pred_mesh, color_pred_mesh, output_cf, output_cb, output_lf, output_lb, process_time, process_time_full


    def forward(self, image, depth_front, depth_back, mask, smpl_params=None, smpl_resource=None, cam_params=None):
        image = torch.Tensor(image).permute(2, 0, 1).float()
        if torch.max(image) > 1.0:
            image = image / torch.Tensor(self.RGB_MAX).view(3, 1, 1)
        image_input = (image - torch.Tensor(self.RGB_MEAN).view(3, 1, 1)) / torch.Tensor(self.RGB_STD).view(3, 1, 1)
                
        mask_input = torch.Tensor(mask).permute(2, 0, 1).float()
        if torch.max(mask_input) > 1.0:
            mask_input = mask_input / 255.0

        front_depth = torch.Tensor(depth_front.copy()).permute(2, 0, 1).float()
        back_depth = torch.Tensor(depth_back.copy()).permute(2, 0, 1).float()
        depth_input = torch.cat([front_depth, back_depth], dim=0)

        datum = dict()
        datum['input'] = (image_input, mask_input, depth_input)
        input_var, target_var = self.init_variables(datum, device=self.device)

        lbs_pred_mesh, color_pred_mesh, color_front, color_back, lbs_front, lbs_back, process_time, process_time_full\
            = self.evaluate(input_var, self.model, self.model_C)
        return (lbs_pred_mesh, color_pred_mesh, color_front, color_back, lbs_front, lbs_back, process_time, process_time_full)

if __name__=='__main__':
    recon = HumanRecon()
