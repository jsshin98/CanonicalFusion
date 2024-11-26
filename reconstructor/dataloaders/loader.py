import os
import numpy as np
import cv2
import torch
import random
import albumentations as albu
import torch.backends.cudnn as cudnn
import json
import trimesh
import pickle
import pdb

from torch.utils.data import Dataset
from PIL import Image
# from torchvision import transforms

from iglovikov_helper_functions.utils.image_utils import pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

# cudnn.benchmark = True
# cudnn.fastest = True
os.environ['PYOPENGL_PLATFORM'] = 'egl'

class AugDataSet(Dataset):
    def __init__(self,
                 dataset_path='./SAMPLE/',
                 data_list='train_split',
                 bg_list='bg_list',
                 seg_model=None,
                 pred_res=512,
                 orig_res=512,
                 real_dist=300.0):

        self.data_list = data_list
        self.bg_list = bg_list
        self.dataset_path = dataset_path
        self.pred_res = pred_res
        self.orig_res = orig_res
        self.real_dist = real_dist

        self.DEPTH_SCALE = 128.0
        self.DEPTH_MAX = 32767
        self.DEPTH_EPS = 0.5

        self.RGB_MAX = np.array([255.0, 255.0, 255.0])
        self.RGB_MEAN = np.array([0.485, 0.456, 0.406]) # vgg mean
        self.RGB_STD = np.array([0.229, 0.224, 0.225]) # vgg std

        self.seg_model = seg_model
        self.seg_transform = albu.Compose(
            [albu.LongestMaxSize(max_size=512), albu.Normalize(p=1)], p=1
        )

        f_name = os.path.join(self.dataset_path, 'list', self.data_list + '.txt')
        self.__init_data__(f_name)

        # self.resize = transforms.Resize((self.orig_res, self.orig_res))
    def image2mask(self, image):
        padded_image, pads = pad(image, factor=int(self.pred_res / 2), border=cv2.BORDER_CONSTANT)
        #x = torch.unsqueeze(tensor_from_rgb_image(padded_image), 0).cuda()
        x = torch.unsqueeze(tensor_from_rgb_image(padded_image), 0)

        with torch.no_grad():
            prediction = self.seg_model(x)[0][0]

        mask = (prediction > 0).cpu().numpy().astype(np.uint8)
        mask = unpad(mask, pads)
        if not image.shape[1] == self.pred_res:
            mask = cv2.resize(mask, (self.pred_res, self.pred_res), interpolation=cv2.INTER_AREA)
        #tensor_mask = torch.Tensor(mask[None, :, :]).cuda().float()
        tensor_mask = torch.Tensor(mask[None, :, :]).float()
        return tensor_mask

    def composite_image(self, image_front, mask_front):
        c, h, w = image_front.size()

        for k in range(200):
            image_idx = random.randrange(0, self.bg_total, 1)
            if os.path.isfile(os.path.join(self.dataset_path, self.bg_image[image_idx])):
                bg_image = Image.open(os.path.join(self.dataset_path, self.bg_image[image_idx]))
                bg_image = np.array(bg_image)
                if bg_image is not None and mask_front is not None and \
                        bg_image.shape[0] > w and bg_image.shape[1] > h:
                    bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
                    bg_image = cv2.resize(bg_image, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
                    bg_image = torch.Tensor(bg_image).permute(2, 0, 1).float()
                    bg_image_front = bg_image / torch.Tensor(self.RGB_MAX).view(3, 1, 1)

                    condition_front = mask_front[0, :, :] > 0
                    blur = cv2.GaussianBlur(mask_front.transpose(1, 2, 0), (3, 3), 0)
                    b_idx = (blur > 0.2) * (blur < 0.8)

                    bg_image_front[0:3, condition_front] = image_front[0:3, condition_front]
                    bg_image_front = bg_image_front.permute(1, 2, 0).numpy()
                    filtered_front = cv2.medianBlur(bg_image_front, 3)
                    filtered_front = cv2.GaussianBlur(filtered_front, (5, 5), 0)
                    bg_image_front[b_idx, :] = filtered_front[b_idx, :]

                    # bg_image_front = self.transform_color(bg_image_front)
                    bg_image_front = (bg_image_front - self.RGB_MEAN) / self.RGB_STD
                    #bg_image_front = torch.Tensor(bg_image_front).permute(2, 0, 1).cuda()
                    bg_image_front = torch.Tensor(bg_image_front).permute(2, 0, 1)

                    bg_image_aft_front = bg_image_front.permute(1, 2, 0).detach().cpu().numpy()
                    bg_image_aft_front = (bg_image_aft_front * self.RGB_STD + self.RGB_MEAN) * self.RGB_MAX
                    # cv2.imwrite('./bg_image_front.png', bg_image_aft_front)

                    # using Graphonomy 2019 model
                    # PIL_image = Image.fromarray(np.uint8(bg_image_aft_front))
                    # mask_front = self.inference(net=self.segment_net, img=PIL_image)

                    # using people-segmentation model
                    image_front = self.seg_transform(image=bg_image_aft_front)["image"]
                    tensor_mask = self.image2mask(image_front)

                    return bg_image_front, tensor_mask
        #return image_front.cuda(), torch.Tensor(mask_front).cuda().float()
        return image_front, torch.Tensor(mask_front).float()

    def file2image(self, file):
        image = np.array(Image.open(self.dataset_path + file))

        if not image.shape[1] == self.pred_res:
            image = cv2.resize(image, dsize=(self.pred_res, self.pred_res), interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        tensor_image = torch.Tensor(image).permute(2, 0, 1).float()
        tensor_image /= torch.Tensor(self.RGB_MAX).view(3, 1, 1)
        return tensor_image

    def file2mask(self, file):
        mask = np.array(Image.open(self.dataset_path + file)).astype('float32')
        if not mask.shape[1] == self.pred_res:
            mask = cv2.resize(mask, dsize=(self.pred_res, self.pred_res), interpolation=cv2.INTER_NEAREST)
        mask /= 255.0
        #tensor_mask = torch.Tensor(mask[None, :, :]).cuda().float()
        tensor_mask = torch.Tensor(mask[None, :, :]).float()

        return tensor_mask

    def file2smpl(self, file):
        if os.path.isfile(self.dataset_path + file):
            smpl_data = np.load(self.dataset_path + file, allow_pickle=True)
            A_inv = torch.Tensor(smpl_data.item()['A_inv']).squeeze()
            A = torch.Tensor(smpl_data.item()['A']).squeeze()
            smpl_vert = torch.Tensor(smpl_data.item()['vertices'])
            centroid_real = torch.Tensor(smpl_data.item()['centroid_real'])
            scale_real = torch.Tensor([smpl_data.item()['scale_real']])
            centroid_smplx = torch.Tensor(smpl_data.item()['centroid_smplx'])
            scale_smplx = torch.Tensor([smpl_data.item()['scale_smplx']])
            scale = torch.Tensor(smpl_data.item()['scale'])
            transl = torch.Tensor(smpl_data.item()['transl'])
            weight = torch.ones((1))
        else:
            A_inv = torch.ones((55, 4, 4))
            A = torch.ones((55, 4, 4))
            smpl_vert = torch.ones((10475, 3))
            centroid_real = torch.ones((3))
            scale_real = torch.ones((1))
            centroid_smplx = torch.ones((3))
            scale_smplx = torch.ones((1))
            scale = torch.ones((1))
            transl = torch.ones((3))
            weight = torch.zeros((1))
        return {'A_inv': A_inv, 'A': A, 'smpl_vert': smpl_vert, 'centroid_real': centroid_real, 'scale_real': scale_real, \
                'centroid_smplx': centroid_smplx, 'scale_smplx': scale_smplx, 'scale': scale, 'transl': transl, 'weight': weight}

    def file2cam(self, file):
        with open(self.dataset_path + file, 'r') as f:
            cam_param = json.load(f)

        scale = torch.Tensor([cam_param['scale']])
        center = torch.Tensor(cam_param['center'])
        K = torch.Tensor(cam_param['K']).squeeze()
        R = torch.Tensor(cam_param['R']).squeeze()
        R[1][1] = -R[1][1]
        R[2][2] = -R[2][2]
        R[2][0] = -R[2][0]
        t = torch.Tensor(cam_param['t']).squeeze()
        return {'scale': scale, 'center': center, 'K': K, 'R': R, 't': t}
    
    def file2lbs(self, file):
        lbs = np.load(self.dataset_path + file)['lbs']
        if not lbs.shape[2] == self.pred_res:
            front_lbs = cv2.resize(lbs[0], dsize=(self.orig_res, self.orig_res), interpolation=cv2.INTER_NEAREST)
            back_lbs = cv2.resize(lbs[1], dsize=(self.orig_res, self.orig_res), interpolation=cv2.INTER_NEAREST)
            lbs = np.stack((front_lbs, back_lbs), axis=0)

        lbs = torch.Tensor(lbs).float()
        return lbs
    
    def preprocess_input(self, image_front, mask_file, smplx_front, smplx_back, bg_flag=False):
        tensor_front = self.file2image(image_front)
        tensor_mask = self.file2mask(mask_file)

        if bg_flag:
            image_scaled_front, mask_front = self.composite_image(tensor_front, tensor_mask.detach().cpu().numpy())
        else:
            #image_scaled_front = tensor_front.cuda()
            #image_scaled_front = (image_scaled_front - torch.Tensor(self.RGB_MEAN).cuda().view(3, 1, 1)) \
            #                     / torch.Tensor(self.RGB_STD).cuda().view(3, 1, 1)
            image_scaled_front = tensor_front
            image_scaled_front = (image_scaled_front - torch.Tensor(self.RGB_MEAN).view(3, 1, 1)) \
                                 / torch.Tensor(self.RGB_STD).view(3, 1, 1)
            mask_front = tensor_mask

        front_depth = np.array(Image.open(self.dataset_path + smplx_front)).astype('float32')
        front_depth = cv2.medianBlur(front_depth, 3)
        if not front_depth.shape[1] == self.orig_res:
            front_depth = cv2.resize(front_depth, dsize=(self.orig_res, self.orig_res), interpolation=cv2.INTER_NEAREST)
        front_depth = (front_depth - self.DEPTH_MAX) / self.DEPTH_SCALE / self.DEPTH_SCALE + self.DEPTH_EPS
        front_depth = np.clip(front_depth, a_min=0, a_max=1)
        front_depth_scaled = np.expand_dims(front_depth, axis=2)
        front_depth_scaled = torch.Tensor(front_depth_scaled).permute(2, 0, 1).float()
        
        back_depth = np.array(Image.open(self.dataset_path + smplx_back)).astype('float32')
        back_depth = cv2.medianBlur(back_depth, 3)
        if not back_depth.shape[1] == self.orig_res:
            back_depth = cv2.resize(back_depth, dsize=(self.orig_res, self.orig_res), interpolation=cv2.INTER_NEAREST)
        back_depth = (back_depth - self.DEPTH_MAX) / self.DEPTH_SCALE / self.DEPTH_SCALE + self.DEPTH_EPS
        back_depth = np.clip(back_depth, a_min=0, a_max=1)
        back_depth_scaled = np.expand_dims(back_depth, axis=2)
        back_depth_scaled = torch.Tensor(back_depth_scaled).permute(2, 0, 1).float()
        return image_scaled_front, mask_front, front_depth_scaled, back_depth_scaled
    
    def preprocess_color_gt(self, color_front, color_back):
        tensor_front = self.file2image(color_front)
        tensor_back = self.file2image(color_back)
        
        image_scaled_front = (tensor_front - torch.Tensor(self.RGB_MEAN).view(3, 1, 1)) \
                            / torch.Tensor(self.RGB_STD).view(3, 1, 1)
        image_scaled_back = (tensor_back - torch.Tensor(self.RGB_MEAN).view(3, 1, 1)) \
                            / torch.Tensor(self.RGB_STD).view(3, 1, 1)
                            
        return image_scaled_front, image_scaled_back

    def preprocess_wrapper(self, depth_file, lbs_file, color_file):
        # target depth -> real scale
        depth = np.array(Image.open(self.dataset_path + depth_file)).astype('float32')
        depth = cv2.medianBlur(depth, 3)

        lbs = np.array(Image.open(self.dataset_path + lbs_file))

        if not lbs.shape[1] == self.orig_res:
            lbs = cv2.resize(lbs, dsize=(self.orig_res, self.orig_res), interpolation=cv2.INTER_NEAREST)
            # lbs_scaled = self.resize(lbs_scaled)
        lbs_scaled = torch.Tensor(lbs).permute(2, 0, 1).float() / 255.0

        if not depth.shape[1] == self.orig_res:
            depth = cv2.resize(depth, dsize=(self.orig_res, self.orig_res), interpolation=cv2.INTER_NEAREST)

        depth = (depth - self.DEPTH_MAX) / self.DEPTH_SCALE / self.DEPTH_SCALE + self.DEPTH_EPS
        # depth = (depth - self.DEPTH_MAX) / self.DEPTH_SCALE + self.real_dist
        depth = np.clip(depth, a_min=0, a_max=1)
        depth_scaled = np.expand_dims(depth, axis=2)
        depth_scaled = torch.Tensor(depth_scaled).permute(2, 0, 1).float()


        tensor_front = self.file2image(color_file)
        color_scaled = (tensor_front - torch.Tensor(self.RGB_MEAN).view(3, 1, 1)) \
                            / torch.Tensor(self.RGB_STD).view(3, 1, 1)
        return depth_scaled, lbs_scaled, color_scaled

    def __init_data__(self, f_name):
        self.input_color = []
        self.input_mask = []
        self.depth_front = []
        self.depth_back = []
        self.lbs_front = []
        self.lbs_back = []
        self.color_front = []
        self.color_back = []

        self.smplx_depth_front = []
        self.smplx_depth_back = []
        self.lbs_full = []
        self.smplx_param = []
        self.smplx_data = []
        self.cam_param = []


        def check_existance(filename):
            if os.path.isfile(self.dataset_path + filename):
                return filename, '.' + filename.split('.')[-1]
            exts = ['.png', '.jpg']
            cur_ext = filename.split('.')[-1]
            for ext in exts:
                tmp_name = filename.replace('.'+cur_ext, ext)
                if os.path.isfile(self.dataset_path + tmp_name):
                    return tmp_name, ext
            return None, None

        with open(f_name) as f:
            for line in f:
                color_input_ = line.strip().split(" ")[0]
                color_input, ext = check_existance(color_input_)
                if color_input is None:
                    print('skipping %s' % color_input_)
                    continue
                sh_num = int(color_input.split("/")[-1].split("_")[2][:-4])
                dataset = color_input.split('/')[1]
                data_type = color_input.split('/')[3]
                data_dir = color_input.split('/')[-2]
                data_name = color_input.split('/')[-1]
                pose_num = data_dir.split('_')[-1]
                angle = color_input.split("/")[-1].split("_")[0]

                # depth and mask are always .png format (to retain 16 bit depth resolution and to avoid jpg artifact)
                if dataset == 'RP' or 'TH2.0':
                    mask_input = color_input.replace('COLOR', 'MASK').replace(data_type, 'OPENGL').replace('%02d'% sh_num+ext, '00.png')
                    color_gt = color_input.replace(data_type, 'OPENGL_OUT').replace('%02d' % sh_num + ext, '00_front.png')
                    depth_gt = color_input.replace('COLOR', 'DEPTH').replace(data_type, 'OPENGL').replace('%02d' % sh_num + ext, '00_front.png')
                    lbs_gt = color_input.replace('COLOR', 'LBS').replace(data_type, 'ENCODED').replace('%02d' % sh_num + ext, '00_front.png')
                    smplx_input = depth_gt.replace('OPENGL', 'OPENGL_SMPLX')


                elif dataset == 'RP_T':
                    mask_input = color_input.replace('COLOR', 'MASK').replace('%02d'% sh_num+ext, '00.png')
                    color_gt = color_input.replace('OPENGL', 'OPENGL_OUT').replace('%02d' % sh_num + ext, '00_front.png')
                    depth_gt = color_input.replace('COLOR', 'DEPTH').replace('%02d' % sh_num + ext, '00_front.png')
                    lbs_gt = color_input.replace('COLOR', 'LBS').replace('OPENGL', 'ENCODED').replace('%02d' % sh_num + ext, '00_front.png')
                    smplx_input = depth_gt.replace('OPENGL', 'OPENGL_SMPLX')

                cam_param = color_input.replace('COLOR', 'PARAM').replace('png', 'json')
                smplx_data = color_input.replace('COLOR', 'SMPLX').replace(data_type, 'DATA').replace(data_name, data_dir + '.npy')

                self.input_color.append(color_input)
                self.input_mask.append(mask_input)
                self.color_front.append(color_gt)
                self.color_back.append(color_gt.replace('front', 'back'))
                self.depth_front.append(depth_gt)
                self.depth_back.append(depth_gt.replace('front', 'back'))
                self.lbs_front.append(lbs_gt)
                self.lbs_back.append(lbs_gt.replace('front', 'back'))
                
                self.smplx_depth_front.append(smplx_input)
                self.smplx_depth_back.append(smplx_input.replace('front', 'back'))

                self.smplx_data.append(smplx_data)
                self.cam_param.append(cam_param)

    def __fetch__(self, idx):
        image_input, mask_input, smplx_input_front, smplx_input_back = self.preprocess_input(self.input_color[idx], self.input_mask[idx], self.smplx_depth_front[idx], self.smplx_depth_back[idx], bg_flag=False)
        
        data_name = self.input_color[idx]
        # target data
        depth_output_front, lbs_output_front, color_output_front = self.preprocess_wrapper(self.depth_front[idx],
                                                                       self.lbs_front[idx],
                                                                       self.color_front[idx])
        depth_output_back, lbs_output_back, color_output_back = self.preprocess_wrapper(self.depth_back[idx], 
                                                                     self.lbs_back[idx],
                                                                     self.color_back[idx])
        color_gt = torch.cat([color_output_front, color_output_back], dim=0) # 6x512x512
        depth_gt = torch.cat([depth_output_front, depth_output_back], dim=0) #2x512x512
        lbs_gt = torch.cat([lbs_output_front, lbs_output_back], dim=0) #6x512x512

        datum = dict()
        datum['input'] = (image_input, mask_input, smplx_input_front, smplx_input_back, data_name)
        datum['label'] = (depth_gt, lbs_gt, color_gt)
        return datum

    def __getitem__(self, idx):
        return self.__fetch__(idx)

    def __len__(self):
        return len(self.depth_front)
