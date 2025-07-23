import torch.nn
import os
import trimesh
import trimesh.sample
import trimesh.proximity
import numpy as np
from utils.core.depth2volume import *
from utils.core.im_utils import get_normal
import cv2
import skimage
from scipy import spatial
from numpy import dot
from numpy.linalg import norm
import math

from PIL import Image
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# define error metrics.
class Evaluator:
    def __init__(self, device=None):
        super (Evaluator).__init__ ()
        self.device = device

    def get_mse(self):
        return 1

    def get_psnr(self):
        return 1

def euler_to_rot_mat(r_x, r_y, r_z):
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

class HumanEvaluator(Evaluator):
    def __init__(self, device=None):
        # do we evaluate depths?
        # self.depth_mse_front = 0
        # self.depth_mse_back = 0
        # self.depth_mse_left = 0
        # self.depth_mse_right = 0
        self.depth_mse = 0
        self.depth_angle = 0

        # do we evaluate colors?
        self.color_psnr_front = 0
        self.color_psnr_back = 0
        self.color_psnr_left = 0
        self.color_psnr_right = 0

        # we primarily evaluate 3D models
        self.mesh_iou = 0
        self.mesh_chamfer = 0
        self.mesh_dist = 0
        self.mesh_chamfer_pifu = 0
        self.mesh_dist_pifu = 0
        self.mesh_chamfer_pifuhd = 0
        self.mesh_dist_pifuhd = 0

        # volume losses
        self.volume_iou = 0
        self.volume_prec = 0
        self.volume_recall = 0
        # self.volume_iou_pifu = 0
        # self.volume_prec_pifu = 0
        # self.volume_recall_pifu = 0
        # self.volume_iou_pifuhd = 0
        # self.volume_prec_pifuhd = 0
        # self.volume_recall_pifuhd = 0

        # do we evaluate depths?
        self.normal_mse = 0
        # self.normal_mse_front = 0
        # self.normal_mse_back = 0
        # self.normal_mse_left = 0
        # self.normal_mse_right = 0
        self.normal_angle = 0
        # self.normal_angle_front = 0
        # self.normal_angle_back = 0
        # self.normal_angle_left = 0
        # self.normal_angle_right = 0

        # unbiased (N+1)
        self.count_volume = 0.000001
        self.count_depth = 0.000001
        self.count_mesh = 0.000001
        self.count_color = 0.000001
        self.count_normal = 0.000001


    # assuming they are depth maps
    def initialize(self, src, dst, target='front'):
        dist = []
        return dist

    @staticmethod
    def get_volume_loss(pred, target):
        thresh = 0.0 # 0.5 or 0

        vol_pred = pred <= thresh
        vol_gt = target <= thresh

        union = vol_pred | vol_gt
        inter = vol_pred & vol_gt

        true_pos = inter.sum()
        union = union.sum()
        vol_pred = vol_pred.sum ()
        vol_gt = vol_gt.sum ()

        IOU = true_pos / union
        prec = true_pos / vol_pred
        recall = true_pos / vol_gt

        return IOU, prec, recall

    @staticmethod
    def get_mse(src, dst, mask, scale=1.0):
        if src.shape[1] > mask.shape[1]:
            mask = mask.repeat(1, src.shape[1], 1, 1)
        # mask_np = mask.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # target_var_np = dst[mask>0].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # pred_var_np = src[mask>0].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # cv2.imshow('mask', mask_np)
        # cv2.imshow('target', target_var_np)
        # cv2.imshow('pred', pred_var_np)
        # cv2.waitKey(0)
        error = torch.sum((src[mask>0]*scale - dst[mask>0]*scale) ** 2)

        # mean squared error.
        return torch.mean(torch.sqrt(error))

    @staticmethod
    def get_angle(src, dst, mask, scale=1.0):
        if src.shape[1] > mask.shape[1]:
            mask = mask.repeat(1, src.shape[1], 1, 1)
        criterion_cos = torch.nn.CosineSimilarity(dim=0)
        error = torch.sum(1 - criterion_cos(src[mask>0], dst[mask>0]))
        # cos_sim = dot(src, dst) / (norm(src) * norm(dst))
        # error = error / (torch.sum(mask, dim=[1, 2, 3]))

        # mean squared error.
        return torch.mean(torch.sqrt(error))

    @staticmethod
    def tensor2np_color(image, scale=2.0, save_img=False, img_path=None, cnt=None, dir=None):
        if image is not None:
            RGB_MEAN = [0.485, 0.456, 0.406]
            RGB_STD = [0.229, 0.224, 0.225]
            image_np = image[0].permute(1, 2, 0).detach().cpu().numpy()
            image_np = image_np*RGB_STD+RGB_MEAN
            # image_up = image_np
            # image_np = cv2.resize (image_np, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            width = image_np.shape[0]
            offset = int(image_np.shape[0] / 4)
            image_up = np.zeros((width, width, 3))
            image_up[:, offset:width - offset, :] = image_np

            if save_img:
                cv2.imwrite(os.path.join(img_path, 'color_%d_%s.png' % (cnt, dir)), (image_up * 255).astype(np.int))
            return image_up
        else:
            return None

    @staticmethod
    def tensor2np_normal(image, mask=None, scale=2.0, save_img=False, img_path=None, cnt=None, dir=None):
        if image is not None:
            image_np = image[0].permute(1, 2, 0).detach().cpu().numpy()
            # image_np = cv2.resize (image_np, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            if mask is not None:
                condition = mask[:, :] > 0
                image_np[condition==False, 0:3] = [1, 1, 1]
            width = image_np.shape[0]
            offset = int(image_np.shape[0] / 4)
            image_up = np.zeros((width, width, 3))
            image_up[:, offset:width - offset, :] = image_np

            if save_img:
                cv2.imwrite(os.path.join(img_path, 'normal_%d_%s.png' % (cnt, dir)), (image_up * 255).astype(np.int))
            return image_up
        else:
            return None

    @staticmethod
    def tensor2np_depth(image, scale=2.0, save_img=False, img_path=None, normal_path=None, cnt=None, dir=None):
        if image is not None:
            normal = get_normal(image[0])
            normal_np = normal.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            image_np = image[0].permute(1, 2, 0).squeeze(2).detach().cpu().numpy()
            # image_np = cv2.resize (image_np, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            width = image_np.shape[0]
            offset = int(image_np.shape[0] / 4)

            image_up = np.zeros((width, width))
            image_up[:, offset:width - offset] = image_np
            image_up[image_up < 0] = 0
            normal_up = np.zeros((width, width, 3))
            normal_up[:, offset:width - offset, :] = normal_np
            # normal_up[normal_up < 0] = 0

            if save_img:
                if width == 256:
                    cv2.imwrite(os.path.join(img_path, 'depth_%d_%s.png' % (cnt, dir)),
                                (image_up * 128 * 512).astype(np.uint16))
                    cv2.imwrite(os.path.join(normal_path, 'normal_%d_%s.png' % (cnt, dir)),
                                (normal_up * 255).astype(np.int))
                else:
                    cv2.imwrite(os.path.join(img_path, 'depth_%d_%s.png' % (cnt, dir)),
                                (image_up * 64 * 512).astype(np.uint16))
                    cv2.imwrite(os.path.join(normal_path, 'normal_%d_%s.png' % (cnt, dir)),
                                (normal_up * 255).astype(np.int))
            return image_up
        else:
            return None

    def visualize_color(self, pred_var, target_var, scale=2.0, save_img=False, pred_path=None, tgt_path=None, cnt=None):
        for idx in range(len(target_var)):
            dir = ['front', 'back', 'left', 'right']

            if pred_var[idx] is not None and target_var[idx] is not None:
                #cv2.imshow('%s color(PRED)' % dir[idx],
                self.tensor2np_color(pred_var[idx], save_img=save_img, img_path=pred_path, cnt=cnt, dir=dir[idx])#)
                # cv2.imshow('%s color(GT)' % dir[idx],
                self.tensor2np_color(target_var[idx], scale=scale, save_img=save_img, img_path=tgt_path, cnt=cnt, dir=dir[idx])#)
            else:
                # cv2.imshow('%s color(GT)' % dir[idx],
                self.tensor2np_color(target_var[idx], scale=scale, save_img=save_img, img_path=tgt_path, cnt=cnt,
                                     dir=dir[idx])  # )

        # cv2.waitKey(0)

    def visualize_normal(self, pred_var, target_var=None, scale=2.0, save_img=False, pred_path=None, tgt_path=None, cnt=None):
        for idx in range(len(target_var)):
            dir = ['front', 'back', 'left', 'right']

            if pred_var[idx] is not None and target_var[idx] is not None:
                # cv2.imshow('%s normal(PRED)' % dir[idx],
                self.tensor2np_normal(pred_var[idx], save_img=save_img, img_path=pred_path, cnt=cnt, dir=dir[idx])#)
                # cv2.imshow('%s normal(GT)' % dir[idx],
                self.tensor2np_normal(target_var[idx], scale=scale, save_img=save_img, img_path=tgt_path, cnt=cnt, dir=dir[idx])#)
            else:
                # cv2.imshow('%s normal(GT)' % dir[idx],
                self.tensor2np_normal(target_var[idx], scale=scale, save_img=save_img, img_path=tgt_path, cnt=cnt,
                                      dir=dir[idx])  # )
        # cv2.waitKey(0)

    def visualize_depth(self, pred_var, target_var=None, scale=2.0, save_img=False, pred_path=None, pred_depth2normal_path=None,
                        tgt_depth2normal_path=None, tgt_path=None, cnt=None):
        for idx in range(len(target_var)):
            dir = ['front', 'back', 'left', 'right']

            if pred_var[idx] is not None and target_var[idx] is not None:
                #cv2.imshow('%s depth(PRED)' % dir[idx],
                self.tensor2np_depth(pred_var[idx], save_img=save_img, img_path=pred_path, normal_path=pred_depth2normal_path, cnt=cnt, dir=dir[idx])#)
                # cv2.imshow('%s depth(GT)' % dir[idx],
                self.tensor2np_depth(target_var[idx], scale=scale, save_img=save_img, img_path=tgt_path, normal_path=tgt_depth2normal_path, cnt=cnt, dir=dir[idx])#)
        # cv2.waitKey(0)

    def save_mesh(self, src, dst,  src_color=None, dst_color=None,
                  pred_path=None, tgt_path=None, cnt=None, mode='2view'):
        if mode == '2view' and src_color is not None:
            src_front = self.tensor2np_color(src_color[0], save_img=False, dir='front')
            src_back = self.tensor2np_color(src_color[1], save_img=False, dir='back')
            dst_front = self.tensor2np_color(dst_color[0], save_img=False, dir='front')
            dst_back = self.tensor2np_color(dst_color[1], save_img=False, dir='back')

        elif mode == '4view' and src_color is not None:
            src_front = self.tensor2np_color(src_color[0], save_img=False, dir='front')
            src_back = self.tensor2np_color(src_color[1], save_img=False, dir='back')
            src_left = self.tensor2np_color(src_color[2], save_img=False, dir='left')
            src_right = self.tensor2np_color(src_color[3], save_img=False, dir='right')
            dst_front = self.tensor2np_color(dst_color[0], save_img=False, dir='front')
            dst_back = self.tensor2np_color(dst_color[1], save_img=False, dir='back')
            dst_left = self.tensor2np_color(dst_color[2], save_img=False, dir='left')
            dst_right = self.tensor2np_color(dst_color[3], save_img=False, dir='right')

        src_mesh, src_model_color = colorize_model(src, src_back, src_front)
        dst_mesh, dst_model_color = colorize_model(dst, dst_back, dst_front)

        # dst_mesh.vertices = (dst_mesh.vertices - dst_mesh.centroid)
        # dst_mesh_np = np.array(dst_mesh.vertices)
        # val = np.maximum(np.max(dst_mesh_np), np.abs(np.min(dst_mesh_np)))
        # dst_mesh.vertices /= val
        # dst_mesh.vertices *= 0.5
        dst_mesh.vertices -= dst_mesh.bounding_box.centroid
        dst_mesh.vertices *= 2 / np.max(dst_mesh.bounding_box.extents)
        dst_mesh.export('%s/result_%d.ply' % (tgt_path, cnt))

        # src_mesh.vertices = (src_mesh.vertices - src_mesh.centroid)
        # src_mesh.vertices /= val
        # src_mesh.vertices *= 0.5
        src_mesh.vertices -= src_mesh.bounding_box.centroid
        src_mesh.vertices *= 2 / np.max(src_mesh.bounding_box.extents)
        src_mesh.export('%s/result_%d.obj' % (pred_path, cnt))

        # src_front_normal = get_normal(src_front_normal)
        # src_back_normal = get_normal(src_back_normal)
        # dst_front_normal = get_normal(dst_front_normal)
        # dst_back_normal = get_normal(dst_back_normal)
        # error_front = abs(src_front_normal-dst_front_normal)
        # error_back = abs(src_back_normal-dst_back_normal)
        # cv2.imshow('er_front', error_front)
        # cv2.imshow('er_back', error_back)
        # cv2.waitKey(0)
        # error_mesh, error_model_color = colorize_model(src, error_front, error_back)
        # error_mesh.show()
        # error_mesh.export('%s/result_%d.obj' % (error_mesh_path, cnt))
        # dst_color_mesh.show()
        # dst_mesh = volume2mesh(dst_volume, visualize=False)
        # dst_mesh.vertices = (dst_mesh.vertices - dst_mesh.centroid)
        # dst_vertices = np.array(dst_mesh.vertices)
        # val = np.maximum(np.max(dst_vertices), np.abs(np.min(dst_vertices)))
        # dst_mesh.vertices /= val
        # dst_mesh.vertices *= 0.50

    def evaluate_volume(self, src, dst, mode='2view'):
        src_volume = src['volume_' + mode].squeeze(0).detach().cpu().numpy()
        dst_volume = dst['volume_' + mode].squeeze(0).detach().cpu().numpy()

        IOU, prec, recall = self.get_volume_loss(src_volume, dst_volume)

        self.volume_iou += IOU
        self.volume_prec += prec
        self.volume_recall += recall

    def evaluate_mesh(self, src, dst, src_color, dst_color, pred_path, tgt_path, eval_metric=None, save_img=False,
                      cnt=0, num_samples=10000, mode='2view'):
        src_volume = src.squeeze(0).detach().cpu().numpy()
        dst_volume = dst.squeeze(0).detach().cpu().numpy()

        if save_img:
            self.save_mesh(src_volume, dst_volume, src_color, dst_color, pred_path, tgt_path, cnt=cnt, mode=mode)

        src_mesh = volume2mesh(src_volume, visualize=False)
        dst_mesh = volume2mesh(dst_volume, visualize=False)

        src_mesh.vertices -= src_mesh.bounding_box.centroid
        src_mesh.vertices *= 2 / np.max(src_mesh.bounding_box.extents)
        dst_mesh.vertices -= dst_mesh.bounding_box.centroid
        dst_mesh.vertices *= 2 / np.max(dst_mesh.bounding_box.extents)

        # dst_mesh.vertices = (dst_mesh.vertices - dst_mesh.centroid)
        # dst_mesh_np = np.array(dst_mesh.vertices)
        # val = np.maximum(np.max(dst_mesh_np), np.abs(np.min(dst_mesh_np)))
        # dst_mesh.vertices /= val
        # dst_mesh.vertices *= 0.5
        #
        # src_mesh.vertices = (src_mesh.vertices - src_mesh.centroid)
        # src_mesh.vertices /= val
        # src_mesh.vertices *= 0.5

        src_surf_pts, _ = trimesh.sample.sample_surface(src_mesh, num_samples)
        tgt_surf_pts, _ = trimesh.sample.sample_surface(dst_mesh, num_samples)

        _, src_tgt_dist, _ = trimesh.proximity.closest_point(dst_mesh, src_surf_pts)
        _, tgt_src_dist, _ = trimesh.proximity.closest_point(src_mesh, tgt_surf_pts)

        src_tgt_dist[np.isnan(src_tgt_dist)] = 0
        tgt_src_dist[np.isnan(tgt_src_dist)] = 0

        src_tgt_dist = src_tgt_dist.mean()
        tgt_src_dist = tgt_src_dist.mean()

        if 'pifu' in eval_metric:
            pifu_mesh = trimesh.load_mesh('/home/keti/results_pifu_noshade/result_image_%05d.obj'%cnt)
            pifuhd_mesh = trimesh.load_mesh('/home/keti/results_pifu_noshade/result_image_%05d.obj' % cnt)
            pifu_surf_pts, _ = trimesh.sample.sample_surface(pifu_mesh, num_samples)
            pifuhd_surf_pts, _ = trimesh.sample.sample_surface(pifuhd_mesh, num_samples)
            _, pifu_tgt_dist, _ = trimesh.proximity.closest_point(dst_mesh, pifu_surf_pts)
            _, tgt_pifu_dist, _ = trimesh.proximity.closest_point(pifu_mesh, tgt_surf_pts)
            _, pifuhd_tgt_dist, _ = trimesh.proximity.closest_point(dst_mesh, pifuhd_surf_pts)
            _, tgt_pifuhd_dist, _ = trimesh.proximity.closest_point(pifuhd_mesh, tgt_surf_pts)
            pifu_tgt_dist[np.isnan(pifu_tgt_dist)] = 0
            tgt_pifu_dist[np.isnan(tgt_pifu_dist)] = 0
            pifuhd_tgt_dist[np.isnan(pifuhd_tgt_dist)] = 0
            tgt_pifuhd_dist[np.isnan(tgt_pifuhd_dist)] = 0

            pifu_tgt_dist = pifu_tgt_dist.mean()
            tgt_pifu_dist = tgt_pifu_dist.mean()
            pifuhd_tgt_dist = pifuhd_tgt_dist.mean()
            tgt_pifuhd_dist = tgt_pifuhd_dist.mean()
            chamfer_dist_pifu = (pifu_tgt_dist + tgt_pifu_dist) / 2
            chamfer_dist_pifuhd = (pifuhd_tgt_dist + tgt_pifuhd_dist) / 2
            self.mesh_chamfer_pifu += chamfer_dist_pifu
            self.mesh_chamfer_pifuhd += chamfer_dist_pifuhd
            self.mesh_dist_pifu += pifu_tgt_dist
            self.mesh_dist_pifuhd += pifuhd_tgt_dist

        chamfer_dist = (src_tgt_dist + tgt_src_dist) / 2

        self.mesh_chamfer += chamfer_dist
        self.mesh_dist += src_tgt_dist
        IoU, prec, recall = \
            self.get_volume_loss(src_volume, dst_volume)
        self.volume_iou += IoU
        self.volume_prec += prec
        self.volume_recall += recall
        self.count_volume += 1
        self.count_mesh += 1

    def evaluate_normal(self, pred_var, target_var):
        # pre_processing.
        normal_mse = 0
        normal_angle = 0
        for idx in range(len(pred_var)):
            if pred_var[idx] is not None and target_var[idx] is not None:
                mask = torch.zeros(target_var[idx].shape)
                mask[target_var[0] > 0] = 255
                # mask_np = mask.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                # target_var_np = target_var[0].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                # pred_var_np = pred_var[0].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                # cv2.imshow('mask', mask_np)
                # cv2.imshow('target', target_var_np)
                # cv2.imshow('pred', pred_var_np)
                # cv2.waitKey(0)
                mask = mask.cuda()
                normal_mse += self.get_mse(pred_var[idx],
                                                target_var[idx],
                                                mask,
                                                scale=1)
                normal_angle += self.get_angle(pred_var[idx],
                                                    target_var[idx],
                                                    mask,
                                                    scale=1)
        self.normal_mse += (normal_mse/len(pred_var))
        self.normal_angle += (normal_angle/len(pred_var))
        self.count_normal += pred_var[0].shape[0]

    def evaluate_depth(self, pred_var, target_var):
        # pre_processing.
        depth_mse = 0
        depth_angle = 0
        for idx in range(len(pred_var)):
            if pred_var[idx] is not None and target_var[idx] is not None:
                mask = torch.zeros(target_var[idx].shape)
                mask[target_var[idx] > 0] = 255
                mask = mask.cuda()
                depth_mse += self.get_mse(pred_var[idx],
                                               target_var[idx],
                                               mask,
                                               scale=1)
                depth_angle += self.get_angle(get_normal(pred_var[idx]),
                                                   get_normal(target_var[idx]),
                                                   mask,
                                                   scale=1)
        self.depth_mse += (depth_mse/len(pred_var))
        self.depth_angle += (depth_angle/len(pred_var))
        self.count_depth += pred_var[0].shape[0]

    def print_results(self):
        if self.count_normal > 1:
            print('Normal(mse): MSE({e1:.5f})'.
                  format(e1=self.normal_mse/self.count_normal))
            print('Normal(angle): MSE({e1:.5f})'.
                  format(e1=self.normal_angle/self.count_normal))

        if self.count_depth > 1:
            print('Depth(mse): MSE({e1:.5f})'.
                  format(e1=self.depth_mse/self.count_depth))
            print('Depth(angle): MSE({e1:.5f})'.
                  format(e1=self.depth_angle/self.count_depth))

        if self.count_volume > 1:
            print('Completeness(%): IoU({e1:.5f}), Prec({e2:.5f}), Recall({e3:.5f})'.
                  format(e1=self.volume_iou/self.count_volume, e2=self.volume_prec/self.count_volume,
                         e3=self.volume_recall/self.count_volume))

        if self.count_mesh > 1:
            print('Mesh(mse): Dist({e1:.5f}), Chamfer({e2:.5f})'.
                  format(e1=self.mesh_dist/self.count_mesh, e2=self.mesh_chamfer/self.count_mesh))
            print('PIFU_Mesh(mse): Dist({e1:.5f}), Chamfer({e2:.5f})'.
                  format(e1=self.mesh_dist_pifu / self.count_mesh, e2=self.mesh_chamfer_pifu / self.count_mesh))
            print('PIFUHD_Mesh(mse): Dist({e1:.5f}), Chamfer({e2:.5f})'.
                  format(e1=self.mesh_dist_pifuhd / self.count_mesh, e2=self.mesh_chamfer_pifuhd / self.count_mesh))

    def save_results(self, path2dir, epoch):

        filename = 'eval_results_epoch{0}.txt'.format(epoch)
        exp_results = '[experimental results]\n' + \
                  '- depth_mse:  {0:3.3f} \n'.format (self.depth_mse / self.count_depth) + \
                  '- depth_angle:   {0:3.3f} \n'.format (self.depth_angle / self.count_depth) + \
                  '- normal_mse: {0:3.3f} \n'.format (self.normal_mse / self.count_normal) + \
                  '- normal_angle: {0:3.3f} \n'.format(self.normal_angle / self.count_normal) + \
                  '- volume(iou):   {0:3.3f} \n'.format (self.volume_iou / self.count_volume) + \
                  '- volume(prec):  {0:3.3f} \n'.format (self.volume_prec / self.count_volume) + \
                  '- volume(recall):{0:3.3f} \n'.format (self.volume_recall / self.count_volume) + \
                  '- mesh(chamfer): {0:3.3f} \n'.format (self.mesh_chamfer / self.count_mesh) + \
                  '- mesh(dist):    {0:3.3f} \n'.format (self.mesh_dist / self.count_mesh) + \
                  '- pifu_mesh(chamfer): {0:3.3f} \n'.format(self.mesh_chamfer_pifu / self.count_mesh) + \
                  '- pifu_mesh(dist):    {0:3.3f} \n'.format(self.mesh_dist_pifu / self.count_mesh) + \
                  '- pifuhd_mesh(chamfer): {0:3.3f} \n'.format(self.mesh_chamfer_pifuhd / self.count_mesh) + \
                  '- pifuhd_mesh(dist):    {0:3.3f} \n'.format(self.mesh_dist_pifuhd / self.count_mesh)

        with open (os.path.join(path2dir, filename), "w") as f:
            f.write (exp_results)
            f.close ()
        print("error")