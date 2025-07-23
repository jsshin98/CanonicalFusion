import torch.nn
import os
import cv2
from utils.core.depth2volume import *
from utils.core.im_utils import get_normal

os.environ['PYOPENGL_PLATFORM'] = 'egl'

class Evaluator:
    def __init__(self, device=None):
        super (Evaluator).__init__ ()
        self.device = device

    def get_mse(self):
        return 1

    def get_psnr(self):
        return 1


class HumanEvaluator(Evaluator):
    def __init__(self, device=None):
        self.depth_mse = 0
        self.depth_angle = 0

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

        # do we evaluate depths?
        self.normal_mse = 0
        self.normal_angle = 0

        # unbiased (N+1)
        self.count_volume = 0.000001
        self.count_depth = 0.000001
        self.count_mesh = 0.000001
        self.count_color = 0.000001
        self.count_normal = 0.000001

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
    def get_angle(src, dst, mask, scale=1.0):
        if src.shape[1] > mask.shape[1]:
            mask = mask.repeat(1, src.shape[1], 1, 1)
        criterion_cos = torch.nn.CosineSimilarity(dim=0)
        error = torch.sum(1 - criterion_cos(src[mask>0], dst[mask>0]))

        # mean squared error.
        return torch.mean(torch.sqrt(error))

    @staticmethod
    def tensor2np_color(image):
        if image is not None:
            RGB_MEAN = [0.485, 0.456, 0.406]
            RGB_STD = [0.229, 0.224, 0.225]
            image_np = image[0].permute(1, 2, 0).detach().cpu().numpy()
            image_np = image_np*RGB_STD+RGB_MEAN

            width = image_np.shape[0]
            offset = int(image_np.shape[0] / 4)
            image_up = np.zeros((width, width, 3))
            image_up[:, offset:width - offset, :] = image_np

            # if save_img:
            #     cv2.imwrite(os.path.join(img_path, 'color_%d_%d.png' % (data_idx, angle)),
            #                 (image_up * 255).astype(np.int))
                # cv2.imwrite(os.path.join(img_path, 'color_%d_%d.png' % (data_idx, angle)),
                #             (image_up * 255).astype(np.int))
            return image_up
        else:
            return None

    @staticmethod
    def tensor2np_normal(image, mask=None):
        if image is not None:
            image_np = image[0].permute(1, 2, 0).detach().cpu().numpy()

            if mask is not None:
                condition = mask[:, :] > 0
                image_np[condition==False, 0:3] = [1, 1, 1]
            width = image_np.shape[0]
            offset = int(image_np.shape[0] / 4)
            image_up = np.zeros((width, width, 3))
            image_up[:, offset:width - offset, :] = image_np

            # if save_img:
            #     cv2.imwrite(os.path.join(img_path, 'normal_%d_%d.png' % (data_idx, angle)),
            #                 (image_up * 255).astype(np.int))
            return image_up
        else:
            return None

    @staticmethod
    def tensor2np_depth(image):
        if image is not None:
            normal = get_normal(image[0])
            normal_np = normal.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            image_np = image[0].permute(1, 2, 0).squeeze(2).detach().cpu().numpy()

            width = image_np.shape[0]
            offset = int(image_np.shape[0] / 4)

            image_up = np.zeros((width, width))
            image_up[:, offset:width - offset] = image_np
            image_up[image_up < 0] = 0
            normal_up = np.zeros((width, width, 3))
            normal_up[:, offset:width - offset, :] = normal_np

            # if save_img:
            #     if width == 256:
            #         cv2.imwrite(os.path.join(img_path, 'depth_%d_%d.png' % (data_idx, angle)),
            #                     (image_up * 128 * 512).astype(np.uint16))
            #         cv2.imwrite(os.path.join(normal_path, 'normal_%d_%d.png' % (data_idx, angle)),
            #                     (normal_up * 255).astype(np.int))
            #     else:
            #         cv2.imwrite(os.path.join(img_path, 'depth_%d_%d.png' % (data_idx, angle)),
            #                     (image_up * 64 * 512).astype(np.uint16))
            #         cv2.imwrite(os.path.join(normal_path, 'normal_%d_%d.png' % (data_idx, angle)),
            #                     (normal_up * 255).astype(np.int))
            return image_up
        else:
            return None

    def visualize_color(self, pred_var, target_var):
        dir = ['front', 'back', 'left', 'right']
        for idx in range(len(pred_var)):
            if pred_var[idx] is not None:
                pred_image = self.tensor2np_color(pred_var[idx])
            else:
                tgr_image = self.tensor2np_color(target_var[idx])
            if dir[idx] == 'front':
                return pred_image

    def visualize_normal(self, pred_var, target_var=None, save_img=False,
                         pred_path=None, tgt_path=None, data_idx=None, angle=None):

        dir = ['front', 'back', 'left', 'right']
        for idx in range(len(pred_var)):
            if pred_var[idx] is not None:
                self.tensor2np_normal(pred_var[idx])
            else:
                self.tensor2np_normal(target_var[idx])

    def visualize_depth(self, pred_var, target_var):
        dir = ['front', 'back', 'left', 'right']
        for idx in range(len(pred_var)):
            if pred_var[idx] is not None:
                self.tensor2np_depth(pred_var[idx])

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
        # print("error")