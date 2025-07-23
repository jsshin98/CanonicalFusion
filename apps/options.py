import os
import argparse


class Configurator:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # paths
        self.parser.add_argument('--dataset', type=str, default='RP')
        self.parser.add_argument('--path2sh', type=str, default='../resource/sh')
        self.parser.add_argument('--path2image', type=str, default='../resource/sample_images')
        self.parser.add_argument('--path2data', type=str, default=None)
        self.parser.add_argument('--path2obj', type=str, default=None)
        self.parser.add_argument('--path2results', type=str, default='./')
        self.parser.add_argument('--path2smpl', type=str, default=None)
        self.parser.add_argument('--path2samples', type=str, default=None)
        self.parser.add_argument('--path2uv', type=str, default=None)
        self.parser.add_argument('--path2semantic', type=str,
                                 default='../resource/smpl_models/body_segmentation/smplx/smplx_vert_segmentation.json')
        self.parser.add_argument('--path2pretrained', type=str, default='../resource/pretrained_models')

        # smpl parameters
        self.parser.add_argument('--smpl_gender', type=str, default='neutral')
        self.parser.add_argument('--smpl_type', type=str, default='smplx')
        self.parser.add_argument('--smpl_path', type=str, default='../resource/smpl_models')
        # self.parser.add_argument('--smpl_pose', type=str, default='da-pose',
        #                          help='da-pose or pose')
        self.parser.add_argument('--smpl_pca', type=bool, default=False, help='use pca of not')
        self.parser.add_argument('--smpl_num_pca_comp', type=int, default=12, help='')
        self.parser.add_argument('--smpl_num_beta', type=int, default=10, help='')
        self.parser.add_argument('--smpl_flat_hand', action='store_true')
        self.parser.add_argument('--no-smpl_flat_hand', dest='smpl_flat_hand', action='store_false')
        self.parser.set_defaults(smpl_flat_hand=False)
        self.parser.add_argument('--age', type=str, default='adult', help='adult or kid')

        # common
        self.parser.add_argument('--device', type=str, default='0')
        self.parser.add_argument('--pose_detector', type=str, default='openpose')
        self.parser.add_argument('--use_ddp', action='store_true')
        self.parser.add_argument('--no-use_ddp', dest='use_ddp', action='store_false')
        self.parser.set_defaults(use_ddp=False)
        self.parser.add_argument('--name', type=str, default='20230719')

        # gender prediction
        self.parser.add_argument('--predict_gender', action='store_true')
        self.parser.add_argument('--no-predict_gender', dest='predict_gender', action='store_false')

        # foreground extraction
        self.parser.add_argument('--seg_method', type=str, default='people',
                                 help='people: people segmentation, remove_bg')
        self.parser.add_argument('--use_free_version', action='store_true')
        self.parser.add_argument('--no-use_free_version', dest='use_free_version', action='store_false')
        self.parser.set_defaults(use_free_version=True)
        self.parser.add_argument('--extract_foreground', action='store_true')
        self.parser.add_argument('--no-extract_foreground', dest='extract_foreground', action='store_false')
        self.parser.set_defaults(extract_foreground=False)

        # recon
        self.parser.add_argument('--esr_gan_path', type=str,
                                 default='../resource/pretrained_models/Real-ESRGAN/weights/RealESRGAN_x2plus.pth')
        self.parser.add_argument('--recon_ckpt', type=str, default='../resource/pretrained_models/CVPRW2022')
        self.parser.add_argument('--model_name', type=str, default='DeepHumanNet',
                                 help='DeepHumanNet, FamozNet')
        self.parser.add_argument('--model_config', type=str, default='NC2D')
        self.parser.add_argument('--recon_res', type=int, default=1024,
                                 help='The actual resolution of input to the network')
        self.parser.add_argument('--replace_hands', action='store_true')
        self.parser.add_argument('--no-replace_hands', dest='replace_hands', action='store_false')
        self.parser.set_defaults(replace_hands=True)
        self.parser.add_argument('--use_lbs', action='store_true')
        self.parser.add_argument('--no-use_lbs', dest='use_lbs', action='store_false')
        self.parser.set_defaults(use_lbs=False)
        self.parser.add_argument('--use_conf', action='store_true')
        self.parser.add_argument('--no-use_conf', dest='use_conf', action='store_false')
        self.parser.set_defaults(use_conf=False)
        self.parser.add_argument('--refine_colors', action='store_true')
        self.parser.add_argument('--no-refine_colors', dest='refine_colors', action='store_false')
        self.parser.set_defaults(refine_colors=True)

        # train.
        self.parser.add_argument('--num_threads', type=int, default=0, help='# sthreads for loading data')
        self.parser.add_argument('--pin_memory', action='store_true', help='pin_memory')
        self.parser.add_argument('--seed_worker', type=str, default='checkpoint')
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--z_size', type=float, default=200.0, help='z normalization factor')
        self.parser.add_argument('--netconfig', type=str, default='keti_cvpr', help='configured date')
        self.parser.add_argument('--no_residual', action='store_true', help='no skip connection in mlp')
        self.parser.add_argument('--loadSize', type=int, default=1024, help='load size of input image')
        self.parser.add_argument('--resume_epoch', type=int, default=-1, help='epoch resuming the training')
        self.parser.add_argument('--learning_rate', type=float, default=1e-3, help='adam learning rate')
        self.parser.add_argument('--num_epoch', type=int, default=100, help='num epoch to train')
        self.parser.add_argument('--progressive', type=int, default=0)
        self.parser.add_argument('--freq_plot', type=int, default=10, help='freqency of the error plot')
        self.parser.add_argument('--freq_save', type=int, default=50, help='freqency of the save_checkpoints')
        self.parser.add_argument('--freq_save_ply', type=int, default=10, help='freqency of the save ply')
        self.parser.add_argument('--no_gen_mesh', action='store_true')
        self.parser.add_argument('--no_num_eval', action='store_true')
        self.parser.add_argument('--num_views', type=int, default=1,
                                 help='How many views to use for multiview network.')

        # hyper parameters
        self.parser.add_argument('--num_samples', type=int, default=10000, help='number of samples to draw')
        self.parser.add_argument('--num_samples_all', type=int, default=50000, help='number of sampled points')

        # path
        self.parser.add_argument('--netG_ckpt', type=str, default='../resource/pretrained_models/ImplicitNet')
        self.parser.add_argument('--netC_ckpt', type=str, default='../resource/pretrained_models/ImplicitNet')
        # below paths will be removed
        self.parser.add_argument('--checkpoints_path', type=str, default='../resource/pretrained_models',
                                 help='path to save checkpoints')
        self.parser.add_argument('--load_netG_checkpoint_path', type=str, default=None, help='path to save checkpoints')
        self.parser.add_argument('--load_netC_checkpoint_path', type=str, default=None, help='path to save checkpoints')
        self.parser.add_argument('--results_path', type=str, default='../results', help='path to save results ply')
        self.parser.add_argument('--load_checkpoint_path', type=str, help='path to save results ply')

        # General
        self.parser.add_argument('--loss_init', action='append')
        self.parser.add_argument('--loss_fine', action='append')
        self.parser.add_argument('--norm', type=str, default='group',
                                 help='instance normalization or batch normalization or group normalization')
        self.parser.add_argument('--norm_color', type=str, default='instance',
                                 help='instance normalization or batch normalization or group normalization')

        # hg filter specify
        self.parser.add_argument('--num_stack', type=int, default=4, help='# of hourglass')
        self.parser.add_argument('--num_hourglass', type=int, default=2, help='# of stacked layer of hourglass')
        self.parser.add_argument('--skip_hourglass', action='store_true', help='skip connection in hourglass')
        self.parser.add_argument('--hg_down', type=str, default='ave_pool', help='ave pool || conv64 || conv128')
        self.parser.add_argument('--hourglass_dim', type=int, default='256', help='256 | 512')

    def parse(self):
        return self.parser.parse_args()
