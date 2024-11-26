import argparse
import os
import json


class Options():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Datasets related
        g_data = parser.add_argument_group('Data')
        g_data.add_argument ('--dataset', type=str, default='',
                             help='somewhere in your PC')
        g_data.add_argument('--data_root', type=str, default='',
                            help='somewhere in your PC')
        g_data.add_argument ('--train_list', type=str, default='train_split', help='somewhere in your PC')
        g_data.add_argument ('--val_list', type=str, default='val_split', help='somewhere in your PC')
        g_data.add_argument ('--eval_list', type=str, default='val_split', help='somewhere in your PC')
        g_data.add_argument ('--bg_list', type=str, default='val_split', help='somewhere in your PC')

        # Experiment related
        g_exp = parser.add_argument_group('Experiment')
        g_exp.add_argument('--exp_name', type=str, default='example',
                           help='name of the experiment. It decides where to store samples and models')
        g_data.add_argument ('--loss_conf', type=str, default='val_split', help='somewhere in your PC')
        g_data.add_argument ('--loader_conf', type=str, default='val_split', help='somewhere in your PC')
        g_data.add_argument ('--disc_conf', type=str, default='val_split', help='somewhere in your PC')
        g_data.add_argument ('--use_GAN', type=bool, default=False, help='somewhere in your PC')
        g_data.add_argument ('--center_crop', type=bool, default=False, help='somewhere in your PC')
        g_data.add_argument ('--model_name', type=str, default='BaseModule')
        g_data.add_argument ('--model_C_name', type=str, default='ColorModule')

        # Training related
        g_train = parser.add_argument_group('Training')
        g_train.add_argument('--workers', type=int, default=8, help='workers')
        g_train.add_argument('--gpu_id', type=int, default=0, help='gpu id for cuda')
        g_train.add_argument('--use_multi_gpus', type=bool, default=True, help='utilize multi-gpus')
        g_train.add_argument('--gpu_ids', type=str, default='0,1,2,3,4,5,6,7', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode') # ,1,2,3,4,5,6,7

        g_train.add_argument('--num_threads', default=4, type=int, help='# sthreads for loading data')
        g_train.add_argument('--pin_memory', type=bool, default=True, help='utilize multi-gpus')

        g_train.add_argument('--batch_size', type=int, default=1, help='input batch size')
        g_train.add_argument ('--batch_size_val', type=int, default=1, help='input batch size')
        g_train.add_argument('--learning_rate', type=float, default=1e-3, help='adam learning rate')

        g_train.add_argument ('--res', type=int, default=512, help='num epoch to train')
        g_train.add_argument ('--voxel_size', type=int, default=256, help='num epoch to train')
        g_train.add_argument ('--print_freq', type=int, default=30, help='num epoch to train')
        g_train.add_argument ('--eval_freq', type=int, default=1, help='num epoch to train')
        g_train.add_argument ('--num_epoch', type=int, default=100, help='num epoch to train')
        g_train.add_argument ('--log_freq', type=int, default=30, help='num epoch to train')

        g_train.add_argument('--start_epoch', type=int, default=1, help='# to the first epoch')
        g_train.add_argument('--continue_train', type=bool, default=False, help='utilize multi-gpus')
        g_train.add_argument('--train_nl_color', type=bool, default=False, help='whether train no light color model or not')
        g_train.add_argument('--phase1', type=bool, default=False, help='whether train on phase2 or phase1')
        g_train.add_argument('--phase2_epoch', type=int, default=100, help='phase2 epoch')
        g_train.add_argument('--path2pretrained', type=str, default='', help='path to save checkpoints')

        # path
        parser.add_argument('--checkpoints_path', type=str, default='./checkpoints', help='path to save checkpoints')
        parser.add_argument('--local-rank', type=int, default=-1, metavar='N', help='Local process rank.')  # you need this argument in your scripts for DDP to work
        parser.add_argument('--use_ddp', type=bool, default=True, help='utilize multi-gpus')
        parser.add_argument('--use_dp', type=bool, default=False, help='utilize multi-gpus')
        parser.add_argument('--is_master', type=bool, default=True, help='indicate whether the currnent is master')

        # special tasks
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic optsions
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        return opt

    # def initialize_from_json(self, parser, filename):
    #     data = json.load(filename)
    #     parser.parse_args = []
    #
    # def save_as_json(self, file_name):
    #     data = []
    #     json.dump(data)


# generate training related files when this function is called as main.
if __name__ == '__main__':
    # opt = BaseOptions()
    # opt.initialize()
    filename = []
    # opt.save_as_json(filename)
