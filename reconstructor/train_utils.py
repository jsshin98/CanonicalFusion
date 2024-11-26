import models
import glob
import os
import torch
import torchvision
import cv2
import shutil
import json
import datetime
import collections
import numpy as np
import platform
import torch.distributed as dist
from PIL import Image, ImageFile
# from utils.core.volume2depth import *
# from utils.core.depth2volume import *

ImageFile.LOAD_TRUNCATED_IMAGES = True



# ddp related functions.
def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost' #'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def ddp_cleanup():
    dist.destroy_process_group()


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def set_model(model_name, device=None, config=None, use_gan=False, half_input=True, half_output=True):
    model = getattr(models, model_name)(device=device, config=config, use_gan=use_gan,
                                        half_input=half_input, half_output=half_output)

    if device is not None:
        model.to(device)
    return model

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.count == 0:
            self.val = val
            self.sum = val * n
            self.count = n
            self.avg = self.sum / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


# write logs for tensorboard.
def write_summary(logger, loss_G, loss_nl, input_var, pred_var, target_var, pred_var_nl, target_var_nl,
                  epoch, index, data_len, is_train=True, loss_D=None,
                  full_logging=True, lr=None, is_train_nl_color=False):
    current_iter = epoch * data_len + index + 1

    if is_train is True:
        mode = 'train'
    else:
        mode = 'val'

    def draw_image(image, name):
        input_color_grid = torchvision.utils.make_grid(image, normalize=True, scale_each=True)
        input_color_grid = input_color_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + name, input_color_grid[::-1, :, :], epoch * index)

    if loss_G is not None and not is_train_nl_color:
        logger.add_scalar(mode + '/lossG', loss_G.data, epoch)
    if loss_nl is not None and is_train_nl_color:
        logger.add_scalar(mode + '/loss_nl', loss_nl.data, epoch)
    if loss_D is not None:
        logger.add_scalar(mode + '/lossD', loss_D.data, epoch)
    if lr is not None:
        logger.add_scalar(mode + '/lr', lr, epoch)

    # gt data.
    #scale_factor = 8.0
    target_normal_front, target_normal_back = target_var['normal']
    target_lbs_front, target_lbs_back = target_var['lbs']

    input_img = input_var[0][0]
    target_lf = target_lbs_front[0]
    target_lb = target_lbs_back[0]
    target_nf = target_normal_front[0]
    target_nb = target_normal_back[0]
    if target_var_nl is not None:
        target_nl_front, target_nl_back = target_var_nl['color']
        target_nl_front = target_nl_front[0]
        target_nl_back = target_nl_back[0]    

        target = torch.stack([input_img, target_lf, target_lb,
                            target_nf, target_nb, target_nl_front, target_nl_back])
    else:
        target = torch.stack([input_img, target_lf, target_lb,
                            target_nf, target_nb])
    draw_image(target, '/target')

    # stage 1.
    pred_lbs_front, pred_lbs_back = pred_var['lbs']
    pred_cf = pred_lbs_front[0].detach()
    pred_cb = pred_lbs_back[0].detach()
    #
    #pred_depth_front, pred_depth_back = pred_var['depth']
    #pred_nf = pred_depth_front[0].detach()
    #pred_nb = pred_depth_back[0].detach()

    pred_depth2normal_front, pred_depth2normal_back = pred_var['depth2normal']
    pred_nf1 = pred_depth2normal_front[0].detach()
    pred_nb1 = pred_depth2normal_back[0].detach()
    if pred_var_nl is not None:
        pred_no_light_front, pred_no_light_back = pred_var_nl['color']
        pred_nlf1 = pred_no_light_front[0].detach()
        pred_nlb1 = pred_no_light_back[0].detach()

        pred = torch.stack([pred_cf, pred_cb,
                            #pred_nf, pred_nb,
                            pred_nf1, pred_nb1,
                            pred_nlf1, pred_nlb1])
    else:
        pred = torch.stack([pred_cf, pred_cb,
                            #pred_nf, pred_nb,
                            pred_nf1, pred_nb1])
    
    draw_image(pred, '/pred_stage1-2')

def init_variables_depth(input_depth_front, target_depth, device=None):
    if device is not None:
        input_depth_front = input_depth_front.to(device)
        target_depth = target_depth.to(device)
    input_depth_front_var = torch.autograd.Variable(input_depth_front)
    target_depth_var = torch.autograd.Variable(target_depth)

    return input_depth_front_var, target_depth_var


def init_variables_color(input_color_front, target_color=None, device=None):
    if device is not None:
        input_color_front = input_color_front.to(device)
    input_color_front_var = torch.autograd.Variable(input_color_front)

    if target_color is not None:
        if device is not None:
            target_color = target_color.to(device)
        target_color_var = torch.autograd.Variable(target_color)

        return input_color_front_var, target_color_var
    else:
        return input_color_front_var


def load_checkpoint(model_paths, model, optimizer, start_epoch, is_evaluate=False, device="cuda:0", optimizer_D=''):
    set_opt_D = False
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

            if optimizer_D is not None and hasattr(checkpoint, 'optimizer_state_dict_disc'):
                try:
                    optimizer_D.load_state_dict(checkpoint['optimizer_state_dict_disc'])
                    set_opt_D = True
                    print('=> discriminator optimizer has been loaded')
                except:
                    print('=> optimizer(D) not loaded')

            print(("=> loaded checkpoint (resumed epoch is {})".format(start_epoch)))
            return model, optimizer, optimizer_D, start_epoch, set_opt_D

    print(("=> no checkpoint found at '{}'".format(model_path)))
    return model, optimizer, optimizer_D, start_epoch, set_opt_D


def load_checkpoint_eval(model_paths, model, optimizer, start_epoch, is_evaluate=False, device="cuda:0", optimizer_D=None):
    set_opt_D = False
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
                if model.model_name == 'UUUUU':
                    model_state_dict = collections.OrderedDict(
                        {k.replace('f2b', 'UUUUU').replace('warp.', 'UUUUU.').replace('cn2c.', 'UUUUU.')
                             .replace('fb2lr.', 'UUUUU.').replace('cn2d.', 'UUUUU.'): v for k, v in checkpoint['model_state_dict'].items()})
                elif model.model_name == 'AAAAA':
                    model_state_dict = collections.OrderedDict(
                        {k.replace('f2b', 'AAAAA').replace('warp.', 'AAAAA.').replace('cn2c.', 'AAAAA.')
                             .replace('fb2lr.', 'AAAAA.').replace('cn2d.', 'AAAAA.'): v for k, v in checkpoint['model_state_dict'].items()})
                elif model.model_name == 'AAAAM':
                    model_state_dict = collections.OrderedDict(
                        {k.replace('f2b', 'AAAAM').replace('warp.', 'AAAAM.').replace('cn2c.', 'AAAAM.')
                             .replace('fb2lr.', 'AAAAM.').replace('cn2d.', 'AAAAM.'): v for k, v in checkpoint['model_state_dict'].items()})
                elif model.model_name == 'AAAXA':
                    model_state_dict = collections.OrderedDict(
                        {k.replace('f2b', 'AAAXA').replace('warp.', 'AAAXA.').replace('cn2c.', 'AAAXA.')
                             .replace('fb2lr.', 'AAAXA.').replace('cn2d.', 'AAAXA.'): v for k, v in checkpoint['model_state_dict'].items()})
                elif model.model_name == 'AAAXM':
                    model_state_dict = collections.OrderedDict(
                        {k.replace('f2b', 'AAAXM').replace('warp.', 'AAAXM.').replace('cn2c.', 'AAAXM.')
                             .replace('fb2lr.', 'AAAXM.').replace('cn2d.', 'AAAXM.'): v for k, v in checkpoint['model_state_dict'].items()})
                elif model.model_name == 'SAAA':
                    model_state_dict = collections.OrderedDict(
                        {k.replace('f2b', 'SAAA').replace('warp.', 'SAAA.')
                             .replace('fb2lr.', 'SAAA.').replace('cn2d.', 'SAAA.'): v for k, v in checkpoint['model_state_dict'].items()})
                elif model.model_name == 'SAXA':
                    model_state_dict = collections.OrderedDict(
                        {k.replace('f2b', 'SAXA').replace('warp.', 'SAXA.')
                             .replace('fb2lr.', 'SAXA.').replace('cn2d.', 'SAXA.'): v for k, v in checkpoint['model_state_dict'].items()})
                elif model.model_name == 'SAXM':
                    model_state_dict = collections.OrderedDict(
                        {k.replace('f2b', 'SAXM').replace('warp.', 'SAXM.')
                             .replace('fb2lr.', 'SAXM.').replace('cn2d.', 'SAXM.'): v for k, v in checkpoint['model_state_dict'].items()})
                elif model.model_name == 'SAXM_v1':
                    model_state_dict = collections.OrderedDict(
                        {k.replace('f2b', 'SAXM_v1').replace('warp.', 'SAXM_v1.')
                             .replace('fb2lr.', 'SAXM_v1.').replace('cn2d.', 'SAXM_v1.'): v for k, v in checkpoint['model_state_dict'].items()})

            model.load_state_dict(model_state_dict, strict=False)
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print('=> generator optimizer has been loaded')
            except:
                print('=> optimizer(g) not loaded (trying to train a new network?)')

            if optimizer_D is not None and hasattr(checkpoint, 'optimizer_state_dict_disc'):
                try:
                    optimizer_D.load_state_dict(checkpoint['optimizer_state_dict_disc'])
                    set_opt_D = True
                    print('=> discriminator optimizer has been loaded')
                except:
                    print('=> optimizer(D) not loaded')

            print(("=> loaded checkpoint (resumed epoch is {})".format(start_epoch)))
            return model, optimizer, optimizer_D, start_epoch, set_opt_D

    print(("=> no checkpoint found at '{}'".format(model_path)))
    return model, optimizer, optimizer_D, start_epoch, set_opt_D

def load_checkpoint_GAN(model_path, model_G, model_D, optimizer_G, optimizer_D, start_epoch,
                        is_evaluate=False, device="cuda:0"):
    items = glob.glob(os.path.join(model_path, '*.pth.tar'))

    if len(items) > 0:
        if is_evaluate is True:
            model_path = os.path.join(model_path, 'model_best.pth.tar')
        else:
            model_path = items[len(items) - 1]

        print(("=> loading checkpoint '{}'".format(model_path)))
        checkpoint = torch.load(model_path, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        model_G.load_state_dict(checkpoint['model_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])
        model_D.load_state_dict(checkpoint['model_state_dict2'])
        optimizer_D.load_state_dict(checkpoint['optimizer_state_dict2'])

        print(("=> loaded checkpoint (resumed epoch is {})".format(start_epoch)))
    else:
        print(("=> no checkpoint found at '{}'".format(model_path)))
    return model_G, model_D, optimizer_G, optimizer_D, start_epoch


def save_checkpoint(model, optimizer, current_epoch, best_loss, is_best, optimizer_D=None,
                    ckpt_path='./checkpoints', ckpt_path_ext='./checkpoints',
                    model_name='human_recon', exp_name=''):
    # model_name = args.model_name.lower () + args.exp_name
    sub_dir = model_name.lower() + exp_name
    # check directories.
    if not os.path.exists(ckpt_path_ext):
        os.makedirs(ckpt_path_ext)
    if not os.path.exists(os.path.join(ckpt_path_ext, sub_dir)):
        os.makedirs(os.path.join(ckpt_path_ext, sub_dir))

    state = {'epoch': current_epoch, 'model': model_name, 'best_loss': best_loss,
             'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
             'optimizer_state_dict_disc': optimizer_D}

    filename = os.path.join(ckpt_path_ext, sub_dir,
                            'model_epoch%03d_loss%0.4f.pth.tar' % (current_epoch, best_loss))
    torch.save(state, filename)

    # save the best results within the directory.
    if is_best is True:
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        if not os.path.exists(os.path.join(ckpt_path, sub_dir)):
            os.makedirs(os.path.join(ckpt_path, sub_dir))

        best_name = os.path.join(ckpt_path, sub_dir, 'model_best.pth.tar')
        shutil.copyfile(filename, best_name)


def save_checkpoint2(model, save_dir, model_name, optimizer, current_epoch, best_loss, is_best,
                     ckpt_path='./checkpoints', ckpt_path_ext='I:/ioys_checkpoints'):
    # model_name = args.model_name.lower () + args.exp_name

    # check directories.
    if not os.path.exists(ckpt_path_ext):
        os.makedirs(ckpt_path_ext)
    if not os.path.exists(os.path.join(ckpt_path_ext, save_dir)):
        os.makedirs(os.path.join(ckpt_path_ext, save_dir))

    state = {'epoch': current_epoch, 'model': model_name, 'best_loss': best_loss,
             'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}

    filename = os.path.join(ckpt_path_ext, save_dir,
                            'model_epoch%03d_loss%0.4f.pth.tar' % (current_epoch, best_loss))
    torch.save(state, filename)

    # save the best results within the directory.
    if is_best is True:
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        if not os.path.exists(os.path.join(ckpt_path, save_dir)):
            os.makedirs(os.path.join(ckpt_path, save_dir))

        best_name = os.path.join(ckpt_path, save_dir, 'model_best.pth.tar')
        shutil.copyfile(filename, save_dir)


def print_info_gan(epoch, i, len_, losses_G, losses_D, batch_time, data_time, optimizer):
    print('Epoch: [{0}][{1}/{2}]\t'
          'LossG: {lossG:.4f}\t'
          'LossD: {lossD:.4f}\t'
          'Batch time: {batch_time:.2f}\t'
          'Data time: {data_time:.2f}\t'
          'Learning rate: {lr:.6f}\t'.
          format(epoch, i, len_, lossG=losses_G, lossD=losses_D, batch_time=batch_time, data_time=data_time,
                 lr=optimizer.param_groups[-1]['lr']))


def print_info(epoch, i, len_, losses_G, batch_time, data_time, optimizer):
    print('Epoch: [{0}][{1}/{2}]\t'
          'LossG: {loss:.4f}\t'
          'Batch time: {batch_time:.2f}\t'
          'Data time: {data_time:.4f}\t'
          'Learning rate: {lr:.8f}\t'.
          format(epoch, i, len_, loss=losses_G, batch_time=batch_time, data_time=data_time,
                 lr=optimizer.param_groups[-1]['lr']))


def save_for_bash(args, name):
    shell_script = 'python train_color2model.py '
    for k, v in sorted(vars(args).items()):
        if isinstance(v, str):
            shell_script += '--{}=\'{}\' '.format(str(k), str(v))
            # shell_script += '\\\n'
        elif isinstance(v, list):
            shell_script += '--{}='.format(str(k))
            for i in range(len(v)):
                shell_script += '{} '.format(str(v[i]))
            # shell_script += '\\\n'
        else:
            shell_script += '--{}={} '.format(str(k), str(v))
            # shell_script += '\\\n'

    filename = os.path.join('./', 'train_%s.sh ' % name)
    with open(filename, "w") as f:
        f.write(shell_script)
        f.close()


def print_exp_summary(args):
    # logging experimental information.
    cur_time = datetime.datetime.now()
    cur_time = cur_time.strftime("%m/%d/%Y, %H:%M:%S")

    exp_summary = '[experiment summary]\n' + \
                  '- batch size: {}(train)/{}(val)\n'.format(args.batch_size, args.batch_size_val) + \
                  '- in & out  : {}\n'.format(args.loader_conf) + \
                  '- network   : {}\n'.format(args.model_name) + \
                  '- lr(init.) : {}\n'.format(args.learning_rate) + \
                  '- loss func : {}\n'.format(args.loss_conf) + \
                  '- dataset   : {}\n'.format(args.dataset) + \
                  '- gpu ids   : {}\n'.format(args.gpu_ids) + \
                  '- launched  : {}\n'.format(cur_time) + \
                  '- run by    : {}\n'.format(args.hostname) + \
                  '- note      : {}\n'.format(args.exp_name)

    print(exp_summary)
    return exp_summary
