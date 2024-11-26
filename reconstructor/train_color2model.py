import sys
from tqdm import tqdm
from train_utils import *
import time
import pdb
# support tensorboard for linux.
if sys.platform == 'win32':
    from tensorboardX import SummaryWriter
else:
    from torch.utils.tensorboard import SummaryWriter
#torch.autograd.set_detect_anomaly (True)


# discriminator is optional
def train(data_loader, dataset, model_DL, model_C, loss_builder, optimizer, scheduler, optimizer_C, scheduler_C, accelerator, epoch, is_train=True,
          is_train_nl_color=False, loss_conf=None, loss_conf_nl=None, res=512, real_dist=300.0, disc_conf=None, optimizer_D=None, scheduler_D=None,
          summary_dir=None, log_freq=100, print_freq=100, use_ddp=False, is_master=True, phase_epoch=100, device=None):
    # set variables.
    loss_batch = AverageMeter()
    loss_batch_D = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_sum = 0
    loss_sum_D = 0
    loss_D = None

    if optimizer_D is not None and scheduler_D is not None:
        use_gan = True
    else:
        use_gan = False

    if is_train is not True:
        model_DL.eval()
        model_C.eval()

    # putting log files outside the shared directory (the size becomes huge!)
    if summary_dir is not None:
        logger = SummaryWriter(summary_dir)

    data_end = time.time()
    iters = len(data_loader)

    if use_ddp:
        dist.barrier()

    loss_G, loss_D = None, None
    train_disc = True

    loss_conf_ind = 0
    loss_conf = loss_conf[loss_conf_ind]
    print('trainable parameters : ', count_parameters(model_DL))
    
    if is_train_nl_color:
        loss_conf_nl = loss_conf_nl[0]

    
    with tqdm(enumerate(data_loader)) as pbar:
        for i, datum in pbar:
            current_iter = epoch * len(data_loader) + i + 1
            # set timers.
            data_time.update(time.time() - data_end)
            batch_end = time.time()
            # initialize variables (in case of multiple images, they are returned as a tuple).
            input_var, target_var, xy = init_variables(datum, device=device)
            # compute and update losses.
            target_nl_color = target_var[-1]

            if not is_train_nl_color:
                loss_G, pred_var, target_var = \
                    loss_builder.build_loss(model_DL,
                                            input_var, target_var,
                                            xy=xy,
                                            config=loss_conf,
                                            w=None,
                                            )
                loss_nl, pred_var_nl, target_var_nl = None, None, None
            else:
                model_DL.eval()
                with torch.no_grad():
                    loss_G, pred_var, target_var = \
                        loss_builder.build_loss(model_DL,
                                                input_var, target_var,
                                                xy=xy,
                                                config=loss_conf,
                                                w=None,
                                                )
                loss_nl, pred_var_nl, target_var_nl = \
                    loss_builder.build_loss_nl(model_C,
                                               torch.cat([input_var[0], input_var[1]], dim=1),
                                               pred_var['depth2normal'],
                                               target_nl_color,
                                               config=loss_conf_nl,
                                               )
            # torch.distributed.all_reduce (loss_G)
            if is_train_nl_color:
                loss_batch.update(loss_nl.data, input_var[0].shape[0])
            else:
                loss_batch.update(loss_G.data, input_var[0].shape[0])

            loss_sum = loss_sum + loss_batch.val

            # proceed one step
            if is_train is True:
                if is_train_nl_color:
                    optimizer_C.zero_grad()
                    loss_nl.backward()
                    optimizer_C.step()
                    scheduler_C.step(epoch=(epoch + i / iters))
                else:
                    optimizer.zero_grad()
                    loss_G.backward()
                    # accelerator.backward(loss_G)
                    optimizer.step()
                    scheduler.step(epoch=(epoch + i / iters))

            # update the batch time
            batch_time.update(time.time() - batch_end)

            # plot information.
            if (i + 1) % print_freq == 0:
                # if accelerator.is_main_process:
                if is_master:
                    if not is_train_nl_color:
                        pbar.set_description('[{0}][{1}/{2}] lossG: {loss:.3f}/lossD: {lossD:.3f}, '
                                            'dataT: {dataT:0.4f}, batchT: {batchT:0.4f}, lr: {lr:0.6f}, batch: {batch_time:0.2f}'
                                            .format(epoch, i, iters,
                                                    loss=loss_batch.val,
                                                    lossD=loss_batch_D.val,
                                                    dataT=data_time.val,
                                                    batchT=batch_time.val,
                                                    lr=optimizer.param_groups[-1]['lr'],
                                                    batch_time=batch_time.val))
                    else:
                        pbar.set_description('[{0}][{1}/{2}] lossC: {loss:.3f}, '
                                            'dataT: {dataT:0.4f}, batchT: {batchT:0.4f}, lr: {lr:0.6f}, batch: {batch_time:0.2f}'
                                            .format(epoch, i, iters,
                                                    loss=loss_batch.val,
                                                    dataT=data_time.val,
                                                    batchT=batch_time.val,
                                                    lr=optimizer.param_groups[-1]['lr'],
                                                    batch_time=batch_time.val))
                        
                batch_time.reset()
                data_time.reset()
                loss_batch.reset()
                pbar.update(i / iters)
                
            # save results for tensorboard
            if summary_dir is not None and is_master and (i + 1) % log_freq == 0:
                write_summary(logger, loss_G, loss_nl, input_var, pred_var, target_var, pred_var_nl, target_var_nl, epoch, i, len(data_loader), 
                              is_train=is_train,
                              loss_D=loss_D,
                              full_logging=False,
                              lr=optimizer.param_groups[-1]['lr'],
                              is_train_nl_color=is_train_nl_color,
                              )
            # data processing time begins (excluding tensorboard logging)
            data_end = time.time()
        return loss_sum / len(data_loader)

def init_variables(datum, device=None):
    image_input, mask_input, smplx_input_f, smplx_input_b, data_name = datum['input']
    if 'label' in datum:
        depth_gt, lbs_gt, color_gt = datum['label']
        # depth_gt, lbs_gt, color_gt = datum['label']
        # depth_gt,  lbs_gt = datum['label']
    else:
        depth_gt, lbs_gt, color_gt = None, None, None
        # depth_gt, lbs_gt, color_gt = None, None
        # depth_gt, lbs_gt = None, None

    if device is not None:
        if image_input is not None:
            image_input = image_input.to(device)
        if mask_input is not None:
            mask_input = mask_input.to(device)
        if smplx_input_f is not None:
            smplx_input_f = smplx_input_f.to(device)
        if smplx_input_b is not None:
            smplx_input_b = smplx_input_b.to(device)
        if depth_gt is not None:
            depth_gt = depth_gt.to(device)
        if lbs_gt is not None:
            lbs_gt = lbs_gt.to(device)
        if color_gt is not None:
            color_gt = color_gt.to(device)

    if image_input is not None:
        image_input = torch.autograd.Variable(image_input)
    if mask_input is not None:
        mask_input = torch.autograd.Variable(mask_input)
    if smplx_input_f is not None:
        smplx_input_f = torch.autograd.Variable(smplx_input_f)
    if smplx_input_b is not None:
        smplx_input_b = torch.autograd.Variable(smplx_input_b)
    if depth_gt is not None:
        depth_gt = torch.autograd.Variable(depth_gt)
    if lbs_gt is not None:
        lbs_gt = torch.autograd.Variable(lbs_gt)
    if color_gt is not None:
        color_gt = torch.autograd.Variable(color_gt)

    input_var = (image_input, mask_input, smplx_input_f, smplx_input_b, data_name)

    if depth_gt is not None and lbs_gt is not None and color_gt is not None:
        target_var = (depth_gt, lbs_gt, color_gt)
    # if depth_gt is not None and lbs_gt is not None and color_gt is not None:
    #     target_var = (depth_gt, lbs_gt, color_gt)
    # if depth_gt is not None and lbs_gt is not None:
    #     target_var = (depth_gt, lbs_gt)
    else:
        target_var = None

    res = image_input.shape[2]
    batch_size = image_input.shape[0]
    focal = np.sqrt(res * res + res * res)
    x = np.reshape((np.linspace(0, res, res) - int(res / 2)) / focal,
                   [1, 1, -1, 1])
    y = np.reshape((np.linspace(0, res, res) - int(res / 2)) / focal,
                   [1, 1, 1, -1])
    x = np.tile(x, [batch_size, 1, 1, res])
    y = np.tile(y, [batch_size, 1, res, 1])
    xy = torch.Tensor(np.concatenate((x, y), axis=1)).to(device)

    return input_var, target_var, xy

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
