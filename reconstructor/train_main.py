# official libraries
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.distributed as dist
import torch.distributed.optim.optimizer
import torch.distributed.autograd
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import sys
sys.path.append('../')
# custom modules
import dataloaders as ds
import options
from train_color2model import train
from train_utils import *
from set_path import set_path
import yaml

torch.backends.cudnn.benchmark = False

# cudnn.benchmark = True
cudnn.fastest = False
torch.autograd.set_detect_anomaly(True)

# main loop.
def main(args):
    # set misc
    if args.use_ddp:
        if args.local_rank != 0:
            args.is_master = False
    else:
        args.local_rank = 0

    if args.verbose and args.is_master:
        print_exp_summary(args)

    accelerator=None

    # set paths
    # args.dataset_path, args.cam_path, args.lbs_ckpt_path, args.checkpoints_path, args.checkpoints_ext, args.logs_dir = set_path()
    summary_root = os.path.join(args.logs_dir, args.model_name.lower() + args.exp_name)

    with open(args.cam_path) as f:
        cam_data = yaml.load(f, Loader=yaml.FullLoader)['CAM_512']
            
    # set GPUs
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda:{}".format(args.local_rank))
    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.device = accelerator.device
    world_size = torch.cuda.device_count()

    # load model
    model_DL = getattr(models, args.model_name)(split_last=args.split_last)
    model_DL.to(args.device)

    if args.train_nl_color:
        model_C = getattr(models, args.model_C_name)(split_last=args.split_last)
        model_C.to(args.device)
    else:
        model_C = None

    # set training scheme
    if world_size > 1:
        if args.use_ddp:
            if not torch.distributed.is_initialized():
                ddp_setup(args.local_rank, world_size)
            model_DL = DDP(
                model_DL,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True
            )
            if args.train_nl_color:
                model_C = DDP(
                    model_C,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank,
                    find_unused_parameters=True
                )
            map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
        elif args.use_dp:  # data parallel.
            gpu_ids = [k for k in range(world_size)]
            model_DL = torch.nn.DataParallel(model_DL, device_ids=gpu_ids)
            if args.train_nl_color:
                model_C = torch.nn.DataParallel(model_C, device_ids=gpu_ids)

            map_location = args.device
        else:
            map_location = args.device
    else:
        map_location = args.device
    # map_location = args.device

    # freeze depth & lbs network if train no light conditioned network
    if args.train_nl_color:
        if hasattr(model_DL, 'module'):
            model_DL.eval()
        else:
            model_DL.eval()

    # set optimizers and schedulers \w & \wo GAN
    if hasattr(model_DL, 'module'):
        optimizer_DL = torch.optim.Adam(model_DL.parameters(), args.learning_rate)
        if args.train_nl_color:
            optimizer_C = torch.optim.Adam(model_C.parameters(), args.learning_rate)
        else:
            optimizer_C = None
    else:
        optimizer_DL = torch.optim.Adam(model_DL.parameters(), args.learning_rate)
        if args.train_nl_color:
            optimizer_C = torch.optim.Adam(model_C.parameters(), args.learning_rate)
        else:
            optimizer_C = None

    optimizer_D = None
    scheduler_D = None

    # load checkpoint if required
    if args.continue_train and args.is_master:
        #model_path = [os.path.join(args.checkpoints_ext, args.model_name.lower().split('_')[0] + args.exp_name)]
        model_path = []
        if args.path2pretrained is not None:
            # model_path.append(os.path.join(args.checkpoints_path, args.path2pretrained)) 
            model_path.append(args.path2pretrained)

        model_DL, optimizer_DL, optimizer_D, args.start_epoch, set_opt_D = \
            load_checkpoint(model_path, model_DL,
                            optimizer_DL,
                            args.start_epoch,
                            is_evaluate=False,
                            device=map_location,
                            optimizer_D=optimizer_D)
        model_path_C = []
        model_path_C.append(args.path2pretrained_C)
        model_C, optimizer_C, optimizer_C, args.start_epoch, set_opt_D = \
            load_checkpoint(model_path_C, model_C,
                            optimizer_C,
                            args.start_epoch,
                            is_evaluate=False,
                            device=map_location,
                            optimizer_D=optimizer_C)

    scheduler_DL = torch.optim.lr_scheduler.LambdaLR(optimizer_DL,
                                                    lr_lambda=lambda epoch: 0.95 ** epoch)

    if args.train_nl_color:
        scheduler_C = torch.optim.lr_scheduler.LambdaLR(optimizer_C,
                                                    lr_lambda=lambda epoch: 0.95 ** epoch)
    else:
        scheduler_C = None

    if optimizer_D is not None:
        scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D,
                                                        lr_lambda=lambda epoch: 0.95 ** epoch)

    loss_builder = getattr(models, 'LossBuilderHuman')(device=args.device, data_path=args.dataset_path, cam=cam_data, lbs_ckpt=args.lbs_ckpt_path, batch=args.batch_size,
                                                       weight_conf=args.weight_conf, accelerator=accelerator)

    # log and save data
    dataset = getattr(ds, 'AugDataSet')
    train_dataset = dataset(dataset_path=args.dataset_path,
                            data_list=args.train_list,
                            bg_list=args.bg_list,
                            seg_model= None, # seg_model,
                            pred_res=args.res,
                            orig_res=args.res)
    if args.use_ddp:
        train_sampler = DistributedSampler(train_dataset)
        #val_sampler = DistributedSampler(val_dataset)
        shuffle = False  # already shuffled.
    else:
        val_sampler = train_sampler = None
        shuffle = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=False, drop_last=True)


    best_loss = np.inf
    for current_epoch in range(args.start_epoch, args.num_epoch):
        if args.use_ddp:
            train_sampler.set_epoch(current_epoch)  # randomize data (necessary?)

        if current_epoch % args.eval_freq == 0 or current_epoch == args.num_epoch:
            is_best = False

            current_loss = train(train_loader, dataset, model_DL, model_C,
                loss_builder, optimizer_DL, scheduler_DL,
                optimizer_C, scheduler_C, accelerator, 
                current_epoch,
                res=args.res,
                real_dist=args.real_dist,
                is_train=args.is_train,
                is_train_nl_color=args.train_nl_color,
                loss_conf=args.loss_conf,
                loss_conf_nl = args.loss_conf_nl_color,
                disc_conf=args.disc_conf, optimizer_D=optimizer_D,
                scheduler_D=scheduler_D, summary_dir=summary_root,
                log_freq=args.log_freq, print_freq=args.print_freq,
                is_master=args.is_master, phase_epoch=args.phase2_epoch, device=args.device)

            # check and save checkpoints (generator loss only)
            if args.is_master:
                if not args.train_nl_color:
                    if best_loss > current_loss:
                        best_loss = current_loss
                        is_best = True
                    save_checkpoint(model_DL, optimizer_DL, current_epoch, best_loss, is_best,
                                    optimizer_D=None,
                                    ckpt_path=args.checkpoints_path,
                                    ckpt_path_ext=args.checkpoints_ext,
                                    model_name=args.model_name.split('_')[0],
                                    exp_name=args.exp_name)
                if args.train_nl_color:
                    if best_loss > current_loss:
                        best_loss = current_loss
                        is_best = True
                    save_checkpoint(model_C, optimizer_C, current_epoch, best_loss, is_best,
                                    optimizer_D=None,
                                    ckpt_path=args.checkpoints_path,
                                    ckpt_path_ext=args.checkpoints_ext,
                                    model_name=args.model_name.split('_')[0],
                                    exp_name=args.exp_name)

    if args.use_ddp:
        ddp_cleanup()


if __name__ == '__main__':
    torch.cuda.empty_cache()
    # torch.multiprocessing.set_start_method('spawn')

    args = options.Options().parse()
    args.hostname = platform.node()
    if args.use_ddp:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"

    args.loader_conf = 'DEPTH+LBS'

    args.dataset_path = '/jisu/nas2/Human_Dataset/RECON_TRAIN/'
    args.cam_path = '../apps/cam_params.yaml'
    args.lbs_ckpt_path = args.dataset_path + 'resource/pretrained_models/lbs_ckpt'
    args.checkpoints_path = './checkpoints'
    args.checkpoints_ext = './checkpoints'
    args.logs_dir = './logs'
    
    args.num_epoch = 200  # absolute number (not relative).
    args.batch_size = 4
    args.batch_size_val = 1
    args.weight_conf = 1
    args.res = 512
    args.real_dist = 300.0
    args.verbose = True  # show status and values.
    args.is_train = True
    args.continue_train = False
    args.path2pretrained = ''#'/jisu/3DHuman/dataset/CanonicalFusion/resource/pretrained_models/main_ckpt/FINAL/DEPTH+LBS_ALL'
    args.path2pretrained_C= ''#'./checkpoints/basemodule_DEPTH+LBS'
    args.lbs_ckpt_path = ''
    args.loss_conf = []
    args.loss_conf.append('depth_l2_depth_ssim_depth2norm_l2_depth2norm_cos_lbs_l2')
    
    args.loss_conf_nl_color = [] ## loss conf for no light conditioned color
    args.loss_conf_nl_color.append('color_l2_color_vgg')

    # training data.
    args.data_name = 'TOTAL' # ['TH2.0', 'RP']
    
    # train all the data!
    args.train_list = '%s_TRAIN' % args.data_name

    args.val_list = '%s_VAL' % args.data_name
    args.test_list = '%s_TEST' % args.data_name
    args.bg_list = 'bg_indoor09'
    args.split_last = True

    custom_message = '_%s' % args.loader_conf
    args.exp_name = custom_message

    # run main function.
    main(args)
