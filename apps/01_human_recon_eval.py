import yaml
import os
import sys
from omegaconf import OmegaConf
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from human_reconstructor import HumanReconstructorFacade

if __name__ == '__main__':
    # config_file = sys.argv[1]
    config_file = 'canonfusion_eval.yaml'
    config = OmegaConf.load(config_file)
    config_cli = OmegaConf.from_cli()
    args = OmegaConf.merge(config, config_cli) 

    data_dir = args.dataset.data_dir
    data_name = args.dataset.dataset_name

    args.path2data = os.path.join(data_dir, '%s/' % data_name)

    # camera parameters.
    with open('cam_params.yaml') as f:
        cam_params = yaml.load(f, Loader=yaml.FullLoader)
        cam_params = cam_params[args.dataset.cam_config]

    # reconstruction (mesh generation)
    recon_model = HumanReconstructorFacade(args, cam_params)

    while recon_model.data_list:
        image, mask, depth_front, depth_back, data_name, smpl_param = recon_model.fetch_next_data()
        process_time, process_time_full = recon_model(image, mask=mask, depth_front=depth_front, depth_back=depth_back, smpl_params=smpl_param, cam_params=cam_params)
        
        print('[{0}] time: network - {1:04f} (sec) / full process - {2:04f} (sec)'.format(data_name, process_time, process_time_full))
