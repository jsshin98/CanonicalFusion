import platform


def set_path():
    hostname = platform.node()

    data_path = '/jisu/3DHuman/dataset/CanonicalFusion'
    cam_path = 'camera_config_512.yaml'
    lbs_ckpt_path = data_path + 'resource/pretrained_models/lbs_ckpt/uv_based'
    checkpoint = './checkpoints'
    checkpoint_ext = './checkpoints'
    log_dir = './logs'
    return data_path, cam_path, lbs_ckpt_path, checkpoint, checkpoint_ext, log_dir

