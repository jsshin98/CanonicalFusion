import torch
from utils.geometry import perspective_projection
import constants

def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared =  x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)

def angle_prior(pose):
    """
    Angle prior that penalizes unnatural bending of the knees and elbows
    """
    # We subtract 3 because pose does not include the global rotation of the model
    return torch.exp(pose[:, [55-3, 58-3, 12-3, 15-3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2

def camera_fitting_loss(model_joints, camera_t, camera_t_est, camera_center, joints_2d, joints_conf,
                        focal_length=5000, depth_loss_weight=100):
    """
    Loss function for camera optimization.
    """

    # Project model joints
    batch_size = model_joints.shape[0]
    rotation = torch.eye(3, device=model_joints.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)
    
    op_joints = ['OP RHip', 'OP LHip', 'OP RShoulder', 'OP LShoulder']
    op_joints_ind = [constants.JOINT_IDS[joint] for joint in op_joints]

    # op_joints_ind=[]
    # for idx in range(49):
    #     op_joints_ind.append(idx)

    gt_joints = ['Right Hip', 'Left Hip', 'Right Shoulder', 'Left Shoulder']
    gt_joints_ind = [constants.JOINT_IDS[joint] for joint in gt_joints]

    # gt_joints_ind=[]
    # for idx in range(49):
    #     gt_joints_ind.append(idx)

    reprojection_error_op = (joints_2d[:, op_joints_ind] -
                             projected_joints[:, op_joints_ind]) ** 2
    reprojection_error_gt = (joints_2d[:, gt_joints_ind] -
                             projected_joints[:, gt_joints_ind]) ** 2

    # Check if for each example in the batch all 4 OpenPose detections are valid, otherwise use the GT detections
    # OpenPose joints are more reliable for this task, so we prefer to use them if possible
    is_valid = (joints_conf[:, op_joints_ind].min(dim=-1)[0][:,None,None] > 0).float()
    reprojection_loss = (is_valid * reprojection_error_op + (1-is_valid) * reprojection_error_gt).sum(dim=(1,2))

    # Loss that penalizes deviation from depth estimate
    depth_loss = (depth_loss_weight ** 2) * (camera_t[:, 2] - camera_t_est[:, 2]) ** 2

    total_loss = reprojection_loss + depth_loss
    return total_loss.sum()
