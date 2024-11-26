import torch
import cv2


def volume2side(volume, device=None):
    B, H, W, D = volume.shape
    depth_left = torch.zeros(B, W, H)
    depth_right = torch.zeros_like(depth_left)

    if device is not None:
        depth_left = torch.autograd.Variable(depth_left)
        depth_right = torch.autograd.Variable (depth_right)
        depth_left = depth_left.to (device)
        depth_right = depth_right.to (device)

    DEPTH_MAX = D

    if D < H:
        fb_offset = D / 2
    if W < H:
        lr_offset = W / 2

    # z_left, z_right
    for z in range (D):
        z_right = volume[:, :, z, :]
        z_left = volume[:, :, (volume.shape[2] - 1) - z, :]
        if volume.shape[3] < volume.shape[1]:
            depth_right[z_right == 0] = (z + lr_offset)
            depth_left[z_left == 0] = (DEPTH_MAX - (z + lr_offset))
        else:
            depth_right[z_right == 0] = z
            depth_left[z_left == 0] = (DEPTH_MAX - z)

    depth_left = depth_left / DEPTH_MAX
    depth_right = depth_right / DEPTH_MAX

    depth_left = torch.flip (depth_left, dims=[2])
    depth_right = torch.flip (depth_right, dims=[2])

    depth_left = depth_left.unsqueeze(1)
    depth_right = depth_right.unsqueeze(1)

    return depth_left, depth_right


def volume2depth(volume):

    B, H, W, D = volume.shape
    depth_front = torch.zeros (B, W, H)
    depth_back = torch.zeros_like(depth_front)
    depth_left = torch.zeros_like(depth_front)
    depth_right = torch.zeros_like(depth_front)

    DEPTH_MAX = D

    if D < H:
        fb_offset = D / 2
    if W < H:
        lr_offset = W / 2

    # z_front, z_back
    for z in range(D):
        z_back = volume[:, :, :, z]
        z_front = volume[:, :, :, (volume.shape[3] - 1) - z]
        if volume.shape[3] < volume.shape[1]:
            depth_back[z_back == 0] = (z + fb_offset)
            depth_front[z_front == 0] = (DEPTH_MAX - (z + fb_offset))
        else:
            depth_back[z_back == 0] = z
            depth_front[z_front == 0] = (DEPTH_MAX - z)

    # z_left, z_right
    for z in range (D):
        z_right = volume[:, :, z, :]
        z_left = volume[:, :, (volume.shape[2] - 1) - z, :]
        if volume.shape[3] < volume.shape[1]:
            depth_right[z_right == 0] = (z + lr_offset)
            depth_left[z_left == 0] = (DEPTH_MAX - (z + lr_offset))
        else:
            depth_right[z_right == 0] = z
            depth_left[z_left == 0] = (DEPTH_MAX - z)

    depth_front = depth_front / DEPTH_MAX
    depth_back = depth_back / DEPTH_MAX
    depth_left = depth_left / DEPTH_MAX
    depth_right = depth_right / DEPTH_MAX

    depth_left = torch.flip (depth_left, dims=[2])
    depth_right = torch.flip (depth_right, dims=[2])

    depth_front = depth_front.unsqueeze(1)
    depth_back = depth_back.unsqueeze(1)
    depth_left = depth_left.unsqueeze(1)
    depth_right = depth_right.unsqueeze(1)

    return depth_front, depth_back, depth_left, depth_right