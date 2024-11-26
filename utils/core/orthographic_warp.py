import numpy as np
import trimesh
import torch
from skimage import measure
import cv2
import torch.nn as nn

def depth2pix(depth_map):
    b, _, h, w = depth_map.size()
    y_range = torch.arange (0, h).view (1, h, 1).expand (b, 1, h, w).type_as(depth_map)
    x_range = torch.arange (0, w).view (1, 1, w).expand (b, 1, h, w).type_as(depth_map)

    pixel_coords = torch.cat((x_range, y_range, depth_map), dim=1)
    pixel_coords_vec = pixel_coords.reshape(b, 3, -1)

    return pixel_coords_vec

def orth2pers(orth_depth, pers_color, pers_depth, fov, res, focal, half_flag, center_crop):
    if half_flag:
        if center_crop:
            res = (res / 2)
            fx = fy = (focal / 2)
            cx = (res / 4)
            cy = cz = (res / 2)
        else:
            res = (res / 2)
            fx = fy = (focal / 2)
            cx = cy = cz = (res / 2)
    else:
        if center_crop:
            fx = fy = focal
            cx = (res / 4)
            cy = cz = (res / 2)
        else:
            fx = fy = focal
            cx = (res / 2)
            cy = cz = (res / 2)

    orth = depth2pix(orth_depth).float()
    v = np.tan(np.radians(fov) / 2.0) * 2
    x = (orth[:, 0, :] - cx) / res * v
    y = (orth[:, 1, :] - cy) / res * v
    z = (orth[:, 2, :] - (cz / res)) * v + 1.0
    z[z < 0.6] = 0

    p = torch.stack([x, y, z], dim=1)
    K = torch.Tensor(np.identity(3)).float()
    K[0, 0] = K[1, 1] = fx
    K[0, 2], K[1, 2] = cx, cy
    pers = torch.matmul((K).cuda(), p)

    pers[:, 0, :] = (pers[:, 0, :] / pers[:, 2, :])
    pers[:, 1, :] = (pers[:, 1, :] / pers[:, 2, :])
    x_ = (pers[:, 0, :].float() - cx) / fx
    y_ = (pers[:, 1, :].float() - cy) / fy

    pers = pers.long()
    orth = orth.long()
    pers = torch.clamp_min(pers, 0)
    orth = torch.clamp_min(orth, 0)
    pers = torch.clamp_max(pers, res - 1)
    orth = torch.clamp_max(orth, res - 1)

    img_backward = torch.zeros_like(pers_color)
    depth_backward = torch.zeros_like(orth_depth)

    for i in range(orth.shape[0]):
        z_ = pers_depth[i, 0, pers[i, 1, :], pers[i, 0, :]].float()
        z_[z_ > 0] = (z_[z_ > 0] - (cz / res)) + 1.0
        x_[i, :] = x_[i, :] * z_
        y_[i, :] = y_[i, :] * z_
        z_p = torch.sqrt(z_ * z_ - x_[i, :] * x_[i, :] - y_[i, :] * y_[i, :]) * res - cz
        z_p[z_p < 0] = 0

        img_backward[i, :, orth[i, 1, :], orth[i, 0, :]] = pers_color[i, :, pers[i, 1, :], pers[i, 0, :]]
        depth_backward[i, 0, orth[i, 1, :], orth[i, 0, :]] = z_p / res

    return img_backward, depth_backward


def pers2orth(pers_color, pers_depth, res, focal, half_flag, center_crop):
    if half_flag:
        if center_crop:
            res = (res / 2)
            fx = fy = (focal / 2)
            cx = res / 4
            cy = cz = res / 2
        else:
            res = (res / 2)
            fx = fy = (focal / 2)
            cx = cy = cz = (res / 2)
    else:
        if center_crop:
            fx = fy = focal
            cx = res / 4
            cy = cz = res / 2
        else:
            fx = fy = focal
            cx = res / 2
            cy = cz = (res / 2)


    pers = depth2pix(pers_depth).float()
    x = pers[:, 0, :]
    y = pers[:, 1, :]
    z = pers[:, 2, :]

    z[z > 0] = (z[z > 0] * res - res / 2) / res + 1.0
    x = (x - cx) / fx * z
    y = (y - cy) / fy * z

    z_ = torch.sqrt(z * z - x * x - y * y)
    z_ = z_ * res - cz
    x_ = x * res + cx
    y_ = y * res + cy

    orth = torch.stack([x_ // 1, y_ // 1, z_ // 1], dim=1)
    orth = orth.long()
    pers = pers.long()

    pers = torch.clamp_min(pers, 0)
    orth = torch.clamp_min(orth, 0)
    pers = torch.clamp_max(pers, res - 1)
    orth = torch.clamp_max(orth, res - 1)

    img_foward = torch.zeros_like(pers_color)
    depth_foward = torch.zeros_like(pers_depth)

    for i in range(pers.shape[0]):
        img_foward[i, :, orth[i, 1, :], orth[i, 0, :]] = pers_color[i, :, pers[i, 1, :], pers[i, 0, :]]
        depth_foward[i, 0, orth[i, 1, :], orth[i, 0, :]] = z_[i, :] / (res)

    return img_foward, depth_foward
