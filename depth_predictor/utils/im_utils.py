import numpy as np
import torch
import torch.nn as nn


# input depth x is normalized by 255.0
def get_normal(x, normalize=True, cut_off=0.2):
    def gradient_x(img):
        img = torch.nn.functional.pad (img, (0, 0, 1, 0), mode="replicate")  # pad a column to the end
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(img):
        img = torch.nn.functional.pad (img, (0, 1, 0, 0), mode="replicate")  # pad a row on the bottom
        gy = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gy

    def normal_from_grad(grad_x, grad_y, depth):
        grad_z = torch.ones_like (grad_x) / 255.0
        n = torch.sqrt (torch.pow (grad_x, 2) + torch.pow (grad_y, 2) + torch.pow (grad_z, 2))
        normal = torch.cat ((grad_y / n, grad_x / n, grad_z / n), dim=1)
        normal += 1
        normal /= 2
        if normalize is False:  # false gives 0~255, otherwise 0~1.
            normal *= 255

        # remove normals along the object discontinuities and outside the object.
        normal[depth.repeat (1, 3, 1, 1) < cut_off] = 0
        return normal

    if x is None:
        return None

    if len(x.shape) == 3:
        x = x.unsqueeze(1)

    x = x.float()
    grad_x = gradient_x(x)
    grad_y = gradient_y(x)

    if x.shape[1] == 1:
        return normal_from_grad(grad_x, grad_y, x)
    else:
        normal = [normal_from_grad
                  (grad_x[:, k, :, :].unsqueeze(1), grad_y[:, k, :, :].unsqueeze(1), x[:, k, :, :].unsqueeze(1)) for k in range(x.shape[1])]
        return torch.cat(normal, dim=1)


def depth2normal(img, normalize=True):
    zy, zx = np.gradient(img.squeeze())
    normal = np.dstack((-zx, -zy, np.ones_like(np.array(img))))

    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    # offset and rescale values to be in 0-255
    normal += 1
    normal /= 2
    if normalize is not True:
        normal *= 255

    return normal


def depth2normal_torch(img, normalize=True):
    # input: B x W X H
    # output: B x 3 x W x H

    B, C, _, _ = img.size()
    g_y, g_x = gradient_torch(img)
    g_z = torch.ones_like(g_x)
    n = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2) + g_z)
    normal = torch.cat((g_y/n, g_x/n, g_z/n), dim=1)

    normal += 1
    normal /= 2
    if normalize is not True:
        normal *= 255
    return normal


def gradient_torch(img):
    img = torch.mean(img, 1, True)
    fx = np.array ([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy (fx).float ().unsqueeze (0).unsqueeze (0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1 (img)

    fy = np.array ([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d (1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy (fy).float ().unsqueeze (0).unsqueeze (0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2 (img)

    #     grad = torch.sqrt(torch.pow(grad_x,2) + torch.pow(grad_y,2))

    return grad_y, grad_x
