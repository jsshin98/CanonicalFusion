import torch


# assuming the shape: [B x C X W X H]
def gradient_x(img):
    img = torch.nn.functional.pad(img, (0, 0, 1, 0), mode="replicate")  # pad a column to the end
    if len(img.shape) == 4:
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
    elif len(img.shape) == 3:
        gx = img[:, :-1, :] - img[:, 1:, :]
    return gx


def gradient_y(img):
    img = torch.nn.functional.pad(img, (0, 1, 0, 0), mode="replicate")  # pad a row on the bottom
    if len(img.shape) == 4:
        gy = img[:, :, :, :-1] - img[:, :, :, 1:]
    elif len(img.shape) == 3:
        gy = img[:, :, :-1] - img[:, :, 1:]
    return gy


# assuming the shape: [B x C X W X H X Z]
def gradient_z(img):
    # img = torch.nn.functional.pad(img, (0, 0, 1, 0), mode="replicate")  # pad a column to the end
    gz = img[:, :, :, :, :-1] - img[:, :, :, :, 1:]
    return gz


import torch.nn.functional as F
def get_grad3(x):
    x_p = F.pad(x, (0, 0, 0, 0, 0, 1))
    y_p = F.pad(x, (0, 0, 0, 1))
    z_p = F.pad(x, (0, 1))
    if len(x.shape) == 3:
        grad_x = x_p[:-1, :, :] - x_p[1:, :, :]
        grad_y = y_p[:, :-1, :] - x_p[:, 1:, :]
        grad_z = z_p[:, :, :-1] - z_p[:, :, 1:]
    elif len(x.shape) == 4:
        grad_x = x_p[:, 1, :, :] - x_p[:, :-1, :, :]
        grad_y = y_p[:, :, :1, :] - x_p[:, :, :-1, :]
        grad_z = z_p[:, :, :, 1:] - z_p[:, :, :, :-1]
    elif len(x.shape) == 5:
        grad_x = x_p[:, :, 1, :, :] - x_p[:, :, :-1, :, :]
        grad_y = y_p[:, :, :, :1, :] - y_p[:, :, :, :-1, :]
        grad_z = z_p[:, :, :, :, 1:] - z_p[:, :, :, :, :-1]
    return grad_x, grad_y, grad_z


def stack_gradients(img, mask=None):
    if mask is None:
        gx = gradient_x(img)
        gy = gradient_y(img)
    else:
        gx = gradient_x(img) * mask
        gy = gradient_y(img) * mask

    normal = torch.cat((gx, gy), dim=1)
    # normal = torch.squeeze(normal)

    return normal


def stack_2nd_order_derivatives(img, mask=None):
    gx = gradient_x(img)
    gy = gradient_y(img)
    gx_gx = gradient_x(gx)
    gx_gy = gradient_x(gy)
    gy_gx = gradient_y(gx)
    gy_gy = gradient_y(gy)

    if mask is not None:
        gx_gx = gx_gx * mask
        gx_gy = gx_gy * mask
        gy_gx = gy_gx * mask
        gy_gy = gy_gy * mask

    normal = torch.stack((gx_gx, gx_gy, gy_gx, gy_gy), dim=1)
    normal = torch.squeeze(normal)

    return normal
