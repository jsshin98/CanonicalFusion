import numpy as np
import trimesh
import torch
from skimage import measure
import time
import cv2
import torch.nn.functional as F
import torch.nn
from PIL import Image
from depth_predictor.utils.visualizer import to_mesh


# transform stacked depth maps to a truncated signed distance function (TSDF) volume
def depth2volume(stacked_depth, voxel_size=256, z_level=64, slope=0.01, max_stack=20, binary_mode=False):
    stacked_depth = stacked_depth * (z_level / voxel_size)
    stacked_depth = np.minimum(stacked_depth, z_level - 1)
    stacked_depth[stacked_depth == 0] = np.nan
    stacked_depth = np.sort(stacked_depth, axis=2)
    stack_size = np.min([stacked_depth.shape[2], max_stack])
    slope = np.float64(slope * (z_level / voxel_size))

    sdf_volume = np.zeros((voxel_size, voxel_size, z_level))
    for i in range(voxel_size):
        for j in range(voxel_size):
            sdf_volume[i, j, :] = compute_sdf_value(stacked_depth[i, j, :], z_level, stack_size, slope)

    if binary_mode == True:
        sdf_volume[sdf_volume > 0] = 1
        sdf_volume[sdf_volume < 0] = -1

    return sdf_volume


# transform stacked depth maps to a truncated signed distance function (TSDF) volume
def depth2volume_float(stacked_depth, voxel_size=256, z_level=64, slope=0.01, max_stack=20, binary_mode=False):

    stacked_depth = stacked_depth * (z_level / voxel_size)
    stacked_depth = np.minimum(stacked_depth, z_level - 1)
    stacked_depth[stacked_depth == 0] = np.nan
    stacked_depth = np.sort(stacked_depth, axis=2)
    stack_size = np.min ([stacked_depth.shape[2], max_stack])
    slope = np.float64 (slope * (z_level / voxel_size))

    sdf_volume = np.zeros ((voxel_size, voxel_size, z_level))
    for i in range (voxel_size):
        for j in range (voxel_size):
            sdf_volume[i, j, :] = compute_sdf_value (stacked_depth[i, j, :], z_level, stack_size, slope)

    if binary_mode == True:
        sdf_volume[sdf_volume > 0] = 1
        sdf_volume[sdf_volume < 0] = 0

    return sdf_volume


def depth2volume_single(depth_map, voxel_size=256, z_level=128, slope=0.01):

    depth_map = depth_map * (z_level / voxel_size)
    slope = np.float64(slope * (z_level / voxel_size))
    sdf_volume = np.ones((voxel_size, voxel_size, z_level))
    depth_map[depth_map == 0] = np.nan
    sdf_volume[:, :, 0] = depth_map.astype(np.float) * slope
    for k in range (1, z_level):
        sdf_volume[:, :, k] = sdf_volume[:, :, k - 1] - slope
    sdf_volume[np.isnan(sdf_volume)] = z_level*slope
    return sdf_volume


def depth2volume_double(depth_front, depth_back, voxel_size=256, slope=0.01):

    sdf_volume = np.ones((voxel_size, voxel_size, voxel_size))
    confidence = np.zeros_like (sdf_volume)

    depth_front[depth_front == 0] = np.Inf
    sdf_volume[:, :, 0] = depth_front.astype(np.float) * slope
    for k in range (1, voxel_size):
        sdf_volume[:, :, k] = sdf_volume[:, :, k - 1] - slope
        confidence[:, :, k] = (sdf_volume[:, :, k] > -0.1).astype(np.int)

    depth_back = (voxel_size - depth_back)
    depth_back[depth_back == 0] = -np.Inf
    sdf_volume[:, :, voxel_size - 1] = depth_back.astype(np.float) * slope
    for k in range (1, voxel_size - 2):
        sdf_volume[:, :, voxel_size - k - 1] = \
            np.maximum(sdf_volume[:, :, voxel_size - k - 1], sdf_volume[:, :, voxel_size - k] - slope)
        confidence[:, :, voxel_size - k - 1] = \
            (sdf_volume[:, :, voxel_size - k - 1] > -0.1).astype(np.int)

    sdf_volume[np.isinf(sdf_volume)] = voxel_size * slope

    return sdf_volume, confidence


def depth2occ_double(depth_front, depth_back, voxel_size=256, slope=0.01):
    depth_front[depth_front == 0] = voxel_size + 1
    occ_grid1 = np.dstack([depth_front * slope]*voxel_size)
    slope_all = np.arange(0, -slope*voxel_size, -slope)
    occ_grid1 = occ_grid1 + slope_all.reshape(1, 1, -1)

    depth_back[depth_back == 0] = voxel_size + 1
    occ_grid2 = np.dstack([(voxel_size - depth_back - slope)*slope]*voxel_size)
    slope_all = np.arange(-slope * voxel_size, 0, slope)
    occ_grid2 = occ_grid2 + slope_all.reshape (1, 1, -1)

    occ_grid = np.maximum(occ_grid1, occ_grid2)
    occ_grid[occ_grid > 0] = 1
    occ_grid[occ_grid <= 0] = 0
    return occ_grid


# modified version of Jumi Kang
def depth2occ_double_torch_jumi(depth_front, depth_back, voxel_size=256, slope=0.01):
    depth_front[depth_front == 0] = voxel_size + 1
    occ_grid1 = torch.stack([depth_front * slope]*voxel_size, dim=3)
    slope_all = torch.arange(0, -slope*voxel_size, -slope)
    occ_grid1 = occ_grid1 + slope_all.reshape(1, 1, -1)

    depth_back[depth_back == 0] = voxel_size + 1
    occ_grid2 = torch.stack([(voxel_size - depth_back - slope)*slope]*voxel_size, dim=3)
    slope_all = torch.arange(-slope * voxel_size, 0, slope)
    occ_grid2 = occ_grid2 + slope_all.reshape(1, 1, -1)

    occ_grid = torch.max(occ_grid1, occ_grid2)
    occ_grid[occ_grid > 0] = 1
    occ_grid[occ_grid <= 0] = 0
    return occ_grid


# modified for actual training on March 21, 2021.
def depth2occ_double_torch(depth_front, depth_back, voxel_size=256, slope=0.01, device=None, binarize=True):
    occ_grid = torch.ones((depth_front.shape[0], depth_front.shape[1], depth_front.shape[2], voxel_size),
                          requires_grad=True)
    occ_grid = torch.autograd.Variable(occ_grid)
    if device is not None:
        occ_grid = occ_grid.to(device)

    cost_front = depth_front * slope * 255.0
    cost_back = (1 - depth_back * 255.0) * slope
    for k in range(0, voxel_size):
        occ_grid[:, :, :, k] = torch.max(cost_front - slope * k, cost_back + slope * k)

    if binarize:
        occ_grid[occ_grid > 0] = 1
        occ_grid[occ_grid <= 0] = 0

    return occ_grid

def depth2occ_2view_torch_wcolor(color_front, color_back, depth_front, depth_back, voxel_size=256, slope=0.01, binarize=True, device=None):

    if depth_front.shape[1] == 1:
        depth_front = depth_front.squeeze(1)
    if depth_back.shape[1] == 1:
        depth_back = depth_back.squeeze(1)

    occ_grid = torch.ones((depth_front.shape[0], depth_front.shape[1], depth_front.shape[2], voxel_size))
    occ_grid_color = torch.ones((color_front.shape[0], color_front.shape[1], color_front.shape[2], voxel_size))

    if device is not None:
        occ_grid = torch.autograd.Variable(occ_grid)
        occ_grid = occ_grid.to(device)
        occ_grid_color = torch.autograd.Variable(occ_grid_color)
        occ_grid_color = occ_grid_color.to(device)

    cost_front = depth_front * slope * voxel_size
    cost_back = (1 - depth_back * voxel_size) * slope

    for k in range(0, voxel_size):
        occ_grid[:, :, :, k] = torch.max(cost_front - slope * k, cost_back + slope * k)

    if occ_grid.shape[2] < occ_grid.shape[3]:
        offset = int(occ_grid.shape[2] / 2)
        occ_grid = torch.nn.functional.pad(occ_grid, (0, 0, offset, offset), "constant", 1)

    if binarize:
        occ_grid[occ_grid > 0] = 1
        occ_grid[occ_grid <= 0] = 0

    return occ_grid

def depth2occ_2view_torch(depth_front, depth_back, z_min=120, z_max=320,
                          voxel_size=256, binarize=False, device=None):
    if depth_front.shape[1] == 1:
        depth_front = depth_front.squeeze(1)
    if depth_back.shape[1] == 1:
        depth_back = depth_back.squeeze(1)
    z_range = z_max - z_min
    slope = z_range / voxel_size
    occ_grid = torch.ones((depth_front.shape[0], depth_front.shape[1], depth_front.shape[2], voxel_size))

    if device is not None:
        occ_grid = torch.autograd.Variable(occ_grid)
        occ_grid = occ_grid.to(device)

    cost_front = depth_front - z_min
    cost_back = z_min - depth_back

    for k in range(0, voxel_size):
        occ_grid[:, :, :, k] = torch.max(cost_front - slope * k, cost_back + slope * k)

    if occ_grid.shape[2] < occ_grid.shape[3]:
        offset = int(occ_grid.shape[2] / 2)
        occ_grid = torch.nn.functional.pad(occ_grid, (0, 0, offset, offset), "constant", 1)

    if binarize:
        occ_grid[occ_grid > 0] = 1.0
        occ_grid[occ_grid <= 0] = -1.0
    else:
        occ_grid[occ_grid > 10] = 5.0
        occ_grid[occ_grid < -10] = -5.0
        occ_grid /= 5.0
    return occ_grid

# modified for actual training on March 21, 2021.
def depth2occ_4view_torch(depth_front, depth_back, depth_left, depth_right,
                          center_crop=False, device=None, binarize=False, voxel_size=512):

    pred_fb = depth2occ_2view_torch(depth_front, depth_back, voxel_size=voxel_size,
                                    device=device, binarize=binarize)
    pred_lr = depth2occ_2view_torch(depth_left, depth_right, voxel_size=voxel_size,
                                    device=device, binarize=binarize)


    if pred_fb.shape[2] < pred_fb.shape[3]:
        offset = int(pred_fb.shape[2] / 2)
        width = pred_fb.shape[1]
        if center_crop:  # crop
            pred_volume1 = pred_fb[:, :, :, offset:width - offset]
            pred_volume2 = pred_lr[:, :, :, offset:width - offset]
        else:  # pad
            pred_volume1 = torch.nn.functional.pad(pred_fb, (0, 0, offset, offset), "constant", 1)
            pred_volume2 = torch.nn.functional.pad(pred_lr, (0, 0, offset, offset), "constant", 1)
    else:  # size not changed.
        pred_volume1 = pred_fb
        pred_volume2 = pred_lr

    # max() gives a complete shape whereas min() gives a visual hull.
    occ_grid = torch.max(pred_volume1, torch.rot90(pred_volume2, -1, [2, 3]))

    if binarize:
        occ_grid[occ_grid > 0] = 1
        occ_grid[occ_grid <= 0] = 0

    return occ_grid

def depth2volume_lstm(depth_map, voxel_size=256, z_level=128, slope=0.01):

    depth_map = depth_map * (z_level / voxel_size)
    slope = slope * (z_level / voxel_size)
    sdf_volume = np.ones ((voxel_size, voxel_size, 2))

    depth_map[depth_map == 0] = np.nan
    sdf_volume[:, :, 0] = depth_map.astype (np.float) * slope
    sdf_volume[:, :, 1] = sdf_volume[:, :, 0] - slope
    sdf_volume[np.isnan(sdf_volume)] = z_level * slope
    return sdf_volume


def compute_sdf_value(stacked_depth_1d, z_level, stack_size, slope):
    data_1d = np.ones(z_level) * z_level * slope

    for k in range (0, stack_size, 2):
        if np.isnan(stacked_depth_1d[k]) or np.isnan(stacked_depth_1d[k + 1]):
            return data_1d

        idx_a = np.arange (0, z_level, 1, dtype=int)  # 0 to z_level - 1 decrease by 1
        y_a = (-idx_a + stacked_depth_1d[k]) * slope

        idx_b = np.arange (0, z_level, 1, dtype=int)
        y_b = (idx_b - stacked_depth_1d[k + 1]) * slope

        new_data = np.maximum (y_a, y_b)
        data_1d = np.minimum (data_1d, new_data)

    return data_1d


def volume2mesh(sdf, visualize=True, level=0.0):

    if np.min(sdf) > level or np.max(sdf) < level:
        print('no surface found\n')
        return

    # image-aligned visualization.
    sdf = np.rot90(sdf, k=1)
    sdf = np.flip(sdf, axis=0)
    sdf = np.flip(sdf, axis=1)
    sdf = np.flip(sdf, axis=2)

    vertices, faces, normals, _ = measure.marching_cubes(sdf, level=level)
    vertex_color = np.ones_like(vertices) * 128
    mesh = trimesh.Trimesh (vertices=vertices, vertex_colors=vertex_color,
                            faces=faces, vertex_normals=normals)

    if visualize is True:
        mesh.show()

    return mesh


def volume2meshinfo(sdf, visualize=True, level=0.0):

    if np.min(sdf) > level or np.max(sdf) < level:
        print('no surface found\n')
        return

    # image-aligned visualization.
    # sdf = np.rot90(sdf, k=1)
    # sdf = np.flip(sdf, axis=0)
    # sdf = np.flip(sdf, axis=1)
    # sdf = np.flip(sdf, axis=2)

    # vertices, faces, normals, _ = measure.marching_cubes(sdf, level=level)
    # return vertices, faces, normals
    from torchmcubes import marching_cubes
    vertices, faces = marching_cubes(torch.Tensor(sdf), level)
    normals = None

    # return vertices.numpy()[:, ::-1], faces.numpy()[:, ::-1], normals
    return vertices.numpy(), faces.numpy(), normals


def volume2colormesh(sdf, visualize=True, level=0.0, binarize=False):
    if np.min(sdf) > level or np.max(sdf) < level:
        print('no surface found\n')
        return

    # truncated sdf (it doesn't change the result)
    # sdf[sdf > 1] = 1
    # sdf[sdf < -1] = -1

    # image-aligned visualization.
    sdf = np.rot90(sdf, k=1)
    sdf = np.flip(sdf, axis=0)
    sdf = np.flip(sdf, axis=1)
    sdf = np.flip(sdf, axis=2)

    vertices, faces, normals, _ = measure.marching_cubes_lewiner(sdf, level=level)
    vertex_color = np.ones_like(vertices) * 128
    mesh = trimesh.Trimesh (vertices=vertices, vertex_colors=vertex_color,
                            faces=faces, vertex_normals=normals)

    if visualize is True:
        mesh.show()


def colorize_model2(pred_mesh, img_front, img_back):
    vertex_num = pred_mesh.vertices.shape[0]
    vertices = pred_mesh.vertices

    pred_normals = trimesh.geometry.weighted_vertex_normals(vertex_num, pred_mesh.faces,
                                                            pred_mesh.face_normals,
                                                            pred_mesh.face_angles,
                                                            use_loop=False)

    model_colors = np.zeros_like(pred_normals)
    # img_front = np.flip(img_front, 0)
    # img_back = np.flip(img_back, 0)

    for k in range(vertex_num):
        u, v = vertices[k, 0], vertices[k, 1]
        u_d = u - np.floor(u)

        if img_front.shape[0] == 256:
            u = min(u.astype(int), 254)
            v_d = v - np.floor(v)
            v = min(v.astype(int), 254)
        else:
            u = min(u.astype(int), 510)
            v_d = v - np.floor(v)
            v = min(v.astype(int), 510)

        if pred_normals[k, 2] < 0.0:
            model_colors[k, 0] = (img_front[v, u, 2] * v_d + img_front[v + 1, u, 2] * (
                    1 - v_d)) * u_d + \
                                 (img_front[v, u + 1, 2] * v_d + img_front[v + 1, u + 1, 2] * (
                                         1 - v_d)) * (1 - u_d)
            model_colors[k, 1] = (img_front[v, u, 1] * v_d + img_front[v + 1, u, 1] * (
                    1 - v_d)) * u_d + \
                                 (img_front[v, u + 1, 1] * v_d + img_front[v + 1, u + 1, 1] * (
                                         1 - v_d)) * (1 - u_d)
            model_colors[k, 2] = (img_front[v, u, 0] * v_d + img_front[v + 1, u, 0] * (
                    1 - v_d)) * u_d + \
                                 (img_front[v, u + 1, 0] * v_d + img_front[v + 1, u + 1, 0] * (
                                         1 - v_d)) * (1 - u_d)
        else:
            model_colors[k, 0] = (img_back[v, u, 2] * v_d + img_back[v + 1, u, 2] * (
                    1 - v_d)) * u_d + \
                                 (img_back[v, u + 1, 2] * v_d + img_back[v + 1, u + 1, 2] * (
                                         1 - v_d)) * (1 - u_d)
            model_colors[k, 1] = (img_back[v, u, 1] * v_d + img_back[v + 1, u, 1] * (
                    1 - v_d)) * u_d + \
                                 (img_back[v, u + 1, 1] * v_d + img_back[v + 1, u + 1, 1] * (
                                         1 - v_d)) * (1 - u_d)
            model_colors[k, 2] = (img_back[v, u, 0] * v_d + img_back[v + 1, u, 0] * (
                    1 - v_d)) * u_d + \
                                 (img_back[v, u + 1, 0] * v_d + img_back[v + 1, u + 1, 0] * (
                                         1 - v_d)) * (1 - u_d)

    color_mesh = trimesh.Trimesh(vertices=vertices,
                                 vertex_colors=model_colors,
                                 faces=pred_mesh.faces,
                                 process=False,
                                 maintain_order=True)
    return color_mesh, model_colors


def colorize_model(volume, img_front, img_back, mask=None, subdivide=False, texture_map=False):
    if isinstance(volume, trimesh.Trimesh):
        pred_mesh = volume
    else:
        vertices, faces, normals = volume2meshinfo(volume, level=0.0)
        pred_mesh = trimesh.Trimesh(vertices=vertices,
                                    faces=faces,
                                    process=True, maintain_order=False)
    if subdivide:
        pred_mesh = pred_mesh.subdivide()

    vertices = pred_mesh.vertices
    faces = pred_mesh.faces
    vertex_num = vertices.shape[0]

    pred_normals = trimesh.geometry.weighted_vertex_normals(vertex_num, faces,
                                                            pred_mesh.face_normals,
                                                            pred_mesh.face_angles,
                                                            use_loop=False)

    model_colors = np.zeros_like(pred_normals)
    if texture_map:
        img_front = np.flip(img_front, axis=0)
        img_back = np.flip(img_back, axis=0)
    else:
        img_front = np.rot90(img_front, k=1)
        img_front = np.flip(img_front, axis=0)
        img_back = np.rot90(img_back, k=1)
        img_back = np.flip(img_back, axis=0)
        mask = np.rot90(mask, k=1)
        mask = np.flip(mask, axis=0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    iter_erode = 3
    iter_dilate = 4
    img_front_eroded = cv2.erode(img_front, kernel, iterations=iter_erode)
    img_back_eroded = cv2.erode(img_back, kernel, iterations=iter_erode)
    if mask is not None:
        mask = cv2.resize(mask, (img_front.shape[0], img_front.shape[1]), interpolation=cv2.INTER_NEAREST)
    else:
        mask = np.sum(img_front_eroded)
        mask[mask > 0] = 255.0
    mask_eroded = cv2.erode(mask, kernel, iterations=iter_erode)

    img_front_dialated = cv2.dilate(img_front_eroded, kernel, iterations=iter_dilate)
    img_back_dialated = cv2.dilate(img_back_eroded, kernel, iterations=iter_dilate)
    mask_dialated = cv2.dilate(mask_eroded, kernel, iterations=iter_dilate)
    dist = cv2.distanceTransform(1 - mask_eroded.astype(np.uint8), cv2.DIST_L2, 5)
    # cv2.imshow("dist", dist / 100)
    # cv2.waitKey(0)
    min_val = 0
    max_val = 7
    bw = np.clip(dist, a_min=min_val, a_max=max_val) / (max_val*2)  # 0.5~1.0
    bw[dist <= min_val] = 0.0
    bw[dist > max_val] = 0.5
    # bw[:] = 0

    for i in range(img_front.shape[1]):
        for j in range(img_front.shape[0]):
            if mask_eroded[j, i] == 0 and mask_dialated[j, i] > 0:
                img_front[j, i, :] = img_front_dialated[j, i, :]
                img_back[j, i, :] = img_back_dialated[j, i, :]
    img_front = (img_front * 255)
    img_back = (img_back * 255)

    # resize and crop
    def resize_and_crop(image, d=1):
        w, h = image.shape[1]+d*2, image.shape[0]+d*2
        image = cv2.resize(image, (h, w))
        image = image[d:w-1, d:h-1, :]
        return image
    img_front = resize_and_crop(img_front, d=1)
    img_back = resize_and_crop(img_back, d=1)
    # normal_front = np.zeros((img_front.shape))
    # normal_back = np.zeros((img_back.shape))

    vts_uv = np.zeros_like(vertices[:, 0:2])

    for k in range(vertex_num):
        # scikit-learn marching cubes (original)
        # u, v = vertices[k, 0], vertices[k, 1]
        # torchmcubes marching cubes (reversed coordinate and colors)
        u, v = vertices[k, 2], vertices[k, 1]
        u *= 2 # res==1024
        v *= 2 # res==1024
        u_d = u - np.floor(u)
        if img_front.shape[0] == 1024:
            u = min(u.astype(int), 1022)
            v_d = v - np.floor(v)
            v = min(v.astype(int), 1022)
        elif img_front.shape[0] == 512:
            u = min(u.astype(int), 510)
            v_d = v - np.floor(v)
            v = min(v.astype(int), 510)
        elif img_front.shape[0] == 2048:
            u = min(u.astype(int), 2046)
            v_d = v - np.floor(v)
            v = min(v.astype(int), 2046)

        rgb_f = np.zeros(3)
        rgb_b = np.zeros(3)

        # if pred_normals[k, 0] < 0.0:
        vts_uv[k, 0] = (v + v_d) / 1024#1024 #512#2048#512
        vts_uv[k, 1] = (u + u_d) / 2048 + 0.5#2048+0.5 #1024 + 0.5#4096 + 0.5#1024 + 0.5

        rgb_f[0] = (img_front[v, u, 2] * v_d + img_front[v + 1, u, 2] * (1 - v_d)) * u_d + \
                             (img_front[v, u + 1, 2] * v_d + img_front[v + 1, u + 1, 2] * (1 - v_d)) * (1 - u_d)
        rgb_f[1] = (img_front[v, u, 1] * v_d + img_front[v + 1, u, 1] * (1 - v_d)) * u_d + \
                             (img_front[v, u + 1, 1] * v_d + img_front[v + 1, u + 1, 1] * (1 - v_d)) * (1 - u_d)
        rgb_f[2] = (img_front[v, u, 0] * v_d + img_front[v + 1, u, 0] * (1 - v_d)) * u_d + \
                             (img_front[v, u + 1, 0] * v_d + img_front[v + 1, u + 1, 0] * (1 - v_d)) * (1 - u_d)

        if pred_normals[k, 0] == 0.0:
            pred_normals[k, :] = 0
        vts_uv[k, 0] = (v + v_d) / 1024 #1024 #512#2048#512
        vts_uv[k, 1] = (u + u_d) / 2048 #2048 #1024#4096#1024

        rgb_b[0] = (img_back[v, u, 2] * v_d + img_back[v + 1, u, 2] * (1 - v_d)) * u_d + \
                             (img_back[v, u + 1, 2] * v_d + img_back[v + 1, u + 1, 2] * (1 - v_d)) * (1 - u_d)
        rgb_b[1] = (img_back[v, u, 1] * v_d + img_back[v + 1, u, 1] * (1 - v_d)) * u_d + \
                             (img_back[v, u + 1, 1] * v_d + img_back[v + 1, u + 1, 1] * (1 - v_d)) * (1 - u_d)
        rgb_b[2] = (img_back[v, u, 0] * v_d + img_back[v + 1, u, 0] * (1 - v_d)) * u_d + \
                             (img_back[v, u + 1, 0] * v_d + img_back[v + 1, u + 1, 0] * (1 - v_d)) * (1 - u_d)

        if pred_normals[k, 0] < 0.0:
            if bw[v, u] > 0:
                model_colors[k, :] = rgb_f*(1 - bw[v, u]) + rgb_b*bw[v, u]
            else:
                model_colors[k, :] = rgb_f
        else:
            if bw[v, u] > 0:
                model_colors[k, :] = rgb_b*(1 - bw[v, u]) + rgb_f*bw[v, u]
            else:
                model_colors[k, :] = rgb_b

    normals = pred_normals
    if texture_map:
        texture_map = np.concatenate([img_front.astype(np.uint8), img_back.astype(np.uint8)], axis=0)
        texture_map = Image.fromarray(texture_map[:, :, ::-1])
        texture_visual = trimesh.visual.TextureVisuals(uv=vts_uv, image=texture_map)
        color_mesh = trimesh.Trimesh(vertices=vertices,
                                     vertex_colors=model_colors,
                                     vertex_normals=normals,
                                     faces=faces,
                                     visual=texture_visual,
                                     process=True,
                                     maintain_order=False)
    else:
        color_mesh = trimesh.Trimesh(vertices=vertices,
                                     vertex_colors=model_colors / 255.,
                                     vertex_normals=normals,
                                     faces=faces,
                                     process=True,
                                     maintain_order=False)

    return color_mesh


if __name__ == '__main__':

    # dataset_path = 'I:/smplx_dataset/VOXEL/SMPLX_50/0_voxel.npy'
    # data = np.load(dataset_path)
    end = time.time ()
    # sdf_volume = depth2volume_float(data, voxel_size=256, z_level=256)
    # print ('%0.2f sec.\n' % (time.time() - end))
    # volume2mesh (np.squeeze (sdf_volume))

    # # depth_front = cv2.imread ('I:/smplx_dataset/DEPTH/SMPLX_50/0_front.png', cv2.IMREAD_GRAYSCALE)
    # # depth_back = cv2.imread ('I:/smplx_dataset/DEPTH_PRED/SMPLX_50/0_back.png', cv2.IMREAD_GRAYSCALE)
    depth_front = cv2.imread('E:/iois_dataset/EVAL/_order1_keticvl-work10_d2d/input/10.png', cv2.IMREAD_GRAYSCALE)
    depth_back = cv2.imread('E:/iois_dataset/EVAL/_order1_keticvl-work10_d2d/pred/10.png', cv2.IMREAD_GRAYSCALE)
    # cv2.imshow ('hi', depth_front)
    # cv2.waitKey (0)
    #
    depth_front = depth_front.astype(np.float)
    depth_back = depth_back.astype(np.float)

    end = time.time()
    for k in range(30):
        sdf_volume = depth2occ_double(depth_front, depth_back, voxel_size=256)
    print('%0.2f sec.\n' % (time.time() - end))
    volume2colormesh (np.squeeze(sdf_volume), level=0.5)

    # # print(np.max(depth_front))
    # # print(np.max(depth_back))
    # # depth_front[depth_front == 0] = 255
    # # print (np.min (depth_front))
    # #
    # input_stacked = np.stack ((depth_front, depth_back), axis=2)
    # sdf_volume = depth2volume_float(input_stacked, voxel_size=256, z_level=256)
    # volume2mesh (np.squeeze (sdf_volume))
    # cv2.imshow('hi', depth_front)
    # cv2.waitKey(0)

    # depth_front = cv2.imread ('F:/NIA(2020)/201012/201012_Inho1_results/Depth/temp.png', cv2.IMREAD_GRAYSCALE)
    # depth_map = depth_front.astype (np.float)
    # # detph_map = cv2.resize(depth_map, (256, 256))
    # sdf_volume = depth2volume_single (depth_map, voxel_size=256, z_level=256)
    # volume2mesh (np.squeeze (sdf_volume))

    # depth_front = cv2.imread ('I:/smplx_dataset/DEPTH/SMPLX_50/0_front.png', cv2.IMREAD_GRAYSCALE)
    # sdf_volume = depth2volume_lstm (depth_front, voxel_size=256, z_level=128)
    # volume2mesh (np.squeeze (sdf_volume))
    # print('hi')
    # end = time.time()
    # out = F.interpolate (torch.Tensor(np.expand_dims(sdf_volume, axis=0)), (256, 256), mode='bilinear',
    #                      align_corners=True)
    # new_sdf = out.numpy()
    # print ('%0.2f sec.\n' % (time.time() - end))



