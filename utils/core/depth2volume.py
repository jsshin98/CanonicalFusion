import numpy as np
import trimesh
import torch
from skimage import measure
import time
import cv2
import torch.nn.functional as F
import torch.nn


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

def depth2occ_2view_torch(depth_front, depth_back, voxel_size=512, slope=0.001, binarize=False, device=None):

    if depth_front.shape[1] == 1:
        depth_front = depth_front.squeeze(1)
    if depth_back.shape[1] == 1:
        depth_back = depth_back.squeeze(1)

    occ_grid = torch.ones((depth_front.shape[0], depth_front.shape[1], depth_front.shape[2], voxel_size))

    if device is not None:
        occ_grid = torch.autograd.Variable(occ_grid)
        occ_grid = occ_grid.to(device)

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
    sdf = np.rot90(sdf, k=1)
    sdf = np.flip(sdf, axis=0)
    sdf = np.flip(sdf, axis=1)
    sdf = np.flip(sdf, axis=2)

    vertices, faces, normals, _ = measure.marching_cubes(sdf, level=level)

    return vertices, faces, normals


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


def colorize_model(volume, img_front, img_back):
    if isinstance(volume, trimesh.Trimesh):
        pred_mesh = volume
    else:
        # volume = np.rot90(volume, k=1, axes=(1, 0))
        vertices, faces, normals = volume2meshinfo(volume, level=0.0)
        vertex_num = vertices.shape[0]
        pred_mesh = trimesh.Trimesh(vertices=vertices, faces=faces,
                                    process=False, maintain_order=True)

    pred_normals = trimesh.geometry.weighted_vertex_normals(vertex_num, faces,
                                                            pred_mesh.face_normals,
                                                            pred_mesh.face_angles,
                                                            use_loop=False)

    model_colors = np.zeros_like(pred_normals)
    img_front = np.flip(img_front, 0)
    img_back = np.flip(img_back, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_front2 = cv2.dilate(img_front, kernel, iterations=2)
    img_back2 = cv2.dilate(img_back, kernel, iterations=2)

    for i in range(img_front.shape[1]):
        for j in range(img_front.shape[0]):
            if img_front2[j, i, 0] > 0 and img_front[j, i, 0] < 0.05:
                img_front[j, i, :] = img_front2[j, i, :]
                img_back[j, i, :] = img_back2[j, i, :]

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
    # vertices[1] = -vertices[1]
    color_mesh = trimesh.Trimesh(vertices=vertices,
                                 vertex_colors=model_colors,
                                 faces=faces,
                                 process=False,
                                 maintain_order=True)
    return color_mesh, model_colors


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



