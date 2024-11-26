import numpy as np
import trimesh
import torch
from skimage import measure
import cv2
import torch.nn as nn
import os
# import open3d as o3d
import time
import math


def depth2pix(depth_map):
    b, _, h, w = depth_map.size()
    y_range = torch.arange(0, h).view(1, h, 1).expand(b, 1, h, w).type_as(depth_map)
    x_range = torch.arange(0, w).view(1, 1, w).expand(b, 1, h, w).type_as(depth_map)

    pixel_coords = torch.cat((x_range, y_range, depth_map), dim=1)
    pixel_coords_vec = pixel_coords.reshape(b, 3, -1)  # [b, 3, 1024, 1024]

    return pixel_coords_vec


def color2pix(color_map):
    b, _, h, w = color_map.size()
    pixel_coords_vec = color_map.reshape(b, 3, -1)

    return pixel_coords_vec


def pers2orth(pers_color, pers_depth, res, focal):
    fx = fy = focal
    cx = cy = (res / 2)
    pers = depth2pix(pers_depth).float()
    x = pers[:, 0, :]
    y = pers[:, 1, :]
    z = pers[:, 2, :]
    x = (x - cx) / fx * z
    y = (y - cy) / fy * z
    z_ = torch.sqrt(z * z + x * x + y * y)
    # z_ = z_ * res - cz
    # x_ = x * res + cx
    # y_ = y * res + cy

    xyz = torch.stack([pers[:, 0, :], pers[:, 1, :], z_], dim=1)
    xyz = xyz.long()

    img_foward = torch.zeros_like(pers_color)
    depth_foward = torch.zeros_like(pers_depth)
    for i in range(pers.shape[0]):
        orth_idx = xyz[i, :, :].long()
        pers_idx = pers[i, :, :].long()
        idx = (z_[i, :] > 0.1)
        img_foward[i, :, orth_idx[1, idx], orth_idx[0, idx]] = pers_color[i, :, pers_idx[1, idx], pers_idx[0, idx]]
        depth_foward[i, 0, orth_idx[1, idx], orth_idx[0, idx]] = z_[i, idx]

    img_foward = img_foward[0].permute(1, 2, 0).detach().cpu().numpy()
    img_foward = cv2.cvtColor(img_foward.astype(np.float32), cv2.COLOR_BGR2RGB)

    # depth_foward = depth_foward[0].permute(1, 2, 0).detach().cpu().numpy()
    return xyz, img_foward, depth_foward


# def pers2orth(pers_depth, res, focal):
#     fx = fy = focal
#     cx = cy = (res/2)
#     pers = depth2pix(pers_depth).float()
#     x = pers[:, 0, :]
#     y = pers[:, 1, :]
#     z = pers[:, 2, :]
#     x = (x - cx) / fx * z
#     y = (y - cy) / fy * z
#     z_ = torch.sqrt(z * z + x * x + y * y)
#     # z_ = z_ * res - cz
#     # x_ = x * res + cx
#     # y_ = y * res + cy
#
#     orth = torch.stack([pers[:, 0, :], pers[:, 1, :], z_], dim=1)
#     orth = orth.long()
#
#     return orth

def pers2pc(pers_color, pers_normal, pers_depth, res, fov):
    pers_depth = pers_depth.reshape(-1, 1)
    pers_depth = np.tile(pers_depth, reps=[1, 3])

    temp = trimesh.scene.Scene()
    temp.camera.fov = [fov, fov]
    temp.camera.resolution = [res, res]
    temp.camera_transform = np.eye(4)
    pers_origins, pers_vectors, pers_pixels = temp.camera_rays()  # [1048576,3]
    pers_vectors = np.rot90(pers_vectors.reshape(res, res, 3), 1).reshape(-1, 3)
    pers_vectors[:, 2] *= -1

    xyz = pers_depth * pers_vectors
    xyz = xyz[xyz[:, 2] > 0.0, :]
    # fx = fy = 532.37887551
    # cx = cy = (512/2)
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    # x = (x - cx) / fx * z
    # y = (y - cy) / fy * z
    z_ = np.sqrt(z * z + x * x + y * y)
    xyz_ = np.stack([x, y, z_], axis=1)

    if pers_color is not None:
        pers_color = torch.Tensor(pers_color).permute(2, 0, 1).unsqueeze(0)
        pers_color = color2pix(pers_color).squeeze()
        pers_color = torch.masked_select(pers_color, torch.from_numpy(pers_depth[:, 2]) > 0.0).view(3, -1).permute(1,
                                                                                                                   0).numpy()
    if pers_normal is not None:
        pers_normal = torch.Tensor(pers_normal).permute(2, 0, 1).unsqueeze(0)
        pers_normal = color2pix(pers_normal).squeeze()
        pers_normal = torch.masked_select(pers_normal, torch.from_numpy(pers_depth[:, 2]) > 0.0).view(3, -1).permute(1,
                                                                                                                     0).numpy()

    return xyz_, pers_color, pers_normal


def pers_depth2Z(pers_depth, res, fov):
    # start_time = time.time()

    focal = res / (2 * np.tan(np.radians(fov) / 2.0))
    # fx = fy = focal
    # cx = cy = (res / 2)

    pers_depth[pers_depth > 0] = (pers_depth[pers_depth > 0] - res / 2) / focal + 1
    pers_depth = pers_depth.reshape(-1, 1)
    pers_depth = np.tile(pers_depth, reps=[1, 3])
    # print(pers_depth[:,2].max())

    temp = trimesh.scene.Scene()
    temp.camera.resolution = [res, res]
    temp.camera.fov = [fov, fov]
    # temp.camera_transform[1, 1] = -1.0
    # temp.camera_transform[2, 2] = -1.0
    # temp.camera_transform[0:3, 3] = 0.0
    # temp.camera_transform[2, 3] = 1.0
    pers_origins, pers_vectors, pers_pixels = temp.camera_rays()  # [1048576,3]
    pers_vectors = np.rot90(pers_vectors.reshape(res, res, 3), 1).reshape(-1, 3)
    pers_vectors[:, 2] *= -1
    # pers_vectors = np.rot90(pers_vectors.reshape(res,res,3), 1).reshape(-1,3)
    # pers_pixels_ = np.rot90(pers_pixels.reshape(res,res,2), 1).reshape(-1,2)

    # R = trimesh.transformations.rotation_matrix(math.pi * 180/180, np.array([1, 0, 0]))[:3, :3]

    # pers_depth[pers_depth>0].min()

    # print(pers_depth.shape)
    # print(pers_vectors.shape)

    xyz = pers_depth * pers_vectors
    map = xyz[:, 2].reshape(res, res)
    map[map > 0] = (map[map > 0] - 1) * focal + res / 2
    # print(map.max())

    return map


def pers_depth2Z_temp(pers_depth, res, fov):
    start_time = time.time()

    focal = res / (2 * np.tan(np.radians(fov) / 2.0))
    # fx = fy = focal
    # cx = cy = (res / 2)

    pers_depth = torch.Tensor(pers_depth).unsqueeze(0).unsqueeze(1)
    pers = depth2pix(pers_depth).float()
    x = pers[:, 0, :]
    y = pers[:, 1, :]
    z = pers[:, 2, :]  # [1, 1048576]

    idx = np.where(z > 0)[1].T  # [1, 1048576] ok
    map = np.zeros_like(z)

    x = x[z > 0]
    y = y[z > 0]
    z = z[z > 0]

    z[z > 0] = (z[z > 0] * res - res / 2) / res + 1.0
    x = (x - cx) / fx * z
    y = (y - cy) / fy * z
    z = torch.sqrt(z * z - x * x - y * y) - 0.5  # ok 일단 이 값으로 marching cube

    map[:, idx] = z * 128.0 * 512.0
    map = np.reshape(map, (1024, 1024))
    map = map.astype(np.uint16)

    print("%.6s seconds for backward\n" % (time.time() - start_time))

    # if pers_color is not None:
    #     pc = trimesh.points.PointCloud(vertices=xyz, colors=pers_color)
    # else:
    #     pc = trimesh.points.PointCloud(vertices=xyz)

    return map


def Z2pc(pers_color, pers_Z, res, fov):
    # start_time = time.time()

    focal = res / (2 * np.tan(np.radians(fov) / 2.0))
    # fx = fy = focal
    # cx = cy = (res / 2)

    temp = trimesh.scene.Scene()
    temp.camera.resolution = [res, res]
    temp.camera.fov = [fov, fov]
    # temp.camera_transform[1, 1] = -1.0
    # temp.camera_transform[2, 2] = -1.0
    # temp.camera_transform[0:3, 3] = 0.0
    # temp.camera_transform[2, 3] = 1.0
    pers_origins, pers_vectors, pers_pixels = temp.camera_rays()  # [1048576,3]
    pers_vectors = np.rot90(pers_vectors.reshape(res, res, 3), 1).reshape(-1, 3)
    pers_vectors[:, 2] *= -1
    pers_vectors = pers_vectors.reshape(res, res, 3)

    xyz = pers_vectors / np.expand_dims(pers_vectors[:, :, 2], axis=2) * np.expand_dims(pers_Z, axis=2)
    xyz = xyz.reshape(-1, 3)
    xyz = xyz[xyz[:, 2] > 0.3, :]

    if pers_color is not None:
        pers_color = torch.Tensor(pers_color).permute(2, 0, 1).unsqueeze(0)
        pers_color = color2pix(pers_color).squeeze()
        pers_color = torch.masked_select(pers_color, torch.from_numpy(pers_Z.reshape(-1, 1)[:, 0]) > 0.3).view(3,
                                                                                                               -1).permute(
            1, 0).numpy()

    return xyz, pers_color


def Z2pc_temp(pers_color, pers_Z, res, fov):
    # start_time = time.time()

    focal = res / (2 * np.tan(np.radians(fov) / 2.0))
    # fx = fy = focal
    # cx = cy = (res / 2)

    pers_Z[pers_Z > 0] = (pers_Z[pers_Z > 0] - res / 2) / focal + 1
    pers_Z = pers_Z.reshape(-1, 1)
    pers_Z = np.tile(pers_Z, reps=[1, 3])

    pers_Z = torch.Tensor(pers_Z).unsqueeze(0).unsqueeze(1)
    pers = depth2pix(pers_Z).float()
    x = pers[:, 0, :]
    y = pers[:, 1, :]
    z = pers[:, 2, :]  # [1, 1048576]

    if pers_color is not None:
        pers_color = torch.Tensor(pers_color).permute(2, 0, 1).unsqueeze(0)
        pers_color = color2pix(pers_color).squeeze()
        pers_color = torch.masked_select(pers_color, z > 0.2).view(3, -1).permute(1, 0).numpy()

    x = x[z > 0.2]
    y = y[z > 0.2]
    z = z[z > 0.2]

    z[z > 0.2] = (z[z > 0.2] * res - res / 2) / res + 1.0
    depth = torch.sqrt(z * z / (1 - ((x - cx) / fx) ** 2 - ((y - cy) / fy) ** 2))
    x = (x - cx) / fx * depth
    y = (y - cy) / fy * depth
    xyz = torch.stack((x, y, z), dim=1).numpy()

    # print("%.6s seconds for Z\n" % (time.time() - start_time))

    # if pers_color is not None:
    #     pc = trimesh.points.PointCloud(vertices=xyz, colors=pers_color)
    # else:
    #     pc = trimesh.points.PointCloud(vertices=xyz)

    return xyz, pers_color


if __name__ == '__main__':

    if not os.path.exists('/workspace/code/test/' + 'ply'):
        os.makedirs('/workspace/code/test/' + 'ply')
    if not os.path.exists('/workspace/code/test/' + 'z'):
        os.makedirs('/workspace/code/test/' + 'z')
    if not os.path.exists('/workspace/code/test/' + 'ply_Z'):
        os.makedirs('/workspace/code/test/' + 'ply_Z')

    # img, Z -> pc
    for i in [3, 4, 5]:  # range(12):
        # Z_front = cv2.imread('/workspace/code/test/DATA_1024/target/z/{i:05d}_gt_front.png'.format(i=i), cv2.IMREAD_ANYDEPTH).astype(np.float64)
        # Z_back  = cv2.imread('/workspace/code/test/DATA_1024/target/z/{i:05d}_gt_back.png'.format(i=i) , cv2.IMREAD_ANYDEPTH).astype(np.float64)
        # image_front = cv2.imread('/workspace/code/test/DATA_1024/input/color_1024_noshaded/{i:05d}.png'.format(i=i), cv2.IMREAD_ANYCOLOR)
        # image_back  = cv2.imread('/workspace/code/test/DATA_1024/input/color_1024_shaded/{i:05d}_back.png'.format(i=i), cv2.IMREAD_ANYCOLOR)

        Z_front = cv2.imread('/workspace/dataset/DATA_1024/PERS/Z/IOYS0_-10_x/iois_{i:08d}_front.png'.format(i=i),
                             cv2.IMREAD_ANYDEPTH).astype(np.float64)
        Z_back = cv2.imread('/workspace/dataset/DATA_1024/PERS/Z/IOYS0_-10_x/iois_{i:08d}_back.png'.format(i=i),
                            cv2.IMREAD_ANYDEPTH).astype(np.float64)
        image_front = cv2.imread(
            '/workspace/dataset/DATA_1024/PERS/COLOR/NOSHADING/IOYS0_-10_x/iois_{i:08d}_front.png'.format(i=i),
            cv2.IMREAD_ANYCOLOR)
        image_back = cv2.imread(
            '/workspace/dataset/DATA_1024/PERS/COLOR/NOSHADING/IOYS0_-10_x/iois_{i:08d}_back.png'.format(i=i),
            cv2.IMREAD_ANYCOLOR)

        image_front = cv2.cvtColor(image_front, cv2.COLOR_BGR2RGB)
        image_back = cv2.cvtColor(image_back, cv2.COLOR_BGR2RGB)

        Z_front = Z_front / 128.0 / 512.0
        Z_back = Z_back / 128.0 / 512.0

        # xyz_f, rgb_f = Z2pc(image_front, Z_front, 1024, 50)
        pc = Z2pc(image_front, Z_front, 1024, 50)
        pc.export('/workspace/code/test/ply_Z/{i:05d}_pd_front.ply'.format(i=i))
        pc = Z2pc(image_back, Z_back, 1024, 50)
        pc.export('/workspace/code/test/ply_Z/{i:05d}_pd_back.ply'.format(i=i))
        print(i)

    # pers depth -> Z
    for i in range(12):
        depth_front = cv2.imread('/workspace/code/test/DATA_1024/target/depth_1024_front/{i:05d}.png'.format(i=i),
                                 cv2.IMREAD_ANYDEPTH).astype(np.float64)
        depth_back = cv2.imread('/workspace/code/test/DATA_1024/target/depth_1024_back/{i:05d}.png'.format(i=i),
                                cv2.IMREAD_ANYDEPTH).astype(np.float64)

        # image_front = cv2.imread('/workspace/dataset/DATA_1024/PERS/COLOR/NOSHADING/IOYS0_0_x/iois_00000004_front.png', cv2.IMREAD_ANYCOLOR)
        # image_back  = cv2.imread('/workspace/dataset/DATA_1024/PERS/COLOR/NOSHADING/IOYS0_0_x/iois_00000004_back.png' , cv2.IMREAD_ANYCOLOR)
        # depth_front = cv2.imread('/workspace/dataset/DATA_1024/PERS/DEPTH/IOYS0_0_x/iois_00000004_front.png', cv2.IMREAD_ANYDEPTH).astype(np.float64)
        # depth_back  = cv2.imread('/workspace/dataset/DATA_1024/PERS/DEPTH/IOYS0_0_x/iois_00000004_back.png' , cv2.IMREAD_ANYDEPTH).astype(np.float64)

        depth_front = depth_front / 128.0 / 512.0
        depth_back = depth_back / 128.0 / 512.0

        d_f = pers_depth2Z(depth_front, 1024, 50)
        cv2.imwrite('/workspace/code/test/z/{i:05d}_gt_front.png'.format(i=i), d_f)
        d_b = pers_depth2Z(depth_back, 1024, 50)
        cv2.imwrite('/workspace/code/test/z/{i:05d}_gt_back.png'.format(i=i), d_b)

        # img, depth -> mesh
    for i in range(12):
        image_front = cv2.imread('/workspace/code/test/DATA_1024/input/color_1024_noshaded/{i:05d}.png'.format(i=i),
                                 cv2.IMREAD_ANYCOLOR)
        image_back = cv2.imread('/workspace/code/test/DATA_1024/input/color_1024_shaded/{i:05d}_back.png'.format(i=i),
                                cv2.IMREAD_ANYCOLOR)
        depth_front = cv2.imread('/workspace/code/test/DATA_1024/pred/depth_1024_front/{i:05d}.png'.format(i=i),
                                 cv2.IMREAD_ANYDEPTH).astype(np.float64)
        depth_back = cv2.imread('/workspace/code/test/DATA_1024/pred/depth_1024_back/{i:05d}.png'.format(i=i),
                                cv2.IMREAD_ANYDEPTH).astype(np.float64)

        # image_front = cv2.imread('/workspace/dataset/DATA_1024/PERS/COLOR/NOSHADING/IOYS0_0_x/iois_00000004_front.png', cv2.IMREAD_ANYCOLOR)
        # image_back  = cv2.imread('/workspace/dataset/DATA_1024/PERS/COLOR/NOSHADING/IOYS0_0_x/iois_00000004_back.png' , cv2.IMREAD_ANYCOLOR)
        # depth_front = cv2.imread('/workspace/dataset/DATA_1024/PERS/DEPTH/IOYS0_0_x/iois_00000004_front.png', cv2.IMREAD_ANYDEPTH).astype(np.float64)
        # depth_back  = cv2.imread('/workspace/dataset/DATA_1024/PERS/DEPTH/IOYS0_0_x/iois_00000004_back.png' , cv2.IMREAD_ANYDEPTH).astype(np.float64)

        image_front = cv2.cvtColor(image_front, cv2.COLOR_BGR2RGB)
        image_back = cv2.cvtColor(image_back, cv2.COLOR_BGR2RGB)
        depth_front = depth_front / 128.0 / 512.0
        depth_back = depth_back / 128.0 / 512.0

        xyz_f, rgb_f = pers2pc(image_front, depth_front, 1024, 50)
        # pc.export('/workspace/code/test/ply/{i:05d}_pd_front.ply'.format(i=i))
        xyz_b, rgb_b = pers2pc(image_back, depth_back, 1024, 50)
        # pc.export('/workspace/code/test/ply/{i:05d}_pd_back.ply'.format(i=i))

        xyz = np.concatenate((xyz_f, xyz_b), axis=0)
        rgb = np.concatenate((rgb_f, rgb_b), axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.estimate_normals()

        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 1.5 * avg_dist

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector([radius, radius * 2]))

        tri_mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), face=np.asarray(mesh.triangles),
                                   vertex_normals=np.asarray(mesh.vertex_normals))  # , vertex_colors=rgb)

        is_convex = trimesh.convex.is_convex(tri_mesh)
        print(is_convex)
        tri_mesh.export('/workspace/code/test/ply/{i:05d}_gt_front_mesh.ply'.format(i=i))

    # img, depth -> pc
    for i in range(12):
        image_front = cv2.imread('/workspace/code/test/DATA_1024/input/color_1024_noshaded/{i:05d}.png'.format(i=i),
                                 cv2.IMREAD_ANYCOLOR)
        image_back = cv2.imread('/workspace/code/test/DATA_1024/input/color_1024_shaded/{i:05d}_back.png'.format(i=i),
                                cv2.IMREAD_ANYCOLOR)
        depth_front = cv2.imread('/workspace/code/test/DATA_1024/target/depth_1024_front/{i:05d}.png'.format(i=i),
                                 cv2.IMREAD_ANYDEPTH).astype(np.float64)
        depth_back = cv2.imread('/workspace/code/test/DATA_1024/target/depth_1024_back/{i:05d}.png'.format(i=i),
                                cv2.IMREAD_ANYDEPTH).astype(np.float64)

        image_front = cv2.cvtColor(image_front, cv2.COLOR_BGR2RGBA)
        image_back = cv2.cvtColor(image_back, cv2.COLOR_BGR2RGBA)
        depth_front = depth_front / 128.0 / 512.0
        depth_back = depth_back / 128.0 / 512.0

        pc = pers2pc(image_front, depth_front, 1024, 50)
        pc.export('/workspace/code/test/ply/{i:05d}_gt_front.ply'.format(i=i))
        pc = pers2pc(image_back, depth_back, 1024, 50)
        pc.export('/workspace/code/test/ply/{i:05d}_gt_back.ply'.format(i=i))

    print('')

    ###########################################################
    # normal_front = depth2normal(depth_front)
    # normal_back = depth2normal(255 - depth_back)

    # # cv2.imshow ("normal_front", normal_front / 255)
    # # cv2.imshow ("normal_back", normal_back / 255)
    # # cv2.waitKey (0)

    # #
    # depth_front_torch = torch.Tensor(depth_front).unsqueeze(0)
    # depth_back_torch = torch.Tensor(depth_back).unsqueeze(0)
    # color_front_torch = torch.Tensor (image_front).unsqueeze(0)
    # color_back_torch = torch.Tensor (image_back).unsqueeze(0)
    # #
    # depth_left_pred_f, depth_right_pred_f, color_left_pred_f, color_right_pred_f \
    #     = warp2side(depth_front_torch, color=color_front_torch, normal=normal_front)
    # depth_left_pred_b, depth_right_pred_b, color_left_pred_b, color_right_pred_b \
    #     = warp2side(depth_back_torch, color=color_back_torch, normal=normal_back)
    # #
    # depth_left_pred = torch.min(depth_left_pred_f, depth_left_pred_b)
    # depth_left_pred[depth_left_pred == 255] = 0
    # depth_right_pred = torch.max (depth_right_pred_f, depth_right_pred_b)

    # depth_left_pred = depth_left_pred.squeeze(0).detach().cpu().numpy()
    # depth_right_pred = depth_right_pred.squeeze (0).detach ().cpu ().numpy ()

    # color_left_pred = torch.max (color_right_pred_f, color_right_pred_b)
    # color_left_pred = color_left_pred.squeeze(0).detach().cpu().numpy()

    # cv2.imshow ("pred_color", color_left_pred)
    # cv2.imshow("left_color", image_left)

    # left_error = (depth_left_pred - depth_left)
    # left_error[depth_left_pred == 0] = 0
    # right_error = (depth_right_pred - depth_right)
    # right_error[depth_right_pred == 0] = 0
    # # cv2.imshow("left", (depth_left_pred)/255)
    # # cv2.imshow ("left_error", left_error / 255)
    # cv2.imshow ("right", depth_right_pred / 255)
    # cv2.imshow ("right_error", right_error / 255)
    # cv2.imshow("right_gt", depth_right /255)

    # cv2.waitKey(0)

    # # cv2.imshow ("normal_front", normal_front / 255)
    # # cv2.imshow ("normal_back", normal_back / 255)
    # # cv2.waitKey (0)

    # # # warp front depth map
    # # print(depth_front[0, 0])

    # # cv2.imwrite('normal.png', normal_back.astype(np.int))
    # # cv2.imwrite('depth_front.png', depth_front.astype(np.int))

    # # normal_front[normal_front < 128] = 0

    # # depth_front = depth_front.astype(np.float) / 128.0
    # # depth_back = depth_back.astype(np.float) / 128.0
    # # end = time.time()
    # # for k in range(30):
    # #     sdf_volume = depth2occ_double(depth_front, depth_back, voxel_size=256)
    # # print ('%0.2f sec.\n' % (time.time() - end))
    # # volume2mesh (np.squeeze(sdf_volume), level=0.5)


