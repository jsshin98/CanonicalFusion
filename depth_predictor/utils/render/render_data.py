
import numpy as np
import os
import cv2
import argparse
import trimesh
import torch
from mesh_to_sdf import mesh_to_voxels
# import open3d as o3d
# import point_cloud_utils as pcu
# import skimage
import PIL.Image as Image


# opt = BaseOptions().parse()
# cuda = torch.device('cuda:%d' % opt.gpu_id)
os.environ['PYOPENGL_PLATFORM'] = 'egl'


def create_mesh_volume_64(mesh):
    voxel_size = 64
    height = 512
    width = 512
    voxels = mesh_to_voxels(mesh, voxel_size, pad=False)
    shape2D = voxels.reshape(height, width)

    return shape2D


def create_mesh_volume_32(mesh):
    voxel_size = 32
    height = 64
    width = 512
    voxels = mesh_to_voxels(mesh, voxel_size, pad=False)
    shape2D = voxels.reshape(height, width)

    return shape2D


def create_depth_map(mesh, scene, res, fov):
    # any of the automatically generated values can be overridden
    # set resolution, in pixels
    scene.camera.resolution = [res, res]
    # set field of view, in degrees
    # make it relative to resolution so pixels per degree is same
    scene.camera.fov = fov * (scene.camera.resolution /
                             scene.camera.resolution.max())

    # convert the camera to rays with one ray per pixel
    origins, vectors, pixels = scene.camera_rays()

    # do the actual ray- mesh queries
    points, index_ray, index_tri = mesh.ray.intersects_location(
        origins, vectors, multiple_hits=False)

    # for each hit, find the distance along its vector
    depth = trimesh.util.diagonal_dot(points - origins[0],
                                      vectors[index_ray])
    # find pixel locations of actual hits
    pixel_ray = pixels[index_ray]

    # create a numpy array we can turn into an image
    # doing it with uint8 creates an `L` mode greyscale image
    a = np.zeros(scene.camera.resolution, dtype=np.uint8)

    # scale depth against range (0.0 - 1.0)
    depth_float = ((depth - depth.min()) / depth.ptp())

    # convert depth into 0 - 255 uint8
    depth_int = (depth_float * 255).round().astype(np.uint8)
    # assign depth to correct pixel locations
    a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_int
    # create a PIL image from the depth queries
    img = Image.fromarray(a)
    img_rotated = img.rotate(90)

    # show the resulting image
    #img_rotated.show()
    #mesh.show()

    return img_rotated


def create_silhouette(mesh, scene, res, fov):
    camera = scene.camera
    camera.fov = (fov,) * 2
    camera.resolution = (res,) * 2

    origins, rays, px = scene.camera_rays()
    origin = origins[0]
    rays = rays.reshape((res, res, 3))
    offset = mesh.vertices - origin

    # dists is vertices projected onto central ray
    dists = np.dot(offset, rays[rays.shape[0] // 2, rays.shape[1] // 2])
    closest = np.min(dists)
    farthest = np.max(dists)
    z = np.linspace(closest, farthest, res)
    print('z range: %f, %f' % (closest, farthest))

    vox = mesh.voxelized(1. / res)

    coords = np.expand_dims(rays, axis=-2) * np.expand_dims(z, axis=-1)
    coords += origin
    frust_vox_dense = vox.is_filled(coords)
    sil = np.any(frust_vox_dense, axis=-1)
    sil = sil.T  # change to image ordering (y, x)
    #sil.show()

    return sil


import open3d as o3d
import copy
import numpy as np

def rotate_mesh():
    mesh = o3d.io.read_point_cloud('')
    mesh_r = copy.deepcopy(mesh)
    R=mesh.get_rotation_matrix_from_xyz((0, np.pi/2, 0))
    mesh_r.rotate(R, center=(0,0,0))
    o3d.visualization.draw_geometries([mesh_r])
    # mesh.get_center will give you the center position
    return mesh_r


def create_normals_map(mesh, scene, fov):
    # set resolution, in pixels
    scene.camera.resolution = [512, 512]
    # set field of view, in degrees
    # make it relative to resolution so pixels per degree is same
    scene.camera.fov = fov * (scene.camera.resolution /
                             scene.camera.resolution.max())

    # convert the camera to rays with one ray per pixel
    origins, vectors, pixels = scene.camera_rays()

    # do the actual ray- mesh queries
    points, index_ray, index_tri = mesh.ray.intersects_location(
        origins, vectors, multiple_hits=False)

    # for each hit, find the distance along its vector
    depth = trimesh.util.diagonal_dot(points - origins[0],
                                      vectors[index_ray])
    # find pixel locations of actual hits
    pixel_ray = pixels[index_ray]

    # create a numpy array we can turn into an image
    # doing it with uint8 creates an `L` mode greyscale image
    a = np.zeros(scene.camera.resolution, dtype=np.uint8)

    # scale depth against range (0.0 - 1.0)
    depth_float = ((depth - depth.min()) / depth.ptp())

    # convert depth into 0 - 255 uint8
    depth_int = (depth_float * 255).round().astype(np.uint8)
    # assign depth to correct pixel locations
    a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_int
    # create a PIL image from the depth queries
    img = Image.fromarray(a)
    img_rotated = np.array(img.rotate(90))

    zy, zx = np.gradient(np.array(img_rotated))
    normal = np.dstack((-zx, -zy, np.ones_like(np.array(img_rotated))))

    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    # offset and rescale values to be in 0-255
    normal += 1
    normal /= 2
    normal *= 255

    return normal

def generate_training_dataset(out_path, folders):
    f_pcd = open(os.path.join(out_path, 'GEO', 'OBJ') + "/list_pcd.txt", 'w')
    f_volume_64 = open(os.path.join(out_path, 'MESH_VOLUME_64') + "/list_volume_64.txt", 'w')
    f_volume_32 = open(os.path.join(out_path, 'MESH_VOLUME_32') + "/list_volume_32.txt", 'w')
    f_depth = open(os.path.join(out_path, 'DEPTH') + "/list_depth.txt", 'w')
    f_mask = open(os.path.join(out_path, 'MASK') + "/list_mask.txt", 'w')
    f_normals = open(os.path.join(out_path, 'NORMALS') + "/list_normals.txt", 'w')

    for fd in folders:
        # sub_folders = sorted(os.listdir(os.path.join(args.input, f)))
        f = fd.split('/')[-1][:-4]#숫자만(000)
        # for subject_name in sub_folders:
        # folder_name = os.path.join(args.input, f)

        # os.makedirs(os.path.join(out_path, 'GEO', 'OBJ'), exist_ok=True)
        # # #os.makedirs(os.path.join(out_path, 'MESH_VOLUME_128', subject_name), exist_ok=True)
        # os.makedirs(os.path.join(out_path, 'MESH_VOLUME_64'), exist_ok=True)
        # os.makedirs(os.path.join(out_path, 'MESH_VOLUME_32'), exist_ok=True)
        # os.makedirs(os.path.join(out_path, 'DEPTH'), exist_ok=True)
        # # os.makedirs(os.path.join(out_path, 'DEPTH_VOLUME', subject_name), exist_ok=True)
        # os.makedirs(os.path.join(out_path, 'NORMALS'), exist_ok=True)
        # os.makedirs(os.path.join(out_path, 'MASK'), exist_ok=True)

        # set path for obj
        # mesh_file = os.path.join(folder_name, '{}.ply'.format(subject_name))
        mesh_file = os.path.join(args.input, '{}.ply'.format(f))
        if not os.path.exists(mesh_file):
            print('ERROR: obj file does not exist!!', mesh_file)
            return

        # create pcd_volume(tsdf)
        mesh = trimesh.load(mesh_file)

        # create_mesh_volume_64
        volume_2d = create_mesh_volume_64(mesh)
        mesh_volume_64 = Image.fromarray(volume_2d.astype('float32'))
        volume_file = os.path.join(out_path, 'MESH_VOLUME_64', '{}.tiff'.format(f))
        mesh_volume_64.save(volume_file)
        f_volume_64.write(os.path.join(out_path, 'MESH_VOLUME_64') + '/{}.tiff\n'.format(f))

        # create_mesh_volume_32
        volume_2d = create_mesh_volume_32(mesh)
        mesh_volume_32 = Image.fromarray(volume_2d.astype('float32'))
        volume_file = os.path.join(out_path, 'MESH_VOLUME_32', '{}.tiff'.format(f))
        mesh_volume_32.save(volume_file)
        f_volume_32.write(os.path.join(out_path, 'MESH_VOLUME_32') + '/{}.tiff\n'.format(f))

        # gt = np.array(mesh_volume)
        # volume3D = gt.reshape(64, 64, 64)
        # mesh_re = volume3D.extract_triangle_mesh()
        # mesh_re.compute_vertex_normals()
        # vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(volume3D, level=0)
        # volume_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        # volume_mesh.show()

        # scene will have automatically generated camera and lights
        scene = mesh.scene()
        scene.camera.resolution = [512, 512]

        # copy obj file
        cmd = 'cp %s %s' % (mesh_file, os.path.join(out_path, 'GEO', 'OBJ'))
        f_pcd.write(os.path.join(out_path, 'GEO', 'OBJ') + '/{}.ply\n'.format(f))
        print(cmd)
        os.system(cmd)

        # # v is a nv by 3 NumPy array of vertices
        # v, _, _, _ = pcu.read_ply(mesh_file)
        # # Estimate a normal at each point (row of v) using its 16 nearest neighbors
        # n = pcu.estimate_normals(v, k=16)
        # n.show()
        # normals = Image.fromarray(n.astype('float32'))
        # normals_file = os.path.join(out_path, 'NORMALS', '{}.png'.format(f))
        # normals.save(normals_file)
        # f_normals.write(os.path.join(out_path, 'NORMALS') + '/{}.png\n'.format(f))

        # create depth_map
        res = 512
        fov = 45.0
        depth = create_depth_map(mesh, scene, res, fov)
        depth_file = os.path.join(out_path, 'DEPTH', '{}.png'.format(f))
        depth.save(depth_file)
        f_depth.write(os.path.join(out_path, 'DEPTH') + '/{}.png\n'.format(f))

        # create normals_map
        normals = create_normals_map(mesh, scene, fov)
        cv2.imwrite(os.path.join(out_path, 'NORMALS', '{}.png'.format(f)), normals[:, :, ::-1])
        f_normals.write(os.path.join(out_path, 'NORMALS') + '/{}.png\n'.format(f))

        # create silhouette
        sil = create_silhouette(mesh, scene, res, fov)
        mask_file = (os.path.join(out_path, 'MASK', '{}.png'.format(f)))
        Image.fromarray(sil).convert("RGB").save(mask_file)
        f_mask.write(os.path.join(out_path, 'MASK') + '/{}.png\n'.format(f))

    f_pcd.close()
    f_volume_64.close()
    f_volume_32.close()
    f_depth.close()
    f_mask.close()
    f_normals.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='')
    parser.add_argument('-o', '--out_dir', type=str, default='')
    args = parser.parse_args()

    folders = sorted(os.listdir(args.input))
    generate_training_dataset(args.out_dir, folders)