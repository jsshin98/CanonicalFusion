import cv2
import numpy as np
import trimesh


def grid_linspace(bounds, count):
    """
    Return a grid spaced inside a bounding box with edges spaced using np.linspace.

    Parameters
    ------------
    bounds: (2,dimension) list of [[min x, min y, etc], [max x, max y, etc]]
    count:  int, or (dimension,) int, number of samples per side

    Returns
    ---------
    grid: (n, dimension) float, points in the specified bounds
    """
    bounds = np.asanyarray(bounds, dtype=np.float64)
    if len(bounds) != 2:
        raise ValueError('bounds must be (2, dimension!')

    count = np.asanyarray(count, dtype=np.int)
    if count.shape == ():
        count = np.tile(count, bounds.shape[1])

    grid_elements = [np.linspace(*b, num=c) for b, c in zip(bounds.T, count)]
    grid = np.vstack(np.meshgrid(*grid_elements, indexing='ij')
                     ).reshape(bounds.shape[1], -1).T
    return grid


def get_camera_rays(camera, dir):
    res = camera.resolution[0]
    v = np.tan(np.radians(camera.fov[0]) / 2.0)
    # v *= 1 - (1/res)

    # create a grid of vectors
    if dir == 'front':
        xy = grid_linspace (
            bounds=[[-v, -v], [v, v]],
            count=res)
        pixels = grid_linspace (
            bounds=[[0, res - 1], [res - 1, 0]],
            count=res).astype (np.int64)
        vectors = np.column_stack ((np.zeros_like (xy[:, :]), -np.ones_like (xy[:, :1])))
        origins = np.column_stack ((xy, np.ones_like (xy[:, :1])))

    elif dir == 'side':
        yz = grid_linspace (
            bounds=[[-v, v], [v, -v]],
            count=res)
        pixels = grid_linspace(
            bounds=[[0, res - 1], [res - 1, 0]],
            count=res).astype (np.int64)
        vectors = np.column_stack ((np.ones_like (yz[:, :1]), np.zeros_like (yz[:, :])))
        origins = np.column_stack ((-np.ones_like (yz[:, :1]), yz))

    elif dir == 'up':  # will be updated.
        xz = grid_linspace (
            bounds=[[-v, -v], [v, v]],
            count=res)
        pixels = grid_linspace (
            bounds=[[0, res - 1], [0, res - 1]],
            count=res).astype (np.int64)
        vectors = np.column_stack ((np.zeros_like (xz[:, :1]),
                                    np.ones_like (xz[:, :1]),
                                    np.zeros_like (xz[:, :1])))
        origins = np.column_stack ((xz[:, 0], -np.ones_like (xz[:, :1]), xz[:, 1]))

    return origins, vectors, pixels


def get_depth_maps(mesh, scene, res, fov, dir):
    scene.camera.resolution = [res, res]
    scene.camera.fov = fov * (scene.camera.resolution /
                              scene.camera.resolution.max())

    origins, vectors, pixels = get_camera_rays(scene.camera, dir)

    points, index_ray, index_tri = mesh.ray.intersects_location(
        origins, vectors, multiple_hits=True)
    depth = trimesh.util.diagonal_dot(points - origins[index_ray], vectors[index_ray])

    colors = mesh.visual.face_colors[index_tri]
    pixel_ray = pixels[index_ray]
    depth_far = np.zeros(scene.camera.resolution, dtype=np.float)
    depth_near = np.ones(scene.camera.resolution, dtype=np.float) * res
    color_far = np.zeros((res, res, 3), dtype=np.float)
    color_near = np.zeros((res, res, 3), dtype=np.float)

    denom = np.tan(np.radians(fov) / 2.0) * 2
    depth_int = (depth - 1)*(res/denom) + res/2

    for k in range(pixel_ray.shape[0]):
        u, v = pixel_ray[k, 0], pixel_ray[k, 1]
        if depth_int[k] > depth_far[v, u]:
            color_far[v, u, ::-1] = colors[k, 0:3] / 255.0
            depth_far[v, u] = depth_int[k]
        if depth_int[k] < depth_near[v, u]:
            depth_near[v, u] = depth_int[k]
            color_near[v, u, ::-1] = colors[k, 0:3] / 255.0

    depth_near = depth_near * (depth_near != res)

    if dir == 'side':
        depth_near = np.rot90 (depth_near, k=1)
        depth_far = np.rot90 (depth_far, k=1)
        color_near = np.rot90 (color_near, k=1)
        color_far = np.rot90 (color_far, k=1)
    return depth_near, depth_far, color_near, color_far


def get_color_mesh(mesh, file, color_front, color_back, scene, res, fov, dir):
    scene.camera.resolution = [res/2, res]
    scene.camera.fov = fov * (scene.camera.resolution /
                              scene.camera.resolution.max())
    # origins, vectors, pixels = get_camera_rays(scene.camera, dir)
    origins, vectors, pixels = scene.camera_rays()
    # pts, ray = camera_rays(fov, width, height)

    points_f, index_ray_f, index_tri_f = mesh.ray.intersects_location(
        origins, vectors, multiple_hits=False)
    pixel_ray_f = pixels[index_ray_f]

    for k in range(pixel_ray_f.shape[0]):
        v = points_f[k]
        cf = color_front[pixel_ray_f[k, 0], pixel_ray_f[k, 1]]  #/ 255.0
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], cf[2], cf[1], cf[0]))

    points_b, index_ray_b, index_tri_b = mesh.ray.intersects_location(
        origins, vectors, multiple_hits=True)

    pixel_ray_b = pixels[index_ray_b]

    for k in range(pixel_ray_b.shape[0]):
        v = points_b[k]
        cb = color_back[pixel_ray_b[k, 0], pixel_ray_b[k, 1]]  #/ 255.0
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], cb[2], cb[1], cb[0]))


def get_color_mesh2(mesh, file, scene, res, fov, dir):
    scene.camera.resolution = [res, res]
    scene.camera.fov = fov * (scene.camera.resolution /
                              scene.camera.resolution.max())

    origins, vectors, pixels = get_camera_rays(scene.camera, dir)

    points_f, index_ray_f, index_tri_f = mesh.ray.intersects_location(
        origins, vectors, multiple_hits=False)

    points_b, index_ray_b, index_tri_b = mesh.ray.intersects_location(
        origins, vectors, multiple_hits=True)

    colors_f = mesh.visual.face_colors[index_tri_f]
    colors_b = mesh.visual.face_colors[index_tri_b]
    pixel_ray_f = pixels[index_ray_f]
    pixel_ray_b = pixels[index_ray_b]

    for k in range(pixel_ray_b.shape[0]):
        v = points_b[k]
        cb = colors_b[k, 0:3] / 255.0
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], cb[0], cb[1], cb[2]))

    for k in range(pixel_ray_f.shape[0]):
        v = points_f[k]
        cf = colors_f[k, 0:3] / 255.0
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], cf[0], cf[1], cf[2]))

    return file


if __name__ == '__main__':
    voxel_resolution = 256
    width = 256
    height = 256
    fov = 90
    cam_res = 256

    RGB_MAX = [255.0, 255.0, 255.0]
    DEPTH_MAX = 255.0

    mesh = trimesh.load ('D:\\DATASET\\OBJ\\IOIS_0\\iois_00000002.ply')
    color_front = cv2.imread ('D:\\DATASET\\COLOR_0\\IOIS_0\\iois_00000002_front.png', cv2.IMREAD_COLOR)
    color_back = cv2.imread ('D:\\DATASET\\COLOR_0\\IOIS_0\\iois_00000002_back.png', cv2.IMREAD_COLOR)
    depth_front = cv2.imread ('D:\\DATASET\\DEPTH_0\\IOIS_0\\iois_00000002_front.png', cv2.IMREAD_ANYDEPTH)
    depth_back = cv2.imread ('D:\\DATASET\\DEPTH_0\\IOIS_0\\iois_00000002_back.png', cv2.IMREAD_ANYDEPTH)

    # color_front = color_front / RGB_MAX
    # color_back = color_back / RGB_MAX
    # depth_front = depth_front.astype(np.float) / 128.0
    # depth_back = depth_back.astype(np.float) / 128.0
    # depth_front = depth_front / DEPTH_MAX
    # depth_back = depth_back / DEPTH_MAX
    #
    # volume = depth2occ_double(depth_front, depth_back)
    # mesh = volume2mesh(volume, visualize=False)

    scene = mesh.scene ()

    # origins, vectors, pixels = scene.camera_rays()
    # origins, vectors, pixels = get_camera_rays(scene.camera, 'front')
    # pts, ray = camera_rays(fov, width, height)

    color_front = np.rot90 (color_front, 3)
    color_back = np.rot90 (color_back, 3)

    file = open ('mesh.obj', 'w')

    get_color_mesh (mesh, file, color_front, color_back, scene, cam_res, fov, 'front')
