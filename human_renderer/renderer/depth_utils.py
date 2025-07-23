import numpy as np
import trimesh

def pers_get_depth_map(mesh, scene, res):
    mesh.scene = scene
    pers_origins, pers_vectors, pers_pixels = mesh.scene.camera_rays()
    pers_points, pers_index_ray, pers_index_tri = mesh.ray.intersects_location(pers_origins,
                                                                               pers_vectors,
                                                                               multiple_hits=False)
    # (A: pers_points)  ----------> (pers_origins[0] -> surface)
    # (B: pers_vectors) ->          (same vector with unit norm)
    # A dot B = distance (cos(theta)|A||B| = cos(theta) = 1, A = depth * B)
    pers_depth = trimesh.util.diagonal_dot(pers_points - pers_origins[0],
                                           pers_vectors[pers_index_ray])
    unbiased_depth = (pers_points - pers_origins[0])[:, 2]
    pers_colors = mesh.visual.face_colors[pers_index_tri] / 255.0

    # 128. retains 7bit sub-pixel precision, 32767.0 to centering for unsigned 16 data type.
    pers_depth = (pers_depth - pers_origins[0][2]) * 128.0 + 32767.0
    unbiased_depth = (unbiased_depth - pers_origins[0][2]) * 128.0 + 32767.0

    pers_pixel_ray = pers_pixels[pers_index_ray]

    INT16_MAX = 65536
    pers_depth_near = np.ones(mesh.scene.camera.resolution, dtype=np.float32) * INT16_MAX
    pers_depth_near_unbiased = np.ones(mesh.scene.camera.resolution, dtype=np.float32) * INT16_MAX
    pers_color_near = np.zeros((res, res, 3), dtype=np.float32)

    for k in range(pers_pixel_ray.shape[0]):
        u, v = pers_pixel_ray[k, 0], pers_pixel_ray[k, 1]
        if pers_depth[k] < pers_depth_near[v, u]:
            pers_depth_near[v, u] = pers_depth[k]
            pers_depth_near_unbiased[v, u] = unbiased_depth[k]
            pers_color_near[v, u, ::-1] = pers_colors[k, 0:3]

    pers_depth_near = pers_depth_near * (pers_depth_near != INT16_MAX)
    pers_depth_near_unbiased = pers_depth_near_unbiased * (pers_depth_near_unbiased != INT16_MAX)
    pers_color_near = np.flip(pers_color_near, 0)
    pers_depth_near = np.flip(pers_depth_near, 0)
    pers_depth_near_unbiased = np.flip(pers_depth_near_unbiased, 0)
    pers_color_near = np.flip(pers_color_near, 1)
    pers_depth_near = np.flip(pers_depth_near, 1)
    pers_depth_near_unbiased = np.flip(pers_depth_near_unbiased, 1)

    return pers_color_near, pers_depth_near, pers_depth_near_unbiased

def pers_get_depth_maps(mesh, scene, res, scaling_factor=128.0):
    mesh.scene = scene
    pers_origins, pers_vectors, pers_pixels = mesh.scene.camera_rays()
    pers_points, pers_index_ray, pers_index_tri = mesh.ray.intersects_location(pers_origins,
                                                                               pers_vectors,
                                                                               multiple_hits=True)
    # (A: pers_points_origin)  ----------> (pers_origins[0] -> surface)
    # (B: pers_unit_vector)    ->          (same vector with unit norm)
    # A dot B = cos(theta)|A||B| = |A||B| = distance
    # s.t. cos(theta) = 1, |B| = 1, |A|=distance x |B|
    pers_depth = trimesh.util.diagonal_dot(pers_points - pers_origins[0],
                                           pers_vectors[pers_index_ray])
    unbiased_depth = np.abs(pers_points - pers_origins[0])[:, 2]
    pers_colors = mesh.visual.face_colors[pers_index_tri] / 255.0

    # 128. retains 7bit sub-pixel precision, 32767.0 to centering for unsigned 16 data type.
    # align depth maps to the plane (z=0)
    pers_depth = (pers_depth - pers_origins[0][2]) * scaling_factor + 32767.0
    unbiased_depth = (unbiased_depth - pers_origins[0][2]) * scaling_factor + 32767.0

    pers_pixel_ray = pers_pixels[pers_index_ray]
    pers_depth_far = np.zeros(mesh.scene.camera.resolution, dtype=np.float32)
    pers_depth_far_unbiased = np.zeros(mesh.scene.camera.resolution, dtype=np.float32)
    pers_color_far = np.zeros((res, res, 3), dtype=np.float32)

    INT16_MAX = 65536
    pers_depth_near = np.ones(mesh.scene.camera.resolution, dtype=np.float32) * INT16_MAX
    pers_depth_near_unbiased = np.ones(mesh.scene.camera.resolution, dtype=np.float32) * INT16_MAX
    pers_color_near = np.zeros((res, res, 3), dtype=np.float32)

    for k in range(pers_pixel_ray.shape[0]):
        u, v = pers_pixel_ray[k, 0], pers_pixel_ray[k, 1]
        if pers_depth[k] > pers_depth_far[v, u]:
            pers_color_far[v, u, ::-1] = pers_colors[k, 0:3]
            pers_depth_far[v, u] = pers_depth[k]
            pers_depth_far_unbiased[v, u] = unbiased_depth[k]
        if pers_depth[k] < pers_depth_near[v, u]:
            pers_depth_near[v, u] = pers_depth[k]
            pers_depth_near_unbiased[v, u] = unbiased_depth[k]
            pers_color_near[v, u, ::-1] = pers_colors[k, 0:3]

    pers_depth_near = pers_depth_near * (pers_depth_near != INT16_MAX)
    pers_depth_near_unbiased = pers_depth_near_unbiased * (pers_depth_near_unbiased != INT16_MAX)
    pers_color_near = np.flip(pers_color_near, 0)
    pers_depth_near = np.flip(pers_depth_near, 0)
    pers_depth_near_unbiased = np.flip(pers_depth_near_unbiased, 0)
    pers_color_far = np.flip(pers_color_far, 0)
    pers_depth_far = np.flip(pers_depth_far, 0)
    pers_depth_far_unbiased = np.flip(pers_depth_far_unbiased, 0)

    pers_color_near = np.flip(pers_color_near, 1)
    pers_depth_near = np.flip(pers_depth_near, 1)
    pers_depth_near_unbiased = np.flip(pers_depth_near_unbiased, 1)
    pers_color_far = np.flip(pers_color_far, 1)
    pers_depth_far = np.flip(pers_depth_far, 1)
    pers_depth_far_unbiased = np.flip(pers_depth_far_unbiased, 1)

    return pers_color_near, pers_depth_near, pers_depth_near_unbiased, \
           pers_color_far, pers_depth_far, pers_depth_far_unbiased