import numpy as np
import trimesh
import mesh_to_sdf.surface_point_cloud
from mesh_to_sdf.surface_point_cloud import BadMeshException
from mesh_to_sdf.utils import scale_to_unit_cube, scale_to_unit_sphere, get_raster_points, check_voxels
from mesh_to_sdf import surface_point_cloud
"""mesh_to_sdf library must be installed"""



def get_surface_point_cloud(mesh, surface_point_method='scan', bounding_radius=None, scan_count=100,
                            scan_resolution=400, sample_point_count=10000000, calculate_normals=True):
    if isinstance (mesh, trimesh.Scene):
        mesh = mesh.dump ().sum ()
    if not isinstance (mesh, trimesh.Trimesh):
        raise TypeError ("The mesh parameter must be a trimesh mesh.")

    if bounding_radius is None:
        bounding_radius = np.max (np.linalg.norm (mesh.vertices, axis=1)) * 1.1

    if surface_point_method == 'scan':
        return surface_point_cloud.create_from_scans (mesh, bounding_radius=bounding_radius, scan_count=scan_count,
                                                      scan_resolution=scan_resolution,
                                                      calculate_normals=calculate_normals)
    elif surface_point_method == 'sample':
        return surface_point_cloud.sample_from_mesh (mesh, sample_point_count=sample_point_count,
                                                     calculate_normals=calculate_normals)
    else:
        raise ValueError ('Unknown surface point sampling method: {:s}'.format (surface_point_method))


def mesh_to_voxels_smpl(mesh, voxel_resolution=64, surface_point_method='scan', sign_method='normal', scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11, pad=False, check_result=False):
    # mesh = scale_to_unit_cube_smpl(mesh)

    surface_point_cloud = mesh_to_sdf.get_surface_point_cloud(mesh, surface_point_method, 3**0.5, scan_count, scan_resolution, sample_point_count, sign_method=='normal')

    return surface_point_cloud.get_voxels(voxel_resolution, sign_method=='depth', normal_sample_count, pad, check_result)


def scale_to_unit_cube_smpl(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - [0.0, -0.4, 0.0]

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

