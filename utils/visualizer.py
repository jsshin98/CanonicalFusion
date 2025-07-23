import open3d as o3d
import numpy as np
import trimesh
import torch
from skimage import measure
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import axes3d, Axes3D


def mesh2o3d(mesh, vertex_color=None, offset=None):
    mesh_vis = o3d.geometry.TriangleMesh ()
    mesh_vis.triangles = o3d.utility.Vector3iVector (mesh.faces)
    # vertex_color = np.clip (vertex_color, 0.0, 1.0)
    if vertex_color is not None:
        mesh_vis.vertex_colors = o3d.utility.Vector3dVector (vertex_color.astype (np.float64))

    if offset is not None:
        offset_ = np.zeros_like(mesh.vertices)
        offset_[:, 0] = offset
        mesh_vis.vertices = o3d.utility.Vector3dVector (mesh.vertices + offset_)
    else:
        mesh_vis.vertices = o3d.utility.Vector3dVector (mesh.vertices)
    mesh_vis.compute_vertex_normals ()

    return mesh_vis


def to_mesh(sdf, vis=False):
    vertices, faces, normals, _ = measure.marching_cubes(sdf, level=0.0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    if vis:
        mesh.show()
    return mesh


def to_trimesh(vertices, faces, normals=None, vertex_colors=None, face_colors=None, process=False):
    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy()
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()
    if vertex_colors is not None and torch.is_tensor(vertex_colors):
        vertex_colors = vertex_colors.detach().cpu().numpy()
    if face_colors is not None and torch.is_tensor(face_colors):
        face_colors = face_colors.detach().cpu().numpy()
    if normals is not None and torch.is_tensor(normals):
        normals = normals.detach().cpu().numpy()
    if vertices.shape[0] == 1:
        vertices = vertices.squeeze()
    if faces.shape[0] == 1:
        faces = faces.squeeze()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals,
                           vertex_colors=vertex_colors, face_colors=face_colors, process=process, maintain_order=True)
    return mesh


def show_meshes(meshes, offset=None, cam=None):
    vis_mesh = []
    for i, mesh in enumerate(meshes):
        if offset is None:
            vis_mesh.append(mesh2o3d(mesh))
        else:
            vis_mesh.append(mesh2o3d(mesh, offset=offset[i]))
    if cam is not None:
        vis_mesh.append(cam)

    o3d.visualization.draw_geometries(vis_mesh)


def vis_flow(f, d=16, res=256):
    u, v, w = f[:, :, :, 0], f[:, :, :, 1], f[:, :, :, 2]
    x, y, z = torch.meshgrid(torch.linspace(0, 1, res),
                             torch.linspace(0, 1, res),
                             torch.linspace(0, 1, res))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.quiver(x[::d, ::d, ::d], y[::d, ::d, ::d], z[::d, ::d, ::d],
              u[::d, ::d, ::d]/res, v[::d, ::d, ::d]/res, w[::d, ::d, ::d]/res)
    plt.show()
