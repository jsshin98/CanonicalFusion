import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R

def rotationx(theta):
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(theta / 180 * np.pi), np.sin(theta / 180 * np.pi), 0.0],
        [0.0, -np.sin(theta / 180 * np.pi), np.cos(theta / 180 * np.pi), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
def rotationy(theta):
    return np.array([
        [np.cos(theta / 180 * np.pi), 0.0, np.sin(theta / 180 * np.pi), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-np.sin(theta / 180 * np.pi), 0.0, np.cos(theta / 180 * np.pi), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
def location(x, y, z):
    return np.array([
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0]
    ])
def rotate_mesh(mesh, angle):
    vertices = mesh.vertices
    vertices_re = (np.zeros_like(vertices))
    # mesh.show()

    for i in range(vertices.shape[0]):
        vec = vertices[i, :]
        rotation_degrees = angle
        rotation_radians = np.radians(rotation_degrees)
        rotation_axis = np.array([0, 1, 0])

        rotation_vector = rotation_radians * rotation_axis
        rotation = R.from_rotvec(rotation_vector)
        rot = rotation.as_matrix()
        rotated_vec = rotation.apply(vec)
        vertices_re[i, :] = rotated_vec

    mesh2 = trimesh.Trimesh(vertices=vertices_re, faces=mesh.faces, vertex_colors=mesh.visual.vertex_colors)

    return mesh2