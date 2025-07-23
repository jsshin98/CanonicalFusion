import cv2
import numpy as np
import torch
import trimesh
import open3d as o3d
import math


def get_cam_matrices(angles, pitches, fov=52, res=512, dist=2.0, focal=1448.15468787):
    # pitches = [i for i in range(-60, 60, 10)]

    N = len(angles)
    M = len(pitches)
    K = np.zeros((N * M, 3, 3))
    R = np.zeros((N * M, 3, 3))
    t = np.zeros((N * M, 3))

    w, h = res, res
    if fov is not None:
        fx = fy = w / (2 * math.tan(fov * math.pi / 180 / 2))
        # fx = fy = 32 / (2 * math.tan(fov * math.pi / 180 / 2))
    elif focal is not None:
        fx = fy = focal
    else:
        fx = fy = math.sqrt(res*res*2)

    cx = w / 2
    cy = h / 2
    K_init = np.array([[fx, 0, cx],
                       [0, fy, cy],
                       [0, 0, 1]])

    R_init = np.eye(3)
    R_init[1, 1] *= -1.0
    R_init[2, 2] *= -1.0

    t_init = np.zeros_like(t[0:1, :])
    t_init[0, 2] = dist
    i = 0
    for j in range(M):
        for k in range(N):
            R_delta = np.matmul(make_rotate(math.radians(pitches[j]), 0, 0),
                                make_rotate(0, math.radians(angles[k]), 0))
            R[i, :, :] = np.matmul(R_init, R_delta)
            K[i, :, :] = K_init
            t[i, :] = t_init
            i += 1

    return K, R, t

def show_mesh(meshes, offset=None, cam=None):
    vis_mesh = []
    for i, mesh in enumerate(meshes):
        if offset is None:
            vis_mesh.append(mesh)
        else:
            vis_mesh.append(mesh, offset=offset[i])
    if cam is not None:
        vis_mesh.append(cam)
    o3d.visualization.draw_geometries(vis_mesh)

def postprocess_mesh(mesh, num_faces=None):
    """Post processing mesh by removing small isolated pieces.

    Args:
        mesh (trimesh.Trimesh): input mesh to be processed
        num_faces (int, optional): min face num threshold. Defaults to 4096.
    """
    total_num_faces = len(mesh.faces)
    if num_faces is None:
        num_faces = total_num_faces // 100
    cc = trimesh.graph.connected_components(
        mesh.face_adjacency, min_len=3)
    mask = np.zeros(total_num_faces, dtype=bool)
    cc = np.concatenate([
        c for c in cc if len(c) > num_faces
    ], axis=0)
    mask[cc] = True
    mesh.update_faces(mask)

    return mesh


def mesh2o3d(mesh, vertex_color=None, offset=None):
    mesh_vis = o3d.geometry.TriangleMesh()
    mesh_vis.triangles = o3d.utility.Vector3iVector(mesh.faces)
    if vertex_color is not None:
        mesh_vis.vertex_colors = o3d.utility.Vector3dVector (vertex_color.astype (np.float64))

    if offset is not None:
        offset_ = np.zeros_like(mesh.vertices)
        offset_[:, 0] = offset
        mesh_vis.vertices = o3d.utility.Vector3dVector (mesh.vertices + offset_)
    else:
        mesh_vis.vertices = o3d.utility.Vector3dVector (mesh.vertices)
    # mesh_vis.vertex_normals = o3d.utility.Vector3dVector (mesh.vertex_normals.copy())

    return mesh_vis


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


def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3, 3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3, 3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3, 3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz, Ry), Rx)
    return R

def augment_R(R):
    new_R = np.ndarray((4, 4))
    new_R[:3, :3] = R
    new_R[3, 3] = 1.0
    return new_R

def get_rotation_matrix_np(axis, theta_degree):
    """Return the rotation matrix for the given axis and angle."""
    theta = np.radians(theta_degree)
    # print(axis)

    if axis == 'x':
        return np.array([
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'y':
        return np.array([
            [np.cos(theta), 0, np.sin(theta), 0],
            [0, 1, 0, 0],
            [-np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'z':
        return np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")


def rotate_mesh(mesh, axis, angle_degree):
    """Rotate mesh around a specific axis by a given angle."""
    R = get_rotation_matrix_np(axis, angle_degree)

    if isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.Trimesh(np.matmul(R[:3, :3], mesh.vertices[:, :, None])[:, :, 0],
                               mesh.faces, visual=mesh.visual)
    else:
        mesh.transform(R)
    return mesh

def perspective_projection(points, intrinsic, rotation, translation):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
    """
    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij, bkj->bki', intrinsic.float(), projected_points.float())
    return projected_points[:, :, :-1]


def visualize_joints(proj_joints, joints, image=None, width_ori=1920, height_ori=1280):
    width, height = 640, 640
    ratio_x = width / width_ori
    ratio_y = height / height_ori
    if image is not None:
        tmp = image.copy()
        tmp = cv2.resize(tmp, (640, 640))
    else:
        tmp = np.zeros((640, 640, 3))

    for i in range(proj_joints.shape[1]):
        pidx = (proj_joints[:, i, :])
        jidx = (joints[:, i, :])
        if int(pidx[0, 1]) < width_ori and int(pidx[0, 0]) < height_ori and \
                int(jidx[0, 1]) < width_ori and int(jidx[0, 0]) < height_ori:
            u1 = int(pidx[0, 0] * ratio_x)
            v1 = int(pidx[0, 1] * ratio_y)
            u2 = int(jidx[0, 0] * ratio_x)
            v2 = int(jidx[0, 1] * ratio_y)
            tmp = cv2.circle(tmp, (u1, v1), 3,
                             [0, 0, 255], 1)
            tmp = cv2.circle(tmp, (u2, v2), 3,
                             [0, 255, 0], 1)
            cv2.putText(tmp, '{}'.format(str(i)), (u1, v1), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 0))
    cv2.imshow('projected joints', tmp)
    cv2.waitKey(0)