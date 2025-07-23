import numpy as np
import trimesh
import cv2
import torch
from pysdf import SDF
from skimage import measure
from scipy.spatial.transform import Rotation as R
import collections


def load_obj_mesh(mesh, texture):
    vertex_data = []
    norm_data = []
    uv_data = []
    dict = collections.defaultdict(int)

    face_data = []
    face_norm_data = []
    face_uv_data = []

    for line in mesh:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)
        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)

            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f_c = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f_c)
                    f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                    dict[f[0] - 1] = f_c[0] - 1
                    dict[f[1] - 1] = f_c[1] - 1
                    dict[f[2] - 1] = f_c[2] - 1
                else:
                    face_uv_data.append([1, 1, 1])

            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertex_colors = []
    for k in range(len(vertex_data)):
        if k in dict:
            vertex_colors.append(uv_data[dict[k]])
        else:
            vertex_colors.append([0.0, 0.0])

    w, h = texture.shape[0], texture.shape[1]

    vertices = np.array(vertex_data)
    visuals = np.array(vertex_colors)
    faces = np.array(face_data) - 1

    vertex_colors = visuals
    vertex_colors = [[int(item[0] * w), int(item[1] * h)] for item in vertex_colors]
    vertices_colors = np.zeros((len(vertex_colors), 3))
    cnt = 0
    for item in vertex_colors:
        u = item[1]
        v = item[0]
        if u < h and v < w:
            vertices_colors[cnt, :] = texture[u, v, :]
        cnt += 1

    visuals = vertices_colors
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=visuals, process=False)

    return mesh, vertices_colors


def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat


def reconstruction_error(S1, S2, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1)).mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re


def mesh2sdf(mesh, res, v_min=-1.0, v_max=1.0):
    x_ind, y_ind, z_ind = torch.meshgrid(torch.linspace(v_min, v_max, res),
                                         torch.linspace(v_min, v_max, res),
                                         torch.linspace(v_min, v_max, res))
    pt = np.concatenate((np.asarray(x_ind).reshape(-1, 1),
                         np.asarray(y_ind).reshape(-1, 1),
                         np.asarray(z_ind).reshape(-1, 1)), axis=1)

    f = SDF(mesh.vertices, mesh.faces)
    sdf = f(pt.astype(np.float))
    sdf = sdf.reshape((res, res, res)) * -1.0
    return sdf


def rotate_mesh(vertices, angle, faces, vertex_colors, axis='x'):
    vertices_re = (np.zeros_like(vertices))
    if axis == 'y':  # pitch
        rotation_axis = np.array([1, 0, 0])
    elif axis == 'x':  # yaw
        rotation_axis = np.array([0, 1, 0])
    elif axis == 'z':  # roll
        rotation_axis = np.array([0, 0, 1])
    else:  # default is x (yaw)
        rotation_axis = np.array([0, 1, 0])

    for i in range(vertices.shape[0]):
        vec = vertices[i, :]
        rotation_degrees = angle
        rotation_radians = np.radians(rotation_degrees)

        rotation_vector = rotation_radians * rotation_axis
        rotation = R.from_rotvec(rotation_vector)
        rotated_vec = rotation.apply(vec)
        vertices_re[i, :] = rotated_vec

    mesh2 = trimesh.Trimesh(vertices=vertices_re, faces=faces, vertex_colors=vertex_colors)
    mesh2.vertices -= mesh2.bounding_box.centroid
    mesh2.vertices *= 2 / np.max(mesh2.bounding_box.extents)

    return mesh2


if __name__ == '__main__':
    voxel_resolution = 512
    width = 256
    height = 256
    fov = 60
    cam_res = 256

    RGB_MAX = [255.0, 255.0, 255.0]
    DEPTH_MAX = 255.0

    mesh = trimesh.load ('./15500_AAM_M/pred_mesh/result_19.obj')
    color_front = cv2.imread ('./15500_AAM_M/pred_color/color_19_front.png', cv2.IMREAD_COLOR)
    color_back = cv2.imread ('./15500_AAM_M/pred_color/color_19_back.png', cv2.IMREAD_COLOR)
    depth_front = cv2.imread ('./15500_AAM_M/pred_depth/depth_19_front.png', cv2.IMREAD_ANYDEPTH)
