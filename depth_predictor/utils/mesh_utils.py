import numpy as np
import trimesh
import cv2
import torch
from pysdf import SDF
from skimage import measure
from scipy.spatial.transform import Rotation as R

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

def trimesh_initialize(mesh):
    # is the current mesh watertight?
    mesh.is_watertight

    # what's the euler number for the mesh?
    mesh.euler_number

    # the convex hull is another Trimesh object that is available as a property
    # lets compare the volume of our mesh with the volume of its convex hull
    print(mesh.volume / mesh.convex_hull.volume)

    # since the mesh is watertight, it means there is a
    # volumetric center of mass which we can set as the origin for our mesh
    mesh.vertices -= mesh.center_mass

    # what's the moment of inertia for the mesh?
    mesh.moment_inertia

    # if there are multiple bodies in the mesh we can split the mesh by
    # connected components of face adjacency
    # since this example mesh is a single watertight body we get a list of one mesh
    mesh.split()

    # facets are groups of coplanar adjacent faces
    # set each facet to a random color
    # colors are 8 bit RGBA by default (n, 4) np.uint8
    # for facet in mesh.facets:
    #     mesh.visual.face_colors[facet] = trimesh.visual.random_color()

    # preview mesh in an opengl window if you installed pyglet and scipy with pip
    # mesh.show()

    # # transform method can be passed a (4, 4) matrix and will cleanly apply the transform
    # mesh.apply_transform(trimesh.transformations.random_rotation_matrix())
    #
    # # axis aligned bounding box is available
    # mesh.bounding_box.extents
    #
    # # a minimum volume oriented bounding box also available
    # # primitives are subclasses of Trimesh objects which automatically generate
    # # faces and vertices from data stored in the 'primitive' attribute
    # mesh.bounding_box_oriented.primitive.extents
    # mesh.bounding_box_oriented.primitive.transform
    #
    # # show the mesh appended with its oriented bounding box
    # # the bounding box is a trimesh.primitives.Box object, which subclasses
    # # Trimesh and lazily evaluates to fill in vertices and faces when requested
    # # (press w in viewer to see triangles)
    # (mesh + mesh.bounding_box_oriented).show()
    #
    # # bounding spheres and bounding cylinders of meshes are also
    # # available, and will be the minimum volume version of each
    # # except in certain degenerate cases, where they will be no worse
    # # than a least squares fit version of the primitive.
    # print(mesh.bounding_box_oriented.volume,
    #       mesh.bounding_cylinder.volume,
    #       mesh.bounding_sphere.volume)

    return mesh

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
