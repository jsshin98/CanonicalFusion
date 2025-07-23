import smplx
import torch
import numpy as np
import pdb
import trimesh
import pickle
import os
import torch.nn.functional as F
import copy
import glob
import matplotlib.pyplot as plt
import json
import tqdm
import copy

import math

def set_smpl_model(smpl_model, smpl_params, device):
    # smpl_model.betas = torch.nn.Parameter(torch.tensor(smpl_params.betas, device=device))
    # smpl_model.transl = torch.nn.Parameter(torch.tensor(smpl_params.transl, device=device))
    # smpl_model.expression = torch.nn.Parameter(torch.tensor(smpl_params.expression, device=device))
    # smpl_model.body_pose = torch.nn.Parameter(torch.tensor(smpl_params.body_pose, device=device))
    # smpl_model.global_orient = torch.nn.Parameter(torch.tensor(smpl_params.global_orient, device=device))
    # smpl_model.jaw_pose = torch.nn.Parameter(torch.tensor(smpl_params.jaw_pose, device=device))
    # smpl_model.left_hand_pose = torch.nn.Parameter(torch.tensor(smpl_params.left_hand_pose, device=device))
    # smpl_model.right_hand_pose = torch.nn.Parameter(torch.tensor(smpl_params.right_hand_pose, device=device))
    smpl_model.betas = torch.nn.Parameter(smpl_params.betas.to(device))
    smpl_model.transl = torch.nn.Parameter(smpl_params.transl.to(device))
    smpl_model.expression = torch.nn.Parameter(smpl_params.expression.to(device))
    smpl_model.body_pose = torch.nn.Parameter(smpl_params.body_pose.to(device))
    smpl_model.global_orient = torch.nn.Parameter(smpl_params.global_orient.to(device))
    smpl_model.jaw_pose = torch.nn.Parameter(smpl_params.jaw_pose.to(device))
    smpl_model.left_hand_pose = torch.nn.Parameter(smpl_params.left_hand_pose.to(device))
    smpl_model.right_hand_pose = torch.nn.Parameter(smpl_params.right_hand_pose.to(device))
    smpl_mesh = smpl_model(return_verts=True, return_full_pose=True)
    smpl_mesh.joints = smpl_mesh.joints * torch.nn.Parameter(
        torch.tensor(smpl_params.scale,
                     device=device))
    smpl_mesh.vertices = (smpl_mesh.vertices * torch.nn.Parameter(
        torch.tensor(smpl_params.scale,
                     device=device))).squeeze().detach().cpu().numpy()
    smpl_mesh.faces = smpl_model.faces
    return smpl_mesh, smpl_model, smpl_params

def subdivide_concat(smpl, uv=None):
    if uv is not None:
        vertices, faces = trimesh.remesh.subdivide(
            vertices = np.hstack((smpl.vertices, uv)),
            faces = smpl.faces,
        )
        return vertices[:, :3], faces, vertices[:, 3:]
    else:
        vertices, faces = trimesh.remesh.subdivide(
            vertices = smpl.vertices,
            faces = smpl.faces,
        )
        return vertices, faces

def real2smpl(mesh1, mesh2):
    center1 = mesh1.bounding_box.centroid
    scale1 = 2.0 / np.max(mesh1.bounding_box.extents)

    center2 = mesh2.bounding_box.centroid
    scale2 = 2.0 / np.max(mesh2.bounding_box.extents)
    vertices = (mesh1.vertices - center1) * scale1 / scale2 + center2
    return vertices
    
def load_gt_mesh(path2obj, avg_height=180.0):
    exts, files = ['.obj', '.ply'], []
    [files.extend(sorted(glob.glob(os.path.join(path2obj, '*'+e)))) for e in exts]

    meshes, scales, centers = [], [], []
    for file in files:
        m = trimesh.load_mesh(file, process=False, maintain_order=True)

        vertices = m.vertices
        vmin = vertices.min(0)
        vmax = vertices.max(0)
        up_axis = 1  # if (vmax - vmin).argmax() == 1 else 2
        center = np.median(vertices, 0)
        center[up_axis] = 0.5 * (vmax[up_axis] + vmin[up_axis])
        scale = avg_height / (vmax[up_axis] - vmin[up_axis])
        #
        # vertices -= center
        # vertices *= scale

        mesh = trimesh.Trimesh(vertices=vertices, faces=m.faces, visual=m.visual, process=False)
        meshes.append(mesh)
        scales.append(scale)
        centers.append(center)
    return meshes, files, scales, centers

def deform_vertices(vertices, smpl_model, lbs, full_pose, inverse=False, return_vshape=False, device='cuda:0'):
    v_shaped = smpl_model.v_template + \
               smplx.lbs.blend_shapes(smpl_model.betas, smpl_model.shapedirs)
    # do not use smpl_model.joints -> it fails (don't know why)
    joints = smplx.lbs.vertices2joints(smpl_model.J_regressor, v_shaped)
    rot_mats = smplx.lbs.batch_rodrigues(full_pose.view(-1, 3)).view([1, -1, 3, 3])
    joints_warped, A = batch_rigid_transform(rot_mats.to(device), joints[:, :55, :].to(device), smpl_model.parents,
                                             inverse=inverse, dtype=torch.float32)

    weights = lbs.unsqueeze(dim=0).expand([1, -1, -1]).to(device)
    num_joints = smpl_model.J_regressor.shape[0]
    T = torch.matmul(weights, A.reshape(1, num_joints, 16)).view(1, -1, 4, 4)
    homogen_coord = torch.ones([1, vertices.shape[1], 1], dtype=torch.float32).to(device)
    v_posed_homo = torch.cat([vertices, homogen_coord], dim=2)

    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]
    if return_vshape:
        return verts, v_shaped
    else:
        return verts

def deform_vertices_with_A_image(vertices, A, lbs, device='cuda'):
    lbs_flat = lbs.reshape(lbs.shape[0], -1, 55)
    weights = lbs_flat.expand([lbs.shape[0], -1, -1]).to(device)
    vertices_flat = torch.transpose(vertices.reshape(lbs.shape[0], 3, -1), 2, 1)
    T = torch.matmul(weights, A.squeeze().reshape(lbs.shape[0], 55, 16)).view(lbs.shape[0], -1, 4, 4)
    homogen_coord = torch.ones([lbs.shape[0], vertices_flat.shape[1], 1], dtype=torch.float32).to(device)
    v_posed_homo = torch.cat([vertices_flat, homogen_coord], dim=2)

    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]

    # verts = torch.transpose(verts, 2, 1).reshape(lbs.shape[0], 3, lbs.shape[1], lbs.shape[2])
    verts = torch.transpose(verts, 2, 1).reshape(lbs.shape[0], 3, -1)

    return verts


def diff_forward_skinning(vertices, A, lbs, homogen_coord, device='cuda'):
    # vertices -> B X N X 3   lbs -> B X N X 55
    weights = lbs.expand([lbs.shape[0], -1, -1])
    T = torch.matmul(weights, A.reshape(lbs.shape[0], 55, 16)).view(lbs.shape[0], -1, 4, 4)
    v_posed_homo = torch.cat([vertices, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]
    return verts

def deform_to_star_pose(vertices, smpl_model, lbs, full_pose,
                        inverse=True, return_vshape=True, device='cuda:0'):
    full_pose = full_pose.clone().detach() * 0
    full_pose[0, 5] = math.radians(20)
    full_pose[0, 8] = -math.radians(20)

    v_shaped = smpl_model.v_template + \
               smplx.lbs.blend_shapes(smpl_model.betas, smpl_model.shapedirs)
    # do not use smpl_model.joints -> it fails (don't know why)
    joints = smplx.lbs.vertices2joints(smpl_model.J_regressor, v_shaped)
    rot_mats = smplx.lbs.batch_rodrigues(full_pose.view(-1, 3)).view([1, -1, 3, 3])
    joints_warped, A = batch_rigid_transform(rot_mats, joints[:, :55, :], smpl_model.parents,
                                             inverse=inverse, dtype=torch.float32)

    weights = lbs.unsqueeze(dim=0).expand([1, -1, -1]).to(device)
    num_joints = smpl_model.J_regressor.shape[0]
    T = torch.matmul(weights, A.reshape(1, num_joints, 16)).view(1, -1, 4, 4)

    verts = None
    if vertices is not None:
        homogen_coord = torch.ones([1, vertices.shape[1], 1], dtype=torch.float32).to(device)
        v_posed_homo = torch.cat([vertices, homogen_coord], dim=2)

        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
        verts = v_homo[:, :, :3, 0]

    if return_vshape:
        weights_v = smpl_model.lbs_weights.unsqueeze(dim=0).expand([1, -1, -1]).to(device)
        T_v = torch.matmul(weights_v, A.reshape(1, num_joints, 16)).view(1, -1, 4, 4)

        homogen_coord_v = torch.ones([1, v_shaped.shape[1], 1], dtype=torch.float32).to(device)
        v_posed_homo_v = torch.cat([v_shaped, homogen_coord_v], dim=2)

        v_homo_v = torch.matmul(T_v, torch.unsqueeze(v_posed_homo_v, dim=-1))
        v_shaped = v_homo_v[:, :, :3, 0]

        return verts, v_shaped
    return verts

def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents,
                          inverse=True,
                          dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints
    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32
    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.view(-1, 3, 3),
        rel_joints.view(-1, 3, 1)).view(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    if inverse is True:
        posed_joints = torch.unsqueeze(posed_joints, dim=-1)
        rel_joints = posed_joints.clone()
        rel_joints[:, 1:] -= posed_joints[:, parents[1:]]
        # rot_inv = torch.transpose(rot_mats.view(-1, 3, 3), dim0=1, dim1=2)
        transforms_mat_inv = transform_mat(
            rot_mats.view(-1, 3, 3),
            torch.zeros_like(rel_joints.view(-1, 3, 1))).view(-1, joints.shape[1], 4, 4)

        transform_chain = [transforms_mat_inv[:, 0]]
        for i in range(1, parents.shape[0]):
            curr_res = torch.matmul(transform_chain[parents[i]],
                                    transforms_mat_inv[:, i])
            transform_chain.append(curr_res)

        for i in range(len(transform_chain)):
            transform_chain[i] = torch.inverse(transform_chain[i])
            transform_chain[i][:, :3, 3] = joints[:, i, :, :].view(-1, 3)

        transforms = torch.stack(transform_chain, dim=1)
        joints_homogen = F.pad(posed_joints, [0, 0, 0, 1])

        rel_transforms = transforms - F.pad(
            torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])
        return posed_joints, rel_transforms
    else:
        joints_homogen = F.pad(joints, [0, 0, 0, 1])

        rel_transforms = transforms - F.pad(
            torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

        return posed_joints, rel_transforms

def pose_extraction(full_pose, smpl_info, smpl_model, device):
    rot_mats = smplx.lbs.batch_rodrigues(full_pose.view(-1, 3)).view([1, -1, 3, 3])
    joints_warped, A = batch_rigid_transform(rot_mats.to(device),
                                                            smpl_info['joints'][:, :55, :].to(device),
                                                            smpl_model.parents,
                                                            inverse=False, dtype=torch.float32)
    joints_warped, A_inv = batch_rigid_transform(rot_mats.to(device),
                                                                smpl_info['joints'][:, :55, :].to(device),
                                                                smpl_model.parents,
                                                                inverse=True, dtype=torch.float32)
    return A.detach().cpu()

def export_obj(mesh,
               include_normals=True,
               include_color=True,
               include_texture=True):
    """
    Export a mesh as a Wavefront OBJ file

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Mesh to be exported

    Returns
    -----------
    export : str
      OBJ format output
    """
    # store the multiple options for formatting
    # vertex indexes for faces
    face_formats = {('v',): '{}',
                    ('v', 'vn'): '{}//{}',
                    ('v', 'vt'): '{}/{}',
                    ('v', 'vn', 'vt'): '{}/{}/{}'}
    # we are going to reference face_formats with this
    face_type = ['v']

    # OBJ includes vertex color as RGB elements on the same line
    if include_color and mesh.visual.kind in ['vertex', 'face']:
        # create a stacked blob with position and color
        v_blob = np.column_stack((
            mesh.vertices,
            to_float(mesh.visual.vertex_colors[:, :3])))
    else:
        # otherwise just export vertices
        v_blob = mesh.vertices

    # add the first vertex key and convert the array
    export = 'v ' + array_to_string(v_blob,
                                         col_delim=' ',
                                         row_delim='\nv ',
                                         digits=8) + '\n'

    # only include vertex normals if they're already stored
    if include_normals and 'vertex_normals' in mesh._cache:
        # if vertex normals are stored in cache export them
        face_type.append('vn')
        export += 'vn '
        export += array_to_string(mesh.vertex_normals,
                                       col_delim=' ',
                                       row_delim='\nvn ',
                                       digits=8) + '\n'


    if include_texture:
        # if vertex texture exists and is the right shape export here
        face_type.append('vt')
        export += 'vt '

        export += array_to_string(mesh.metadata['vertex_texture'],
                                       col_delim=' ',
                                       row_delim='\nvt ',
                                       digits=8) + '\n'


    # the format for a single vertex reference of a face
    face_format = face_formats[tuple(face_type)]
    faces = 'f ' + array_to_string(mesh.faces + 1,
                                        col_delim=' ',
                                        row_delim='\nf ',
                                        value_format=face_format)
    # add the exported faces to the export
    export += faces

    return export