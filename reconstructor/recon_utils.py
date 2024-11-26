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


def set_smpl_model(smpl_model, smpl_params, device):
    smpl_model.betas = torch.nn.Parameter(torch.tensor(smpl_params['betas'], device=device))
    smpl_model.transl = torch.nn.Parameter(torch.tensor(smpl_params['transl'], device=device))
    smpl_model.expression = torch.nn.Parameter(torch.tensor(smpl_params['expression'], device=device))
    smpl_model.body_pose = torch.nn.Parameter(torch.tensor(smpl_params['body_pose'], device=device))
    smpl_model.global_orient = torch.nn.Parameter(torch.tensor([smpl_params['global_orient']], device=device))
    smpl_model.jaw_pose = torch.nn.Parameter(torch.tensor(smpl_params['jaw_pose'], device=device))
    smpl_model.left_hand_pose = torch.nn.Parameter(torch.tensor(smpl_params['left_hand_pose'], device=device))
    smpl_model.right_hand_pose = torch.nn.Parameter(torch.tensor(smpl_params['right_hand_pose'], device=device))
    smpl_mesh = smpl_model(return_verts=True, return_full_pose=True)
    smpl_mesh.joints = smpl_mesh.joints * torch.nn.Parameter(
        torch.tensor(smpl_params['scale'],
                     device=device))
    smpl_mesh.vertices = smpl_mesh.vertices * torch.nn.Parameter(
        torch.tensor(smpl_params['scale'],
                     device=device))
    return smpl_mesh, smpl_model
    
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


def deform_vertices_with_A(vertices, A, lbs, homogen_coord, device='cuda'):
    # vertices -> B X N X 3   lbs -> B X N X 55
    weights = lbs.expand([lbs.shape[0], -1, -1])
    T = torch.matmul(weights, A.reshape(lbs.shape[0], 55, 16)).view(lbs.shape[0], -1, 4, 4)
    v_posed_homo = torch.cat([vertices, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]
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
