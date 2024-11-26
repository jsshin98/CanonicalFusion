import smplx
import torch
import numpy as np
import pdb
import trimesh
# from sklearn.manifold import TSNE
import pickle
import os
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
import json
import tqdm
import json
import os.path as osp

from PIL import Image

def set_smpl_model(model_folder, device="cuda"):
    """
        create smpl-x instance
        :return: a smpl instance
    """
    return smplx.create(model_path=model_folder,
                        model_type='smplx',
                        gender='male',
                        num_betas=10, ext='npz',
                        use_face_contour=True,
                        flat_hand_mean=True,
                        num_pca_comps=12,
                        use_pca=True,
                        ).to(device)
 
def get_smpl_model(smpl_params, smpl_model):
    # return None if self.smpl_model is None:
    return smpl_model(transl=smpl_params['transl'],
                            betas=smpl_params['betas'],
                            body_pose=smpl_params['body_pose'],
                            # global_orient=smpl_params['global_orient'],
                            jaw_pose=smpl_params['jaw_pose'],
                            #joints=smpl_params['joints'],
                            expression=smpl_params['expression'],
                            left_hand_pose=smpl_params['left_hand_pose'],
                            right_hand_pose=smpl_params['right_hand_pose'],
                            return_verts=True)

def set_smplx_model(smpl_json, device="cuda:0"):
    with open(smpl_json, 'r') as f:
        smpl_model = json.load(f)
    smpl_model_template = set_smpl_model("./resource/smpl_models/smplx/SMPLX_MALE_2020.npz")
    
    # mesh = trimesh.load(mesh_path)
    smpl_model_template.betas = torch.nn.Parameter(torch.tensor([smpl_model['betas']], device=device))
    
    return smpl_model_template

def set_smplx_model_beta(smpl_beta, smpl_model, pose=None, device="cuda:0"):
    # mesh = trimesh.load(mesh_path)
    smpl_model.betas = torch.nn.Parameter(smpl_beta.to(device))
    smpl_model_posed = None

    if pose is not None:
        smpl_model_posed = copy.deepcopy(smpl_model)
        smpl_model_posed.body_pose = torch.nn.Parameter(pose[1:22, :].unsqueeze(0).to(device))
        return smpl_model, smpl_model_posed
    return smpl_model, smpl_model_posed

def real2smpl(mesh1, mesh2):
    vts1 = mesh1.vertices
    center1 = mesh1.bounding_box.centroid
    scale1 = 2.0 / np.max(mesh1.bounding_box.extents)

    #align mesh1 to mesh2 if mesh2 path provided
    vts2 = mesh2.vertices
    center2 = mesh2.bounding_box.centroid
    scale2 = 2.0 / np.max(mesh2.bounding_box.extents)
    new_vertices = (vts1 - center1) * scale1 / scale2 + center2
    #else:
    #    #align mesh1 to origin
    #    new_vertices = (vts1 - center1) * scale1

    mesh1.vertices = new_vertices
    return mesh1        

def real2smpl_vert(mesh1, mesh2, get_transform=False):
    vts1 = mesh1.vertices
    center1 = mesh1.bounding_box.centroid
    scale1 = 2.0 / np.max(mesh1.bounding_box.extents)
    #align mesh1 to mesh2 if mesh2 path provided
    vts2 = mesh2.vertices
    center2 = mesh2.bounding_box.centroid
    scale2 = 2.0 / np.max(mesh2.bounding_box.extents)
    
    new_vertices = (vts1 - center1) * scale1 / scale2 + center2
    #mesh1.vertices = new_vertices
    
    if get_transform:
        return mesh1, (center1, scale1 / scale2, center2)
    
    return new_vertices

def subdivide_lbs(vertices, faces, smpl_lbs):
    num = faces.shape[0]
    new_lbs = torch.zeros((num*3, 55), dtype=float)
    new_vertices = torch.zeros((num*3, 3), dtype=float)
    vertices = torch.Tensor(vertices)
    for p in range(num):
        i, j, k = faces[p]
        weights = np.random.dirichlet([1,1,1])
        new_lbs[p, :] = smpl_lbs[i, :] * weights[0] + smpl_lbs[j, :] * weights[1] + smpl_lbs[k, :] * weights[2]
        new_vertices[p, :] = vertices[i, :] * weights[0] + vertices[j, :] * weights[1] + vertices[k, :] * weights[2]
        new_lbs[p+num, :] = smpl_lbs[i, :] * weights[1] + smpl_lbs[j, :] * weights[0] + smpl_lbs[k, :] * weights[2]
        new_vertices[p+num, :] = vertices[i, :] * weights[1] + vertices[j, :] * weights[0] + vertices[k, :] * weights[2]
        new_lbs[p+num*2, :] = smpl_lbs[i, :] * weights[2] + smpl_lbs[j, :] * weights[1] + smpl_lbs[k, :] * weights[0]
        new_vertices[p+num*2, :] = vertices[i, :] * weights[2] + vertices[j, :] * weights[1] + vertices[k, :] * weights[0]

    return new_lbs, new_vertices

def subdivide_concat(vertices, faces, smpl_data):
    vertices, faces = trimesh.remesh.subdivide(
        vertices = np.hstack((vertices, smpl_data)),
        faces = faces,
    )
    return vertices[:, :3], faces, vertices[:, 3:]

def subdivide(vertices, faces):
    vertices, faces = trimesh.remesh.subdivide(
        vertices = vertices,
        faces = faces,
    )
    return vertices, faces
    

def postprocess_mesh(mesh, num_faces=2000):
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
    mask = np.zeros(total_num_faces, dtype=np.bool)
    cc = np.concatenate([
        c for c in cc if len(c) > num_faces
    ], axis=0)
    mask[cc] = True
    mesh.update_faces(mask)
    return mesh


def flatten_lbs(smpl_mesh, q_vts, lbs, smplx_seg):
    smplx_seg_keys = smplx_seg.keys()

    inverse_smplx_seg = dict()
    for key in smplx_seg_keys:
        for idx in smplx_seg[key]:
            inverse_smplx_seg[idx] = key
    part_seg = dict()
    for key in smplx_seg.keys():
        part_seg[key] = []

    eyeremoved_mesh = postprocess_mesh(smpl_mesh, num_faces=2000)

    nearest_distances, nearest_sources = eyeremoved_mesh.kdtree.query(q_vts[smplx_seg['leftEye']])
    for dst, src in enumerate(nearest_sources):
        lbs[src] = lbs[dst]

    nearest_distances, nearest_sources = eyeremoved_mesh.kdtree.query(q_vts[smplx_seg['rightEye']])
    for dst, src in enumerate(nearest_sources):
        lbs[src] = lbs[dst]

    nearest_distances, nearest_sources = smpl_mesh.kdtree.query(q_vts)
    for dst, src in enumerate(nearest_sources):
        try:
            seg = inverse_smplx_seg[src]
            part_seg[seg].append(dst)
        except:
            continue

    for key in smplx_seg.keys():
        lbs_part = lbs[part_seg[key]]
        
        if key == 'rightHand' or key == 'rightHandIndex1':
            ids = smplx_partids['handr']
        elif key == 'leftHand' or key == 'leftHandIndex1':
            ids = smplx_partids['handl']
        # elif key == 'rightEye' or key == 'leftEye':
        #     ids = smplx_partids['eyeball']
        else:
            continue
        
        lbs_part_joints = lbs_part[:, ids]
        lbs_part_joints_avg = torch.mean(lbs_part_joints, dim=1, keepdims=True)
        lbs_part[:, ids] = lbs_part_joints_avg.repeat(1, lbs_part_joints.shape[1])
        lbs[part_seg[key]] = lbs_part
    
    return lbs


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
    
def positional_encoding(coord, order=1):
    encoded_verts = []
    for c in range(coord.shape[0]):
        for k in range(order):
            encoded_verts.append(np.sin(2.0 ** float(k) * np.pi * coord[c]))
            encoded_verts.append(np.cos(2.0 ** float(k) * np.pi * coord[c]))


    encoded_verts = torch.stack(encoded_verts, dim=0)
    return encoded_verts

def write_ply(ply_name, verts, rgbs=None, eps=1e-8):
    if rgbs is None:
        #print('Warning: rgb not specified, use normalized 3d coords instead...')
        v_min = np.amin(verts, axis=0, keepdims=True)
        v_max = np.amax(verts, axis=0, keepdims=True)
        rgbs = (verts - v_min) / np.maximum(eps, v_max - v_min)
    if rgbs.max() < 1.001:
        rgbs = (rgbs * 255.).astype(np.uint8)
    
    with open(ply_name, 'w') as f:
        # headers
        f.writelines([
            'ply\n'
            'format ascii 1.0\n',
            'element vertex {}\n'.format(verts.shape[0]),
            'property float x\n',
            'property float y\n',
            'property float z\n',
            'property uchar red\n',
            'property uchar green\n',
            'property uchar blue\n',
            'end_header\n',
            ]
        )
        
        for i in range(verts.shape[0]):
            str = '{:10.8f} {:10.8f} {:10.8f} {:d} {:d} {:d}\n'\
                .format(verts[i,0], verts[i,1], verts[i,2],
                    rgbs[i,0], rgbs[i,1], rgbs[i,2])
            f.write(str)
            
    return verts, rgbs


def plt_scatter(feature, path):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    color = np.array([2 * np.random.random_sample() for i in range(feature.shape[0])])

    ax.scatter3D(feature[:, 0], feature[:, 1], feature[:, 2], c=color)
    plt.show()

def get_key_from_value(data, val):
    keys = [k for k, v in data.items() if val in v]
    if keys:
        return keys[0]
    return None

def canonicalization(verts, full_lbs, full_pose, smpl_params, smpl_path, real_dist=300.0, device='cuda'):
    # real space to smpl space
    verts = (verts - smpl_params['centroid_real'].to(device)) * smpl_params['scale_real'].to(device)[0]  \
            / smpl_params['scale_smplx'].to(device)[0] + smpl_params['centroid_smplx'].to(device)
    
    _, _ = utils.write_ply('posed_pred_depth.ply', verts.squeeze(0).detach().cpu().numpy())

    # verts, _ = write_ply('posed_temp.ply', verts.detach().cpu().numpy())
    # pcd = trimesh.PointCloud(verts.detach().cpu().numpy())
    # pcd.show()
    pdb.set_trace()
    smpl_model_posed = utils.set_smplx_model_beta(smpl_params['betas'], smpl_path, full_pose, device=device) # posed smpl model
    
    # smpl_temp = trimesh.Trimesh(smpl_model_posed(return_verts=True).vertices.detach().cpu().numpy().squeeze(), smpl_model_posed.faces, process=False)
    # scan_temp = trimesh.Trimesh(verts, )
    # smpl_temp.export('posed_smpl.obj')

    # canonicalization
    warped_verts = utils.deform_vertices(torch.Tensor(verts).to(device).unsqueeze(0),
                                   smpl_model_posed, full_lbs.to(device),
                                   full_pose.to(device),
                                   inverse=True,
                                   return_vshape=False,
                                   device=device)
    _, _ = utils.write_ply('canon_pred_depth.ply', warped_verts.squeeze(0).detach().cpu().numpy())
    #pcd = trimesh.PointCloud(warped_verts.squeeze(0).detach().cpu().numpy())
    #pcd.show()
    return warped_verts


################################ previous codes #######################################

def mapping(smplx_path, mesh_path):

    default_smpl = set_smpl_model(smplx_path)

    with open('data/SMPLX/0_0_00.json', 'rb') as f:
        data = json.load(f)

    default_smpl.betas = torch.nn.Parameter(torch.tensor([data['betas']], device='cuda'))
    
    smplx = trimesh.Trimesh(default_smpl(return_verts=True).vertices.detach().cpu().numpy().squeeze(), default_smpl.faces, process=False)

    mesh = trimesh.load(os.path.join(mesh_path, '0_0_00.obj'), process=False)


    with open(os.path.join(smplx_path, 'body_segmentation/smplx/smplx_vert_segmentation.json'), 'rb') as f:
        smplx_path = json.load(f)

    mesh = real2smpl(mesh, smplx)

    nearest_distances, nearest_sources = smplx.kdtree.query(mesh.vertices)

    seg_color = np.zeros((mesh.vertices.shape[0], 3))

    color_map = dict()
    mesh_seg = dict()

    for key in smplx_path.keys():
        color_map[key] = np.random.rand(3)

        #color_map['rightHandIndex1'] = 50
        #color_map['rightHand'] = 100
        #color_map['leftHandIndex1'] = 150
        #color_map['leftHand'] = 200
        mesh_seg[key] = []
    
    for dst, src in enumerate(nearest_sources):
        seg = get_key_from_value(smplx_path, src)
        if seg is None:
            continue
        mesh_seg[seg].append(dst)
        seg_color[dst] = color_map[seg]
    
    with open('mesh_segmented.pkl', 'wb') as f:
        pickle.dump(mesh_seg, f)
    mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, vertex_colors=seg_color)
    mesh.export('mesh_segmented.obj')

def hand_dump(seg_file, pkl_file):
    smplx_partids = {
        'body': [0,1,2,3,4,5,6,9,13,14,16,17,18,19],
        'face': [12, 15, 22],
        'eyeball': [23, 24],
        'leg': [4, 5, 7, 8, 10, 11],
        'arm': [18, 19, 20, 21],
        'handl': [20] + list(range(25, 40)),
        'handr': [21] + list(range(40, 55)),
        'footl': [7,10],
        'footr': [8,11],
        'ftip': [27, 30, 33, 36, 39, 42, 45, 48, 51, 54]
    }

    with open(seg_file, 'rb') as f:
        seg_info = pickle.load(f)
    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    lbs = data['lbs']

    right_finger_lbs = lbs[seg_info['rightHandIndex1']]
    right_hand_lbs = lbs[seg_info['rightHand']]
    left_finger_lbs = lbs[seg_info['leftHandIndex1']]
    left_hand_lbs = lbs[seg_info['leftHand']]

    right_finger_lbs_joints = right_finger_lbs[:, smplx_partids['handr']]
    right_finger_lbs_joints_avg = torch.sum(right_finger_lbs_joints, dim=1)/len(right_finger_lbs_joints[1])
    right_finger_lbs[:, smplx_partids['handr']] = right_finger_lbs_joints_avg.unsqueeze(1).repeat(1, len(right_finger_lbs_joints[1]))

    right_hand_lbs_joints = right_hand_lbs[:, smplx_partids['handr']]
    right_hand_lbs_joints_avg = torch.sum(right_hand_lbs_joints, dim=1)/len(right_hand_lbs_joints[1])
    right_hand_lbs[:, smplx_partids['handr']] = right_hand_lbs_joints_avg.unsqueeze(1).repeat(1, len(right_hand_lbs_joints[1]))

    left_finger_lbs_joints = left_finger_lbs[:, smplx_partids['handl']]
    left_finger_lbs_joints_avg = torch.sum(left_finger_lbs_joints, dim=1)/len(left_finger_lbs_joints[1])
    left_finger_lbs[:, smplx_partids['handl']] = left_finger_lbs_joints_avg.unsqueeze(1).repeat(1, len(left_finger_lbs_joints[1]))

    left_hand_lbs_joints = left_hand_lbs[:, smplx_partids['handl']]
    left_hand_lbs_joints_avg = torch.sum(left_hand_lbs_joints, dim=1)/len(left_hand_lbs_joints[1])
    left_hand_lbs[:, smplx_partids['handl']] = left_hand_lbs_joints_avg.unsqueeze(1).repeat(1, len(left_hand_lbs_joints[1]))

    lbs[seg_info['rightHandIndex1']] = right_finger_lbs
    lbs[seg_info['rightHand']] = right_hand_lbs
    lbs[seg_info['leftHandIndex1']] = left_finger_lbs
    lbs[seg_info['leftHand']] = left_hand_lbs

    data['lbs'] = lbs

    with open('data/MESH/FB-00002_RE/0_0_00_hand_dumped.pkl', 'wb') as f:
        pickle.dump(data, f)
    

def tsne(data_path):
    from parametric_tsne.parametric_tSNE import Parametric_tSNE
    import tensorflow as tf
    
    mesh = trimesh.load(os.path.join(data_path, '0_0_00.obj'), process=False)
    verts, faces = mesh.vertices, mesh.faces

    with open(os.path.join(data_path, '0_0_00_hand_dumped.pkl'), 'rb') as f:
        data = pickle.load(f)
    lbs = data['lbs'].detach().cpu().numpy()

    #for i in range(1):
    #    subdiv_verts, subdiv_faces, subdiv_lbs = subdivide_concat(subdiv_verts, subdiv_faces, subdiv_lbs)

    #encoded_verts = positional_encoding(smpl(return_verts=True).vertices)
    #feats = np.hstack((subdiv_lbs, subdiv_verts))

    #tsne_np = TSNE(n_components=3).fit_transform(feats)
    ptSNE = Parametric_tSNE(num_inputs=55, num_outputs=3, perplexities=3)

    ptSNE.fit(lbs)
    tsne_np = ptSNE.transform(lbs)

    result = {'verts':verts, 'faces':faces, 'lbs':lbs, 'tsne_np':tsne_np}

    with open(os.path.join(data_path, 'subdivided_lbs_real_handdumped.pkl'), 'wb') as f:
        pickle.dump(result, f)

    write_ply(os.path.join(data_path, 'real_handdumped.ply'), verts=verts, rgbs=((tsne_np/np.max(tsne_np)/2+0.5)*255).astype(np.uint8))   
    mesh = trimesh.Trimesh(verts, faces, vertex_colors=((tsne_np/np.max(tsne_np)/2+0.5)*255).astype(np.uint8))
    mesh.export(os.path.join(data_path, 'real_mesh_handdumped.obj'))

if __name__=='__main__':

    # mapping('resource/smpl_models', 'data/MESH/FB-00002_RE')
    # hand_dump('mesh_segmented.pkl', 'data/MESH/FB-00002_RE/0_0_00.pkl')
    tsne('data/MESH/FB-00002_RE')

