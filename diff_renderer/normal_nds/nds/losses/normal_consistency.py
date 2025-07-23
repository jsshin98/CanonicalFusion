import torch

from diff_renderer.normal_nds.nds.core import Mesh

def normal_consistency_loss(mesh: Mesh):
    """ Compute the normal consistency term as the cosine similarity between neighboring face normals.

    Args:
        mesh (Mesh): Mesh with face normals.
    """

    loss = 1 - torch.cosine_similarity(mesh.face_normals[mesh.connected_faces[:, 0]], mesh.face_normals[mesh.connected_faces[:, 1]], dim=1)
    return (loss**2).mean()

def normal_consistency_loss_l2(normal, init_normal, loss_function=torch.nn.MSELoss()):
    """
    keep the normal consistent with initial (smpl) normals
    """
    loss = loss_function(normal, init_normal)
    return loss