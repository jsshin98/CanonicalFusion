import torch

from diff_renderer.normal_nds.nds.core import Mesh


def mesh_smoothness_loss(mesh, vertex, loss_function=torch.nn.L1Loss()):
    V = mesh.vertices
    F = mesh.indices
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    v0_ori = vertex[F[:, 0]]
    v1_ori = vertex[F[:, 1]]
    v2_ori = vertex[F[:, 2]]
    e0 = (v0 - v0_ori).norm(dim=1, p=2)
    e1 = (v1 - v1_ori).norm(dim=1, p=2)
    e2 = (v2 - v2_ori).norm(dim=1, p=2)

    loss = loss_function(e0, e1) + loss_function(e1, e2) + loss_function(e2, e0)
    return loss.mean()

