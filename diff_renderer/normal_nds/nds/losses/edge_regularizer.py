import torch

from diff_renderer.normal_nds.nds.core import Mesh


def mesh_face_loss(mesh, loss_function=torch.nn.L1Loss()):
    V = mesh.vertices
    F = mesh.indices
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    e0 = (v0 - v1).norm(dim=1, p=2)
    e1 = (v1 - v2).norm(dim=1, p=2)
    e2 = (v2 - v0).norm(dim=1, p=2)
    avg_len = torch.mean(e0 + e1 + e2)/3
    loss = loss_function(e0, e1) + loss_function(e1, e2) + loss_function(e2, e0) \
           + loss_function(e0, avg_len) + loss_function(e1, avg_len)+ loss_function(e2, avg_len)

    return loss.mean()

def mesh_edge_loss(meshes, target_length: float = 0.0):
    """
    Computes mesh edge length regularization loss averaged across all meshes
    in a batch. Each mesh contributes equally to the final loss, regardless of
    the number of edges per mesh in the batch by weighting each mesh with the
    inverse number of edges. For example, if mesh 3 (out of N) has only E=4
    edges, then the loss for each edge in mesh 3 should be multiplied by 1/E to
    contribute to the final loss.

    Args:
        meshes: Meshes object with a batch of meshes.
        target_length: Resting value for the edge length.

    Returns:
        loss: Average loss across the batch. Returns 0 if meshes contains
        no meshes or all empty meshes.
    """
    N = len(meshes)
    edges_packed = meshes.edges_packed()  # (sum(E_n), 3)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    edge_to_mesh_idx = meshes.edges_packed_to_mesh_idx()  # (sum(E_n), )
    num_edges_per_mesh = meshes.num_edges_per_mesh()  # N

    # Determine the weight for each edge based on the number of edges in the
    # mesh it corresponds to.
    # TODO (nikhilar) Find a faster way of computing the weights for each edge
    # as this is currently a bottleneck for meshes with a large number of faces.
    weights = num_edges_per_mesh.gather(0, edge_to_mesh_idx)
    weights = 1.0 / weights.float()

    verts_edges = verts_packed[edges_packed]
    v0, v1 = verts_edges.unbind(1)
    loss = ((v0 - v1).norm(dim=1, p=2) - target_length) ** 2.0
    loss = loss * weights

    return loss.sum() / N