import torch
from typing import Dict, List

from diff_renderer.normal_nds.nds.core import View
from torchvision import transforms
import matplotlib.pyplot as plt
from pytorch3d.loss import chamfer_distance

def chamfer_loss(mesh_vert, smpl_vert):
    """ Compute the mask term as the mean difference between the original masks and the rendered masks.

    Args:
        views (List[View]): Views with masks
        gbuffers (List[Dict[str, torch.Tensor]]): G-buffers for each view with the 'mask' channel
        loss_function (Callable): Function for comparing the masks or generally a set of pixels
    """

    loss = chamfer_distance(mesh_vert, smpl_vert)[0]
    return loss
