import torch
from typing import Dict, List

from diff_renderer.normal_nds.nds.core import View
from torchvision import transforms
import matplotlib.pyplot as plt
from pytorch3d.loss import chamfer_distance

def lbs_regularizer(lbs, ones, loss_function=torch.nn.MSELoss()):
    """ Compute the mask term as the mean difference between the original masks and the rendered masks.

    Args:
        views (List[View]): Views with masks
        gbuffers (List[Dict[str, torch.Tensor]]): G-buffers for each view with the 'mask' channel
        loss_function (Callable): Function for comparing the masks or generally a set of pixels
    """

    loss = loss_function(torch.sum(lbs, 1), ones)
    return loss
