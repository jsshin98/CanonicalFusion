from .laplacian import laplacian_loss
from .mask import mask_loss, side_loss
from .normal_consistency import normal_consistency_loss, normal_consistency_loss_l2
from .shading import shading_loss, normal_map_loss, offset_map_loss
from .smoothing import mesh_smoothness_loss
from .edge_regularizer import mesh_face_loss
from .color import color_loss
from .chamfer import chamfer_loss
from .lbs import lbs_regularizer