from __future__ import print_function
import torch.utils.data
from models.unet_attention import *
# from utils.core.depth2volume import *
# from utils.core.im_utils import get_plane_params

class BaseModule(nn.Module):
    def __init__(self, split_last=True):
        super(BaseModule, self).__init__()
        self.split_last = split_last
        self.img_encoder = ATUNet_Encoder_SMPLX(in_ch=6, out_ch=8)
        self.depth_branch = ATUNet_Decoder(in_ch=1024, out_ch=2)
        self.lbs_branch = ATUNet_Decoder(in_ch=1024, out_ch=6)
        # weight initialization
        for m in self.modules():
            m = weight_init_basic(m)

    def forward(self, x):
        f_nd = self.img_encoder(x)
        depth = self.depth_branch(f_nd)
        lbs = self.lbs_branch(f_nd)
        return {'pred_depth': depth,
                'pred_lbs': lbs}
        
class ColorModule(nn.Module):
    def __init__(self, split_last=True):
        super(ColorModule, self).__init__()
        self.split_last = split_last
        self.img_encoder = ATUNet_Encoder_SMPLX(in_ch=10, out_ch=8)
        self.color_branch = ATUNet_Decoder(in_ch=1024, out_ch=6)
        # weight initialization
        for m in self.modules():
            m = weight_init_basic(m)
            
    def forward(self, x):
        f_nd = self.img_encoder(x)
        color = self.color_branch(f_nd)
        return {'pred_color': color}
    
class JointRecon(nn.Module):
    def __init__(self, split_last=True):
        super(JointRecon, self).__init__()
        self.im2d = BaseModule(split_last=split_last)
        self.im2im = ColorModule()

    def forward(self, x, mode=1):
        pred_var = list()
        if mode == 1:
            pred_var.append(self.im2d.forward(x))
        else:
            pred_var.append(self.im2im.forward(x))
        # pred_var.append(self.im2d.forward(x))
        return pred_var

    def forward_nl_color(self, x):
        pred_var = list()
        pred_var.append(self.im2im.forward(x))
        return pred_var

def weight_init_basic(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m


# for test
if __name__ == '__main__':
    # input = Variable(torch.randn(4, 3, 256, 256)).float().cuda()
    input = Variable(torch.randn(4, 2, 5, 5)).float().cuda()
    _, b = torch.Tensor.chunk(input, chunks=2, dim=1)
    print(b.shape)

    print(b)
    print(input)

    # target = Variable(torch.randn(4, 3, 256, 256)).float().cuda()
    # print(len(input.shape))

    # tmp = [1, 2, 3]
    # print(tmp[-1])
    # model = Hourglass(4, '2D')
    # model = DeepHumanNet_A()
    # print(model.modules)
    # model.freeze()
    # model = VolumeDecoder (3, 128)
    # output = model.forward(input, target)

    # output, pre, post = model(input, None, None)
    # print((output - target))
