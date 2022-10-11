import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from networks import GatedConv2dWithActivation, GatedDeConv2dWithActivation, SNConvWithActivation, get_pad
from utils import gen_hole_area, crop

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation,with_attn=False):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        if self.with_attn:
            return out ,attention
        else:
            return out

class Concatenate(nn.Module):
    def __init__(self, dim=-1):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)

class InpaintSANet(torch.nn.Module):
    """
    Inpaint generator, input should be 5*256*256, where 3*256*256 is the masked image, 1*256*256 for mask, 1*256*256 is the guidence
    """
    def __init__(self, n_in_channel=5):
        super(InpaintSANet, self).__init__()
        cnum = 32
        self.coarse_net = nn.Sequential(
            #input is 5*256*256, but it is full convolution network, so it can be larger than 256
            GatedConv2dWithActivation(n_in_channel, cnum, 5, 1, padding=get_pad(256, 5, 1)),
            # downsample 128
            GatedConv2dWithActivation(cnum, 2*cnum, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(2*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            #downsample to 64
            GatedConv2dWithActivation(2*cnum, 4*cnum, 4, 2, padding=get_pad(128, 4, 2)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            # atrous convlution
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            #Self_Attn(4*cnum, 'relu'),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            # upsample
            GatedDeConv2dWithActivation(2, 4*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            #Self_Attn(2*cnum, 'relu'),
            GatedConv2dWithActivation(2*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedDeConv2dWithActivation(2, 2*cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),

            GatedConv2dWithActivation(cnum, cnum//2, 3, 1, padding=get_pad(256, 3, 1)),
            #Self_Attn(cnum//2, 'relu'),
            GatedConv2dWithActivation(cnum//2, 3, 3, 1, padding=get_pad(128, 3, 1), activation=None)
        )

        self.refine_conv_net = nn.Sequential(
            # input is 5*256*256
            GatedConv2dWithActivation(n_in_channel, cnum, 5, 1, padding=get_pad(256, 5, 1)),
            # downsample
            GatedConv2dWithActivation(cnum, cnum, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            # downsample
            GatedConv2dWithActivation(2*cnum, 2*cnum, 4, 2, padding=get_pad(128, 4, 2)),
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            #Self_Attn(4*cnum, 'relu'),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),

            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))
        )
        self.refine_attn = Self_Attn(4*cnum, 'relu', with_attn=False)
        self.refine_upsample_net = nn.Sequential(
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),

            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedDeConv2dWithActivation(2, 4*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedConv2dWithActivation(2*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedDeConv2dWithActivation(2, 2*cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),

            GatedConv2dWithActivation(cnum, cnum//2, 3, 1, padding=get_pad(256, 3, 1)),
            #Self_Attn(cnum//2, 'relu'),
            GatedConv2dWithActivation(cnum//2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None),
        )


    def forward(self, imgs, masks, img_exs=None):
        # Coarse
        masked_imgs =  imgs * (1 - masks) + masks
        if img_exs == None:
            input_imgs = torch.cat([masked_imgs, masks, torch.full_like(masks, 1.)], dim=1)
        else:
            input_imgs = torch.cat([masked_imgs, img_exs, masks, torch.full_like(masks, 1.)], dim=1)
        #print(input_imgs.size(), imgs.size(), masks.size())
        x = self.coarse_net(input_imgs)
        x = torch.clamp(x, -1., 1.)
        coarse_x = x
        # Refine
        masked_imgs = imgs * (1 - masks) + coarse_x * masks
        if img_exs is None:
            input_imgs = torch.cat([masked_imgs, masks, torch.full_like(masks, 1.)], dim=1)
        else:
            input_imgs = torch.cat([masked_imgs, img_exs, masks, torch.full_like(masks, 1.)], dim=1)
        x = self.refine_conv_net(input_imgs)
        x= self.refine_attn(x)
        #print(x.size(), attention.size())
        x = self.refine_upsample_net(x)
        x = torch.clamp(x, -1., 1.)
        return coarse_x, x

class InpaintSADirciminator(nn.Module):
    def __init__(self):
        super(InpaintSADirciminator, self).__init__()
        cnum = 32
        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(5, 2*cnum, 4, 2, padding=get_pad(256, 5, 2)),
            SNConvWithActivation(2*cnum, 4*cnum, 4, 2, padding=get_pad(128, 5, 2)),
            SNConvWithActivation(4*cnum, 8*cnum, 4, 2, padding=get_pad(64, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(32, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(16, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(8, 5, 2)),
            Self_Attn(8*cnum, 'relu'),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(4, 5, 2)),
        )
        self.linear = nn.Linear(8*cnum*2*2, 1)

    def forward(self, input):
        x = self.discriminator_net(input)
        x = x.view((x.size(0),-1))
        #x = self.linear(x)
        return x

class LocalDiscriminator(nn.Module):
    def __init__(self):
        super(LocalDiscriminator, self).__init__()
        cnum = 32
        self.output_shape = (1024,)
        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(5, 2*cnum, 4, 2, padding=get_pad(128, 5, 2)),
            SNConvWithActivation(2*cnum, 4*cnum, 4, 2, padding=get_pad(64, 5, 2)),
            SNConvWithActivation(4*cnum, 8*cnum, 4, 2, padding=get_pad(32, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(16, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(8, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(4, 5, 2)),
        )
        self.linear = nn.Linear(8*cnum*2*2, 1)

    def forward(self, input):
        x = self.discriminator_net(input)
        x = x.view((x.size(0),-1))
        #x = self.linear(x)
        return x

class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super(GlobalDiscriminator, self).__init__()
        cnum = 32
        self.output_shape = (1024,)
        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(5, 2*cnum, 4, 2, padding=get_pad(256, 5, 2)),
            SNConvWithActivation(2*cnum, 4*cnum, 4, 2, padding=get_pad(128, 5, 2)),
            SNConvWithActivation(4*cnum, 8*cnum, 4, 2, padding=get_pad(64, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(32, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(16, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(8, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(4, 5, 2)),
        )
        self.linear = nn.Linear(8*cnum*2*2, 1)

    def forward(self, input):
        x = self.discriminator_net(input)
        x = x.view((x.size(0),-1))
        #x = self.linear(x)
        return x

class MultiDiscriminator(nn.Module):
    def __init__(self, local_dis_size, random_bbox_size, dis_mode='cat'):
        super(MultiDiscriminator, self).__init__()
        self.model_ld = LocalDiscriminator()
        self.model_gd = GlobalDiscriminator()
        self.local_size = local_dis_size
        self.mask_size = random_bbox_size
        self.dis_mode = dis_mode
        # input_shape: [(None, 1024), (None, 1024)]
        in_features = self.model_ld.output_shape[-1] + self.model_gd.output_shape[-1]
        self.concat = Concatenate(dim=-1)
        # input_shape: (None, 2048)
        self.linear1 = nn.Linear(in_features, 1)
        self.linear2 = nn.Linear(self.model_ld.output_shape[-1], 1)
        self.linear3 = nn.Linear(self.model_gd.output_shape[-1], 1)
        self.act = nn.Sigmoid()
        # output_shape: (None, 1)

    def forward(self, x):
        x_gd = x
        ld_area = gen_hole_area(self.local_size, self.mask_size)
        x_ld = crop(x, ld_area)
        x_ld = self.model_ld(x_ld)
        x_gd = self.model_gd(x_gd)
        if self.dis_mode == 'cat':
            out = self.act(self.linear1(self.concat([x_ld, x_gd])))
        elif self.dis_mode == 'mix':
            out = self.act(self.linear2(x_ld)) + self.act(self.linear3(x_ld))
        else:
            raise NotImplementedError('dis_mode [%s] is not implemented', self.dis_mode)
        return out

if __name__ == '__main__':
    a = torch.randn(32,5,256,256)
    multidis = MultiDiscriminator([128,128],[128,128],'mix')
    print(multidis.dis_mode)
    out = multidis(a)
    print(out.shape)
