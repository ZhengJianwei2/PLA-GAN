from ast import Global
from inspect import getargvalues
from opcode import stack_effect
from re import S
from xml.sax import SAXNotRecognizedException
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
from model.PLAGAN.ChannelAttention import channelAtten
import math
from model.PLAGAN import NLSAttention

##########################################################################
# Basic modules
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


def conv_down(in_chn, out_chn, bias=False):
    convmodel1 = nn.Sequential(     
                nn.MaxPool2d(2,2),
                nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias),
        )

    return convmodel1


def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)


class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y 

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,bias=True, bn=False, res_scale=1):

        super(ResBlock, self).__init__()
        act = RELU_()
        m = []
        for i in range(2):
            if i == 0:
                m.append(conv(n_feats, 64, kernel_size, bias=bias))
            else:
                m.append(conv(64, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class RELU_(nn.Module):
    def forward(self, x):
        return F.relu(x)

##########################################################################
## U-Net    
class U_Net(nn.Module):
    def __init__(self,in_channel, n_feat):
        super(U_Net, self).__init__()
        self.convhead = nn.Conv2d(in_channel, n_feat, 3, 1, 1)
        self.convtail = nn.Conv2d(n_feat, in_channel, 3, 1, 1)
        self.convdec1 = nn.Conv2d(4*n_feat, 2*n_feat, 3, 1, 1)
        self.convdec2 = nn.Conv2d(2*n_feat, n_feat, 3, 1, 1)
        self.conv1_1 = nn.Conv2d(4*n_feat, 2*n_feat, 1, 1, 0)
        self.conv1_2 = nn.Conv2d(2*n_feat, n_feat, 1, 1, 0)

        self.bn1 =  nn.BatchNorm2d(n_feat)
        self.bn2 =  nn.BatchNorm2d(2*n_feat)
        self.bn3 =  nn.BatchNorm2d(2*n_feat)
        self.bn4 =  nn.BatchNorm2d(n_feat)

        self.NLS =  NLSAttention.NonLocalSparseAttention(
                    channels=n_feat, chunk_size=144, n_hashes=4, reduction=4, res_scale=1)
        self.NLS2 = NLSAttention.NonLocalSparseAttention(
                    channels=2*n_feat, chunk_size=144, n_hashes=4, reduction=4, res_scale=1)
        self.NLS3 = NLSAttention.NonLocalSparseAttention(
                    channels=4*n_feat, chunk_size=144, n_hashes=4, reduction=4, res_scale=1)

        self.NLS4 = NLSAttention.NonLocalSparseAttention(
                     channels=2*n_feat, chunk_size=144, n_hashes=4, reduction=4, res_scale=1)
        self.NLS5 = NLSAttention.NonLocalSparseAttention(
                     channels=n_feat, chunk_size=144, n_hashes=4, reduction=4, res_scale=1)

        self.cdown1 = conv_down(n_feat,2*n_feat, bias=False)
        self.cdown2 = conv_down(2*n_feat,4*n_feat, bias=False)

        # self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.deconv1 = nn.ConvTranspose2d(4*n_feat,2*n_feat, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(2*n_feat, n_feat, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.act = RELU_()
    def forward(self, x):
        # encoder
        x_head = self.convhead(x)
        x_head = self.bn1(x_head)
        x_head = self.act(x_head)
        x_nls1 = self.NLS(x_head)
        x_encdown1 = self.cdown1(x_nls1)
        x_encdown1 = self.bn2(x_encdown1)
        x_encdown1 = self.act(x_encdown1)
        x_nls2 = self.NLS2(x_encdown1)
        x_encdown2 = self.cdown2(x_nls2)
        x_nls3 = self.NLS3(x_encdown2)
        
        # decoder
        x_dec1 = self.deconv1(x_nls3)
        x_concat1 = torch.cat((x_dec1, x_nls2),1)
        x_decnls1 = self.conv1_1(x_concat1)
        x_decnls1 = self.NLS2(x_decnls1)
        x_decnls1 = self.bn3(x_decnls1)
        x_decnls1 = self.act(x_decnls1)
        x_dec2 = self.deconv2(x_decnls1)
        x_concat2 = torch.cat((x_dec2, x_nls1),1)
        x_decnls2 = self.conv1_2(x_concat2)
        x_decnls2 = self.bn4(x_decnls2)
        x_decnls2 = self.act(x_decnls2)
        x_tail = self.convtail(x_decnls2)

        out = x_tail + x
        return out 

class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 1, kernel_size, bias=bias)

    def forward(self, x):
        x1 = self.conv1(x)
        img = self.conv2(x1) 
        return  img  
            
##########################################################################
class Block(nn.Module):
    def __init__(self, in_channel = 1, n_feat = 64, l1_feat=64, kernel_size =3, reduction =4, size = 64,bias=False):
        super(Block, self).__init__()
        act = RELU_()
        # self.weight=nn.Parameter(torch.zeros(1))
        self.convmodel1 = nn.Sequential(
            nn.Conv2d(in_channel, l1_feat, 3, 1, 1),
            act,
            nn.Conv2d(l1_feat, l1_feat, 3, 1, 1),
        )
        self.convmodel2 = nn.Sequential(
            nn.Conv2d(l1_feat, l1_feat, 3, 1, 1),
            act,
            nn.Conv2d(l1_feat, in_channel, 3, 1, 1),
        )


        self.A_0 = ResBlock(default_conv,in_channel,3)   # A 
        self.At_0 = ResBlock(default_conv,in_channel,3)  # AT 
        self.G_1 = ResBlock(default_conv,in_channel,3)   # G
        self.Gt_1 = ResBlock(default_conv,in_channel,3)  #  GT
        
        self.q_1 = nn.Parameter(0.02*torch.ones(size,size)).cuda()
        self.q_2 = nn.Parameter(0.02**torch.ones(size,size)).cuda()

        self.chanatten = channelAtten(l1_feat)

        self.alpha = nn.Parameter(torch.Tensor([0.2])) 
        self.beta = nn.Parameter(torch.Tensor([1.05]))

        self.U_net = U_Net(in_channel,n_feat)
        self.shallow_feat2 = nn.Sequential(conv(1, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.sam = SAM(n_feat, kernel_size=1, bias=bias)

    def forward(self,Stage_img_x,stage_img_J,input_img,fai):
        # update X
        x_p1 = self.At_0(self.A_0(Stage_img_x)-input_img) + self.alpha * self.Gt_1(self.G_1(Stage_img_x) - stage_img_J + fai*(1/self.alpha))
        # x_tmp = self.t_1(Stage_img_x)
        x_k1 = Stage_img_x - ( self.q_1*x_p1 )
        
        x_k=  self.convmodel1(x_k1)
        x_k=  self.chanatten(x_k)
        x_k = self.convmodel2(x_k)
        out_x =x_k+x_k1 

        # update J
        J_p2 = self.alpha * (stage_img_J - self.G_1(out_x) - fai*(1/self.alpha) )

        J_k = stage_img_J - ( self.q_2*J_p2  )
        out_J =  self.U_net(J_k)
        # update fai
        fai = fai + self.alpha * self.beta* (self.G_1(out_x) - out_J)
        
        return out_x, out_J, fai


## PLAGAN
class Net(nn.Module):
    def __init__(self,**kwargs):
        super(Net, self).__init__()
        self.iternum=kwargs['blocknum']
        self.imgsize=kwargs['imgsize']
        self.in_channel=kwargs['in_channel']
        self.n_feat = kwargs['n_feat']
        Blocklist=[]
        for i in range(0, self.iternum):
            Blocklist.append(Block(in_channel=self.in_channel , n_feat= self.n_feat, size=self.imgsize ))
        self.Blocklist = nn.ModuleList(Blocklist)

        self.conv_tail1 = nn.Conv2d(self.in_channel, self.in_channel, 3, 1, 1)
        self.conv_tail2 = nn.Conv2d(self.in_channel, self.in_channel, 3, 1, 1)
        self.conv_tail2.weight.data = torch.permute(self.conv_tail1.weight.data,(0,1,3,2))
        self.mu =nn.Parameter(torch.ones(1)).cuda()
        self.E = nn.Parameter(torch.zeros(1))

    def forward(self, img):
        outputlist_x=[]
        outputlist_j=[]

        outputlist_x.append(img)
        outputlist_j.append(img)
        b,c,h,w=img.shape
        fai=torch.zeros([b,c,h,w]).cuda()
        for layer_idx in range(self.iternum):
            output_x, output_j, fai = self.Blocklist[layer_idx](outputlist_x[layer_idx],outputlist_j[layer_idx],img, fai)
            outputlist_x.append(output_x)
            outputlist_j.append(output_j)

        X_K = outputlist_x[-1]
        J_K = outputlist_j[-1]
        x_hat = self.mu*(X_K + self.conv_tail1(X_K)) + (self.E-self.mu)*(J_K+self.conv_tail2(J_K))

        outputlist_x.append(x_hat)
        # return x_hat
        return outputlist_x[-1]

class discriminator(nn.Module):
    def __init__(self, dim=32, stage=3):
        super(discriminator, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.in_proj = nn.Conv2d(1, self.dim, 3, 1, 1, bias=False)


        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                NLSAttention.NonLocalSparseAttention(
                    channels=dim_stage, chunk_size=144, n_hashes=4, reduction=4, res_scale=1),
                NLSAttention.NonLocalSparseAttention(
                    channels=dim_stage, chunk_size=144, n_hashes=4, reduction=4, res_scale=1),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            ]))
            dim_stage *= 2


        # Output projection
        self.out_proj= nn.Conv2d(dim_stage, 1, kernel_size=3, stride=1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        # Input projection
        fea = self.lrelu(self.in_proj(x))

        # Encoder
        fea_encoder = []  # [c 2c 4c 8c]
        for (NLS1,NLS2, DownSample) in self.encoder_layers:
            fea1 = NLS1(fea)
            fea2 = NLS2(fea1)  
            fea_encoder.append(fea2)
            fea = DownSample(fea2)

        fea = self.out_proj(fea)

        fea = fea.view(fea.shape[0], -1)

        return fea
