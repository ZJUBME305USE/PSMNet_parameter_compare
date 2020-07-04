from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

use_dilation = True

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
                         nn.BatchNorm3d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:   # 当输入与输出size不一样或者通道数不一样时，1*1的2D卷积
            x = self.downsample(x)

        out += x

        return out


class matchshifted(nn.Module):
    def __init__(self):
        super(matchshifted, self).__init__()

    def forward(self, left, right, shift):
        batch, filters, height, width = left.size()
        shifted_left = F.pad(torch.index_select(left,  3, Variable(torch.LongTensor([i for i in range(shift,width)])).cuda()),(shift,0,0,0))
        shifted_right = F.pad(torch.index_select(right, 3, Variable(torch.LongTensor([i for i in range(width-shift)])).cuda()),(shift,0,0,0))
        out = torch.cat((shifted_left,shifted_right),1).view(batch,filters*2,1,height,width)
        return out


class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])).cuda(), requires_grad=False)

    def forward(self, x):
        # x是batch*disp*height*width的tensor
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x*disp, 1)
        return out


class reprojection(nn.Module):

    def __init__(self):
        super(reprojection, self).__init__()

    def forward(self, disp, Q):
        b, h, w = disp.shape
        xx = torch.range(0, w - 1).cuda()
        yy = torch.range(0, h - 1).cuda()
        Y, X = torch.meshgrid([yy, xx])
        YY = torch.empty(b, h, w).cuda()
        YY[:, :, :] = Y
        XX = torch.empty(b, h, w).cuda()
        XX[:, :, :] = X
        ww = torch.ones([b, h, w]).cuda()
        Disp = torch.stack((XX, YY, disp, ww), dim=1)
        Disp = Disp.view([b, 4, h * w])
        del xx
        del yy
        del Y
        del X
        del YY
        del XX
        del ww
        Q = torch.tensor(Q, dtype=torch.float32).cuda()
        depth = torch.matmul(Q, Disp)
        W = depth[:, 3, :]
        depth = depth.permute([1, 0, 2])
        depth = torch.div(depth, W)
        depth = depth.permute([1, 0, 2])
        depth = depth.view([b, 4, h, w])
        depth = depth[:, 0:3, :, :]
        # depth = depth.permute([0, 2, 3, 1])     # 输出是b*3*h*w
        # depth = torch.zeros_like(disp).cuda()    # 此时depth和disp都是b*h*w，只考虑z方向
        # b, h, w = disp.shape
        # for i in range(b):
        #     depth[i, :, :] = (bb[i] * ff[i])/disp[i, :, :]
        return depth


class feature_extraction(nn.Module):
    def __init__(self, use_dilation, use_ssp):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.use_dilation = use_dilation
        self.use_ssp = use_ssp
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),   # conv0_1/conv0_2/conv0_3
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)    # conv1_x
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)   # conv2_x
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)   # conv3_x, 但论文中dila=2，这里是1
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)   # conv4_x，但论文中dila=4，这里是2

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))      # 4个分支，得到不同size的输出

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),    # concate之后通道数是320，是layer2和layer4的输出以及各个branch连接后的feature map
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))  # 最后的feature维度是32

        self.lastconv1 = nn.Sequential(convbn(128, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        if self.use_dilation == False:
            dilation = 1
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        if self.use_ssp == True:
            output_branch1 = self.branch1(output_skip)
            output_branch1 = F.upsample(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

            output_branch2 = self.branch2(output_skip)
            output_branch2 = F.upsample(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

            output_branch3 = self.branch3(output_skip)
            output_branch3 = F.upsample(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

            output_branch4 = self.branch4(output_skip)
            output_branch4 = F.upsample(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

            output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        # 将SPP各个分支（平均池化的结果）与原来两个中间feature map连接，则现在的feature map 既有高频特征（两种）又有低频特征（四种）
            output_feature = self.lastconv(output_feature)

        if self.use_ssp == False:
            output_feature = self.lastconv1(output_feature)

        return output_feature



