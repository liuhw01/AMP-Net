import torch.nn as nn
import torch
from torch.nn import functional as F
from attention import CBAM
from attention import CBAM_O
from attention import OcclusionGate
import numpy as np
import torchvision.ops as ops
from OSA import _OSA_module
import torchsnooper


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 预处理模块
class BasicBlock(nn.Module):
    # 实现子module：Residual Block
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


# 自适应分支 注意力模块 (需改)
class CSO_AttentionBlock(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CSO_AttentionBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.cbam = CBAM(planes, 16)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


# 局部分支 注意力模块 (需改)
class CO_AttentionBlock(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CO_AttentionBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.cbam = CBAM_O(planes, 16)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


#  全局分支 OSA
class _OSA_stage(nn.Sequential):
    def __init__(
            self,
            in_ch,
            stage_ch,
            concat_ch,
            block_per_stage,
            layer_per_block,
            stage_num, SE=False,
            depthwise=False):

        super(_OSA_stage, self).__init__()

        if not stage_num == 2:
            self.add_module("Pooling", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        if block_per_stage != 1:
            SE = False
        module_name = f"OSA{stage_num}_1"
        self.add_module(
            module_name, _OSA_module(in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE, identity=False,
                                     depthwise=depthwise)
        )
        for i in range(block_per_stage - 1):
            if i != block_per_stage - 2:  # last block
                SE = False
            module_name = f"OSA{stage_num}_{i + 2}"
            self.add_module(
                module_name,
                _OSA_module(
                    concat_ch, stage_ch, concat_ch, layer_per_block, module_name, SE, identity=True, depthwise=depthwise
                ),
            )


#@torchsnooper.snoop()
class MCNet(nn.Module):
    # rect=torch.tensor(np.random.rand(16,16)).cuda()
    def __init__(self, block_b, block_a, block_o, rect, rect_local, num_classes):
        super(MCNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 预处理模块
        self.layer1 = self._make_layer(block_b, 64, 64, 3)  # 56x56x64
        self.layer2 = self._make_layer(block_b, 64, 128, 4, stride=2)  # 第一个stride=2,剩下3个stride=1;28x28x128

        # 自适应分子模块
        # rect.shape=[x.shape[0],16]
        self.rect = rect
        self.layer3_1_1 = self._make_layer(block_a, 128, 256, 2, stride=2)
        self.layer3_2_1 = self._make_layer(block_a, 256, 512, 2, stride=1)

        self.layer3_1_2 = self._make_layer(block_a, 128, 256, 2, stride=2)
        self.layer3_2_2 = self._make_layer(block_a, 256, 512, 2, stride=1)

        self.layer3_1_3 = self._make_layer(block_a, 128, 256, 2, stride=2)
        self.layer3_2_3 = self._make_layer(block_a, 256, 512, 2, stride=1)

        self.layer3_1_4 = self._make_layer(block_a, 128, 256, 2, stride=2)
        self.layer3_2_4 = self._make_layer(block_a, 256, 512, 2, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_1 = nn.Linear(2048, 512)
        self.fc_1_1 = nn.Linear(512, num_classes)
        # 局部模块
        self.rect_local = rect_local
        self.layer4_1_1 = self._make_layer(block_a, 128, 256, 2, stride=2)
        self.layer4_2_1 = self._make_layer(block_a, 256, 512, 2, stride=1)
        self.layer4_1_2 = self._make_layer(block_a, 128, 256, 2, stride=2)
        self.layer4_2_2 = self._make_layer(block_a, 256, 512, 2, stride=1)
        self.layer4_1_3 = self._make_layer(block_a, 128, 256, 2, stride=2)
        self.layer4_2_3 = self._make_layer(block_a, 256, 512, 2, stride=1)
        self.layer4_1_4 = self._make_layer(block_a, 128, 256, 2, stride=2)
        self.layer4_2_4 = self._make_layer(block_a, 256, 512, 2, stride=1)
        self.layer4_1_5 = self._make_layer(block_a, 128, 256, 2, stride=2)
        self.layer4_2_5 = self._make_layer(block_a, 256, 512, 2, stride=1)

        self.Occlusionlayer = OcclusionGate(256)
        self.Occlusionlayer1 = OcclusionGate(512)
        self.fc_2 = nn.Linear(256 * 5, 512)
        self.fc_2_1 = nn.Linear(512, num_classes)

        self.fc_4 = nn.Linear(512 * 4 + 256 * 5, 512)
        self.fc_4_1 = nn.Linear(512, num_classes)
        self.fc_5 = nn.Linear(512 * 2, num_classes)
        # 全局模块
        # OSA stages
        self.in_ch_list = [128, 256, 384]
        self.config_concat_ch = [256, 384, 512]
        self.config_stage_ch = [128, 144, 160]
        self.block_per_stage = [1, 1, 1]
        self.layer_per_block = 3

        self.stage_names = []
        for i in range(3):  # num_stages
            name = "stage%d" % (i + 2)  # stage 2 ... stage 5
            self.stage_names.append(name)
            self.add_module(
                name,
                _OSA_stage(self.in_ch_list[i], self.config_stage_ch[i], self.config_concat_ch[i],
                           self.block_per_stage[i], self.layer_per_block, i + 2, SE=True, depthwise=False, ), )
        self.fc_3 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_rect(self, rect):
        self.rect = rect

    def get_rect(self):
        return self.rect

    def set_rect_local(self, rect_local):
        self.rect_local = rect_local

    def get_rect_local(self):
        return self.rect_local

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(conv1x1(inplanes, planes, stride), norm_layer(planes))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 56x56x64 残差块 预处理 完毕
        x = self.layer1(x)  # 56x56x64
        x = self.layer2(x)  # 28x28x128

        ######## Adaptive module 输入 28*28*128  修改1. attention  2. 获取rect

        patch_11 = torch.zeros(x.shape[0], x.shape[1], 14, 14).cuda()
        patch_12 = torch.zeros(x.shape[0], x.shape[1], 14, 14).cuda()
        patch_21 = torch.zeros(x.shape[0], x.shape[1], 14, 14).cuda()
        patch_22 = torch.zeros(x.shape[0], x.shape[1], 14, 14).cuda()
        # x[:, :, 0:14, 0:14].clone()
        # patch_12 = x[:, :, 0:14, 14:28].clone()
        # patch_21 = x[:, :, 14:28, 0:14].clone()

        # patch_22 = x[:, :, 14:28, 14:28].clone()

        for i in range(len(x)):
            #print(i)
            if self.rect[i][2] - self.rect[i][0] == 14:
                patch_11[i] = x[i][:, self.rect[i][1]:self.rect[i][3], self.rect[i][0]:self.rect[i][2]].clone()
                patch_12[i] = x[i][:, self.rect[i][5]:self.rect[i][7], self.rect[i][6]:self.rect[i][4]].clone()
            if self.rect[i][2] - self.rect[i][0] != 14:
                padding = 14 - (self.rect[i][2] - self.rect[i][0])
                pad_11 = nn.ZeroPad2d(padding=(padding, 0, padding, 0))
                path_11_min = x[i][:, self.rect[i][1]:self.rect[i][3], self.rect[i][0]:self.rect[i][2]].clone()
                patch_11[i] = pad_11(path_11_min)
                pad_12 = nn.ZeroPad2d(padding=(0, padding, padding, 0))
                path_12_min = x[i][:, self.rect[i][5]:self.rect[i][7], self.rect[i][6]:self.rect[i][4]].clone()
                patch_12[i] = pad_12(path_12_min)

            if self.rect[i][10] - self.rect[i][8] == 14:
                patch_21[i] = x[i][:, self.rect[i][11]:self.rect[i][9], self.rect[i][8]:self.rect[i][10]].clone()
                patch_22[i] = x[i][:, self.rect[i][15]:self.rect[i][13], self.rect[i][14]:self.rect[i][12]].clone()
            if self.rect[i][10] - self.rect[i][8] != 14:
                padding = 14 - (self.rect[i][10] - self.rect[i][8])
                pad_21 = nn.ZeroPad2d(padding=(padding, 0, 0, padding))
                path_21_min = x[i][:, self.rect[i][11]:self.rect[i][9], self.rect[i][8]:self.rect[i][10]].clone()

                patch_21[i] = pad_21(path_21_min)
                pad_22 = nn.ZeroPad2d(padding=(0, padding, 0, padding))
                path_22_min = x[i][:, self.rect[i][15]:self.rect[i][13], self.rect[i][14]:self.rect[i][12]].clone()
                patch_22[i] = pad_22(path_22_min)

        # indx = torch.tensor(range(self.rect.shape[0]), dtype=torch.float).view(-1, 1).cuda()
        # print(self.rect.shape)
        # top_left = self.rect[:, 0:4].float().cuda()
        # top_right = self.rect[:, 4:8].float().cuda()
        # bottom_left = self.rect[:, 8:12].float().cuda()
        # bottom_right = self.rect[:, 12:16].float().cuda()
        # top_left = ops.roi_align(x, torch.cat([indx, top_left], dim=1), [12, 12], spatial_scale=28. / 224.,
        #                         sampling_ratio=-1)
        # top_right = ops.roi_align(x, torch.cat([indx, top_right], dim=1), [12, 12], spatial_scale=28. / 224.,
        #                         sampling_ratio=-1)
        # bottom_left = ops.roi_align(x, torch.cat([indx, bottom_left], dim=1), [12, 12], spatial_scale=28. / 224.,
        #                    sampling_ratio=-1)
        # bottom_right = ops.roi_align(x, torch.cat([indx, bottom_right], dim=1), [12, 12], spatial_scale=28. / 224.,
        #                     sampling_ratio=-1)

        top_left_out = self.layer3_1_1(patch_11)
        top_left_out = self.layer3_2_1(top_left_out)

        top_right_out = self.layer3_1_2(patch_12)
        top_right_out = self.layer3_2_2(top_right_out)

        bottom_left_out = self.layer3_1_3(patch_21)
        bottom_left_out = self.layer3_2_3(bottom_left_out)

        bottom_right_out = self.layer3_1_4(patch_22)
        bottom_right_out = self.layer3_2_4(bottom_right_out)

        # branch_1_out_1 = torch.cat([top_left_out, top_right_out], dim=3)
        # branch_1_out_2 = torch.cat([bottom_left_out, bottom_right_out], dim=3)
        # branch_1_out = torch.cat([branch_1_out_1, branch_1_out_2], dim=2)

        # branch_1_out = self.avgpool(branch_1_out)
        # branch_1_out = torch.flatten(branch_1_out, 1)
        # brand_1_out = self.fc_1_1(branch_1_out)

        # fc
        top_left_out_fc = self.avgpool(top_left_out)
        top_left_out_fc = torch.flatten(top_left_out_fc, 1)

        top_right_out_fc = self.avgpool(top_right_out)
        top_right_out_fc = torch.flatten(top_right_out_fc, 1)

        bottom_left_out_fc = self.avgpool(bottom_left_out)
        bottom_left_out_fc = torch.flatten(bottom_left_out_fc, 1)

        bottom_right_out_fc = self.avgpool(bottom_right_out)
        bottom_right_out_fc = torch.flatten(bottom_right_out_fc, 1)

        # top_left_out_fc=self.Occlusionlayer1(top_left_out)
        # top_right_out_fc = self.Occlusionlayer1(top_right_out)
        # bottom_left_out_fc = self.Occlusionlayer1(bottom_left_out)
        # bottom_right_out_fc = self.Occlusionlayer1(bottom_right_out)

        # 自适应特征合并 512*4→512→7

        brand_1_out = torch.cat([top_left_out_fc, top_right_out_fc, bottom_left_out_fc, bottom_right_out_fc], dim=1)
        brand_1_out = self.fc_1(brand_1_out)
        #brand_1_out = self.fc_1_1(brand_1_out)
        # brand_1_out1 = self.fc_1_1(top_left_out_fc)
        # brand_2_out1 = self.fc_1_1(top_right_out_fc)
        # brand_3_out1 = self.fc_1_1(bottom_left_out_fc)
        # brand_4_out1 = self.fc_1_1(bottom_right_out_fc)




        ################## 局部特征提取
        rang = self.rect_local[0][1] - self.rect_local[0][0]
        eye1 = x[:, :, 0:rang, 0:rang].clone()
        eye2 = x[:, :, 0:rang, 0:rang].clone()
        eye_midd = x[:, :, 0:rang, 0:rang].clone()
        mouth1 = x[:, :, 0:rang, 0:rang].clone()
        mouth2 = x[:, :, 0:rang, 0:rang].clone()

        for i in range(x.shape[0]):
            eye1[i] = x[i][:, self.rect_local[i][2]:self.rect_local[i][3], self.rect_local[i][0]:self.rect_local[i][1]]
            eye2[i] = x[i][:, self.rect_local[i][6]:self.rect_local[i][7], self.rect_local[i][4]:self.rect_local[i][5]]
            eye_midd[i] = x[i][:, self.rect_local[i][10]:self.rect_local[i][11],
                          self.rect_local[i][8]:self.rect_local[i][9]]
            mouth1[i] = x[i][:, self.rect_local[i][14]:self.rect_local[i][15],
                        self.rect_local[i][12]:self.rect_local[i][13]]
            mouth2[i] = x[i][:, self.rect_local[i][18]:self.rect_local[i][19],
                        self.rect_local[i][16]:self.rect_local[i][17]]

        # 128→256
        eye1_out = self.layer4_1_1(eye1)
        # eye1_out=self.layer4_2_1(eye1_out)
        eye2_out = self.layer4_1_2(eye2)
        # eye2_out=self.layer4_2_2(eye2_out)
        eye_midd_out = self.layer4_1_3(eye_midd)
        # eye_midd_out = self.layer4_2_3(eye_midd_out)
        mouth_out1 = self.layer4_1_4(mouth1)
        # mouth_out1 = self.layer4_2_4(mouth_out1)
        mouth_out2 = self.layer4_1_5(mouth2)
        # mouth_out2 = self.layer4_2_5(mouth_out2)

        # fc
        eye1_out_fc = self.avgpool(eye1_out)
        eye1_out_fc = torch.flatten(eye1_out_fc, 1)

        eye2_out_fc = self.avgpool(eye2_out)
        eye2_out_fc = torch.flatten(eye2_out_fc, 1)

        eye_midd_out_fc = self.avgpool(eye_midd_out)
        eye_midd_out_fc = torch.flatten(eye_midd_out_fc, 1)

        mouth1_out_fc = self.avgpool(mouth_out1)
        mouth1_out_fc = torch.flatten(mouth1_out_fc, 1)

        mouth2_out_fc = self.avgpool(mouth_out2)
        mouth2_out_fc = torch.flatten(mouth2_out_fc, 1)
        # 256*5→ 7

        # fc
        # eye1_out_fc = self.Occlusionlayer(eye1_out)

        # eye2_out_fc = self.Occlusionlayer(eye2_out)

        # eye_midd_out_fc = self.Occlusionlayer(eye_midd_out)

        # mouth1_out_fc = self.Occlusionlayer(mouth_out1)

        # mouth2_out_fc = self.Occlusionlayer(mouth_out2)

        # jubu 特征合并 256*5→512→7
        brand_2_out = torch.cat([eye1_out_fc, eye2_out_fc, eye_midd_out_fc, mouth1_out_fc, mouth2_out_fc], dim=1)
        brand_2_out = self.fc_2(brand_2_out)
        #brand_2_out = self.fc_2_1(brand_2_out)

        brand_out = torch.cat([brand_1_out, brand_2_out], dim=1)
        #brand_out=brand_1_out
        brand_out = self.fc_5(brand_out)

        # 全局特征

        for i in range(len(self.stage_names)):
            if i == 0:
                glabl_out = getattr(self, self.stage_names[i])(x)
            if i >= 1:
                glabl_out = getattr(self, self.stage_names[i])(glabl_out)

        glabl_out_fc = self.avgpool(glabl_out)
        glabl_out_fc = torch.flatten(glabl_out_fc, 1)

        brand_3_out = self.fc_3(glabl_out_fc)

        # brand_3_out = torch.cat([glabl_out_fc, brand_2_out], dim=1)
        # brand_3_out = self.fc_5(brand_3_out)

        return  brand_out,brand_3_out


def mcnet():
    # 初始定义一个随机数
    rect = torch.tensor(np.random.rand(16))
    rect_local = np.random.rand(20)
    return MCNet(block_b=BasicBlock, block_a=CSO_AttentionBlock, block_o=CO_AttentionBlock, rect=rect,
                 rect_local=rect_local, num_classes=8631)