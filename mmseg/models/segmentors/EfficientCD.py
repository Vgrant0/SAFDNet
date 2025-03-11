import logging
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .encoder_decoder import EncoderDecoder
import torch
from einops import rearrange


import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmcv.cnn import ConvModule
from mmseg.models.backbones.resnet import BasicBlock

#ssim
class CustomDistance:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.padding = window_size // 2

    def ssim(self, img1, img2, window_size=3, C1=0.01**2, C2=0.03**2):
        """
        计算两个窗口之间的SSIM。

        参数:
        img1 (Tensor): 第一个窗口，形状为 (batch_size, channels, height, width)
        img2 (Tensor): 第二个窗口，形状为 (batch_size, channels, height, width)
        window_size (int): 窗口大小
        C1, C2 (float): SSIM公式中的常数

        返回:
        ssim_value (Tensor): SSIM值，形状为 (batch_size, 1, height, width)
        """
        # 计算均值
        mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=self.padding)
        mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=self.padding)

        # 计算方差
        sigma1 = F.avg_pool2d((img1 - mu1) ** 2, window_size, stride=1, padding=self.padding)
        sigma2 = F.avg_pool2d((img2 - mu2) ** 2, window_size, stride=1, padding=self.padding)
        sigma12 = F.avg_pool2d((img1 - mu1) * (img2 - mu2), window_size, stride=1, padding=self.padding)

        # 计算SSIM
        ssim_value = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))

        return ssim_value

    def structural_similarity_distance(self, img1, img2):
        # 确保输入张量具有相同的形状
        assert img1.shape == img2.shape, "Input images must have the same shape"

        # 计算SSIM值
        ssim_values = self.ssim(img1, img2)

        # 在第三个维度（通道维度）上取平均
        ssim_values = torch.mean(ssim_values, dim=1, keepdim=True)

        # 计算结构相似性距离
        distances = 1 - ssim_values

        # 归一化距离
        max_distance = torch.max(distances)
        normalized_distances = torch.sigmoid(distances / max_distance)

        return normalized_distances

def create_STE_(in_channels, down_rate, groups=None):
    mid_channels = int(in_channels/down_rate/down_rate)
    out_channels = int(in_channels/down_rate)
    if groups is None:
        groups = in_channels
    
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=groups),
        nn.GELU(),
        # nn.Dropout2d(p=0.2),  # 添加 Dropout2d 层，p 表示关闭神经元的概率
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels, mid_channels, kernel_size=1),
        nn.GELU(),
        # nn.Dropout2d(p=0.2),  # 添加 Dropout2d 层，p 表示关闭神经元的概率
        nn.BatchNorm2d(mid_channels),
        nn.Conv2d(mid_channels, out_channels, kernel_size=1),
        nn.BatchNorm2d(out_channels),
    )


class CDNet(nn.Module):
    def __init__(self, neck, model_name='efficientnet_b5'):
        super(CDNet, self).__init__()
        self.model = timm.create_model('efficientnet_b5', pretrained=True, features_only=True)
        # self.model = timm.create_model('efficientnet_b5', pretrained=True, pretrained_cfg_overlay=dict(file='/home/jicredt1/pretrained/efficientnet_b5/pytorch_model.bin'), features_only=True)
        self.interaction_layers = ['blocks']
        # self.up_layer = [5, 3, 2, 1, 0]
        FPN_DICT = neck
        norm_cfg = dict(type='SyncBN', requires_grad=True)

        # FPN_DICT = {'type': 'FPN', 'in_channels': [16, 24, 40, 80, 112, 192, 320], 'out_channels': 128, 'num_outs': 7}
        # FPN_DICT['in_channels'] = [i*2 for i in FPN_DICT['in_channels']]
        self.fpnA = MODELS.build(FPN_DICT)
        self.fpnB = MODELS.build(FPN_DICT)

        self.decode_layers1 = nn.Sequential(
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg)
        )
        self.decode_layers2 = nn.Sequential(
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg)
        )
        self.sigmoid = nn.Sigmoid()
    
    def change_feature(self, x, y):
        i = 2
        for index in range(0, len(x), i):
            x[index], y[index] = y[index], x[index]
        return x, y
    
    def euclidean_distance(self, img1, img2):
        diff_squared = (img1 - img2) ** 2
        distances = torch.sqrt(torch.sum(diff_squared, dim=1)).unsqueeze(1)
        max_distance = torch.max(distances)
        normalized_distances = torch.sigmoid(distances / max_distance)
        return normalized_distances

    def forward(self, xA, xB):
        for name, module in self.model.named_children():
            # print(f"Module Name: {name}")
            if name not in self.interaction_layers:
                xA = module(xA)
                xB = module(xB)
            else:
                xA_list = []
                xB_list = []
                for sub_name, sub_module in module.named_children():
                    # print(f"Module Name: {name}, Submodule Name: {sub_name}")
                    xA = sub_module(xA)
                    xB = sub_module(xB)
                    xA_list.append(xA)
                    xB_list.append(xB)
                break
        xA_list, xB_list = self.change_feature(xA_list, xB_list)
        xA_list = self.fpnA(xA_list)
        xB_list = self.fpnB(xB_list)
        xA_list, xB_list = self.change_feature(list(xA_list), list(xB_list))

        change_map = []
        curAB6 = torch.cat([xA_list[6], xB_list[6]], dim=1)
        curAB6 = self.euclidean_distance(xA_list[6], xB_list[6])*self.decode_layers1[6](curAB6)
        change_map.append(curAB6)

        curAB5 = torch.cat([xA_list[5], xB_list[5]], dim=1)
        curAB5 = curAB6+self.decode_layers1[5](curAB5)
        curAB5 = F.interpolate(curAB5, scale_factor=2, mode='bilinear', align_corners=False)
        dist5 = self.euclidean_distance(xA_list[5], xB_list[5])
        dist5 = F.interpolate(dist5, scale_factor=2, mode='bilinear', align_corners=False)
        curAB5 = dist5*self.decode_layers2[5](curAB5)
        change_map.append(curAB5)

        curAB4 = torch.cat([xA_list[4], xB_list[4]], dim=1)
        curAB4 = curAB5+self.decode_layers1[4](curAB4)
        curAB4 = self.euclidean_distance(xA_list[4], xB_list[4])*self.decode_layers2[4](curAB4)
        change_map.append(curAB4)

        curAB3 = torch.cat([xA_list[3], xB_list[3]], dim=1)
        curAB3 = curAB4+self.decode_layers1[3](curAB3)
        curAB3 = F.interpolate(curAB3, scale_factor=2, mode='bilinear', align_corners=
                                False)
        dist3 = self.euclidean_distance(xA_list[3], xB_list[3])
        dist3 = F.interpolate(dist3, scale_factor=2, mode='bilinear', align_corners=False)
        curAB3 = dist3*self.decode_layers2[3](curAB3)
        change_map.append(curAB3)

        curAB2 = torch.cat([xA_list[2], xB_list[2]], dim=1)
        curAB2 = curAB3+self.decode_layers1[2](curAB2)
        curAB2 = F.interpolate(curAB2, scale_factor=2, mode='bilinear', align_corners=
                                False)
        dist2 = self.euclidean_distance(xA_list[2], xB_list[2])
        dist2 = F.interpolate(dist2, scale_factor=2, mode='bilinear', align_corners=False)
        curAB2 = dist2*self.decode_layers2[2](curAB2)
        change_map.append(curAB2)

        curAB1 = torch.cat([xA_list[1], xB_list[1]], dim=1)
        curAB1 = curAB2+self.decode_layers1[1](curAB1)
        curAB1 = F.interpolate(curAB1, scale_factor=2, mode='bilinear', align_corners=
                                False)
        dist1 = self.euclidean_distance(xA_list[1], xB_list[1])
        dist1 = F.interpolate(dist1, scale_factor=2, mode='bilinear', align_corners=False)
        curAB1 = dist1*self.decode_layers2[1](curAB1)
        change_map.append(curAB1)
        
        return change_map

class CDNet_triangle_dense_share_ssim(nn.Module):
    def __init__(self, neck, model_name='efficientnet_b5'):
        super(CDNet_triangle_dense_share_ssim, self).__init__()
        self.model = timm.create_model('efficientnet_b5', pretrained=True, features_only=True)
        # self.model = timm.create_model('efficientnet_b5', pretrained=True, pretrained_cfg_overlay=dict(file='/home/jicredt1/pretrained/efficientnet_b5/pytorch_model.bin'), features_only=True)
        self.interaction_layers = ['blocks']
        # self.up_layer = [5, 3, 2, 1, 0]
        FPN_DICT = neck
        norm_cfg = dict(type='SyncBN', requires_grad=True)

        FPN_DICT = {'type': 'FPN', 'in_channels': [40, 64, 176,512], 'out_channels': 128, 'num_outs': 4}
        #FPN_DICT['in_channels'] = [i*2 for i in FPN_DICT['in_channels']]
        self.fpnA = MODELS.build(FPN_DICT)
        self.fpnB = MODELS.build(FPN_DICT)
        self.allowed_values = [1, 2, 4, 6]

        # 初始化可学习的权重因子列表
        # self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(2)])

        BN_MOMENTUM = 0.1
        relu_inplace = True
        self.ALIGN_CORNERS = True

        self.A1_conv_3_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )
        self.A1_conv_1_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )
        self.A1_conv_2_1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )
        self.A1_conv_0_1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )
        self.A1_conv_1_0 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )

        self.A2_conv_2_1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )
        self.A2_conv_0_1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )
        self.A2_conv_1_0 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )

        self.A3_conv_1_0 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )


        self.B1_conv_3_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )
        self.B1_conv_1_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )
        self.B1_conv_2_1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )
        self.B1_conv_0_1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )
        self.B1_conv_1_0 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )

        self.B2_conv_2_1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )
        self.B2_conv_0_1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )
        self.B2_conv_1_0 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )

        self.B3_conv_1_0 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )

        self.A_B_layers=nn.Sequential(
            nn.Sequential(
            nn.Conv2d(80, 40, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(40, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
            ),
            nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
            ),
            nn.Sequential(
            nn.Conv2d(352, 176, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(176, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
            ),
            nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
            )
        )

        


        self.decode_layers1 = nn.Sequential(
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg)
        )
        self.decode_layers2 = nn.Sequential(
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg)
        )
        self.sigmoid = nn.Sigmoid()

        self.A_STE_0_1 = nn.Sequential(
            nn.Conv2d(24+40, 24+40, kernel_size=3, padding=1, groups=24+40),
            nn.GELU(),
            nn.BatchNorm2d(24+40),
            nn.Conv2d(24+40, 32, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 24+40, kernel_size=1),
            nn.BatchNorm2d(24+40),)
        
        self.A_STE_layers=nn.Sequential(
            #create_STE_(in_channels=24+24, down_rate=2),
            create_STE_(in_channels=40+40, down_rate=2),
            create_STE_(in_channels=64+64, down_rate=2),
            #create_STE_(in_channels=128+128, down_rate=2),
            create_STE_(in_channels=176+176, down_rate=2),
            #create_STE_(in_channels=304+304, down_rate=2),
            create_STE_(in_channels=512+512, down_rate=2),
        )
        self.B_STE_layers=nn.Sequential(
            #create_STE_(in_channels=24+24, down_rate=2),
            create_STE_(in_channels=40+40, down_rate=2),
            create_STE_(in_channels=64+64, down_rate=2),
            #create_STE_(in_channels=128+128, down_rate=2),
            create_STE_(in_channels=176+176, down_rate=2),
            #create_STE_(in_channels=304+304, down_rate=2),
            create_STE_(in_channels=512+512, down_rate=2),
        )
        
        
    
    def change_feature(self, x, y):
        i = 2
        for index in range(0, len(x), i):
            x[index], y[index] = y[index], x[index]
        return x, y
    
    def euclidean_distance(self, img1, img2):
        diff_squared = (img1 - img2) ** 2
        distances = torch.sqrt(torch.sum(diff_squared, dim=1)).unsqueeze(1)
        max_distance = torch.max(distances)
        normalized_distances = torch.sigmoid(distances / max_distance)
        return normalized_distances

    def forward(self, xA, xB):
        xA_list=[]
        xB_list=[]

        mark=0
        
        for name, module in self.model.named_children():
            # print(f"Module Name: {name}")conda 
            if name not in self.interaction_layers:
                xA = module(xA)
                xB = module(xB)
            else:
                xA_list = []
                xB_list = []
                for sub_name, sub_module in module.named_children():
                    #print(f"Module Name: {name}, Submodule Name: {sub_name}")
                    xA = sub_module(xA)
                    xB = sub_module(xB)


                    if int(sub_name) in self.allowed_values:
                        if int(sub_name)==2 or int(sub_name)==6:
                        # if int(sub_name)==2:
                            # sub=torch.abs(xA-xB)
                            sub=torch.cat([xA, xB], dim=1)
                            sub=self.A_B_layers[mark](sub)

                            # # 根据 sub_name 选择对应的权重因子
                            # alpha_idx = 0 if int(sub_name) == 2 else 1
                            # sub=sub*self.alpha[alpha_idx]

                            xA = torch.cat([xA, sub], dim=1)
                            xA=self.A_STE_layers[mark](xA)

                            xB = torch.cat([xB, sub], dim=1)
                            xB=self.B_STE_layers[mark](xB)
                            
                        mark+=1

                        xA_list.append(xA)
                        xB_list.append(xB)
                #break
        # x_sum=(xA+xB)/2
        # xA_list = xA_list[-4:]
        # xB_list = xB_list[-4:]

        xA_list = list(self.fpnA(xA_list))
        xB_list = list(self.fpnB(xB_list))

        # # 创建注意力融合模块并移动到GPU
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # attention_fusion = AttentionFusion(128).to(device)
        # #特征融合
        outA_list = []
        outB_list = []

        # #特征对齐
        # mark=0
        # for main, aux in zip(xA_list, xB_list):
        #     fused_main, fused_aux = attention_fusion(main, aux)
        #     xA_list[mark] = fused_main
        #     xB_list[mark] = fused_aux
        #     mark=mark+1


        
        outA_list.append(xA_list[3])

        #第一级
        sample=F.interpolate( xA_list[3], size=xA_list[2].shape[-2:], mode='bilinear', align_corners=self.ALIGN_CORNERS)
        xA_list[2] =torch.cat([sample,xA_list[2]], 1)
        xA_list[2]=self.A1_conv_3_2(xA_list[2])
        sample=F.interpolate( xA_list[1], size=xA_list[2].shape[-2:], mode='bilinear', align_corners=self.ALIGN_CORNERS)
        xA_list[2] =torch.cat([sample,xA_list[2]], 1)
        xA_list[2]=self.A1_conv_1_2(xA_list[2])
        outA_list.append(xA_list[2])

        sample=F.interpolate( xA_list[2],size=xA_list[1].shape[-2:], mode='bilinear', align_corners=self.ALIGN_CORNERS)
        xA_list[1] =torch.cat([sample,xA_list[1]], 1)
        xA_list[1]=self.A1_conv_2_1(xA_list[1])
        sample=F.interpolate( xA_list[0],size=xA_list[1].shape[-2:], mode='bilinear', align_corners=self.ALIGN_CORNERS)
        xA_list[1] =torch.cat([sample,xA_list[1]], 1)
        xA_list[1]=self.A1_conv_0_1(xA_list[1])

        sample=F.interpolate( xA_list[1], size=xA_list[0].shape[-2:], mode='bilinear', align_corners=self.ALIGN_CORNERS)
        xA_list[0] =torch.cat([sample,xA_list[0]], 1)
        xA_list[0]=self.A1_conv_1_0(xA_list[0])

        #第二级
        sample=F.interpolate( xA_list[2],size=xA_list[1].shape[-2:], mode='bilinear', align_corners=self.ALIGN_CORNERS)
        xA_list[1] =torch.cat([sample,xA_list[1]], 1)
        xA_list[1]=self.A2_conv_2_1(xA_list[1])
        sample=F.interpolate( xA_list[0],size=xA_list[1].shape[-2:], mode='bilinear', align_corners=self.ALIGN_CORNERS)
        xA_list[1] =torch.cat([sample,xA_list[1]], 1)
        xA_list[1]=self.A2_conv_0_1(xA_list[1])
        outA_list.append(xA_list[1])

        sample=F.interpolate( xA_list[1], size=xA_list[0].shape[-2:], mode='bilinear', align_corners=self.ALIGN_CORNERS)
        xA_list[0] =torch.cat([sample,xA_list[0]], 1)
        xA_list[0]=self.A2_conv_1_0(xA_list[0])

        #第三级
        sample=F.interpolate( xA_list[1], size=xA_list[0].shape[-2:], mode='bilinear', align_corners=self.ALIGN_CORNERS)
        xA_list[0] =torch.cat([sample,xA_list[0]], 1)
        xA_list[0]=self.A3_conv_1_0(xA_list[0])
        outA_list.append(xA_list[0])

        #B
        outB_list.append(xB_list[3])
        #第一级
        sample=F.interpolate( xB_list[3], size=xB_list[2].shape[-2:], mode='bilinear', align_corners=self.ALIGN_CORNERS)
        xB_list[2] =torch.cat([sample,xB_list[2]], 1)
        xB_list[2]=self.B1_conv_3_2(xB_list[2])
        sample=F.interpolate( xB_list[1], size=xB_list[2].shape[-2:], mode='bilinear', align_corners=self.ALIGN_CORNERS)
        xB_list[2] =torch.cat([sample,xB_list[2]], 1)
        xB_list[2]=self.B1_conv_1_2(xB_list[2])
        outB_list.append(xB_list[2])

        sample=F.interpolate( xB_list[2],size=xB_list[1].shape[-2:], mode='bilinear', align_corners=self.ALIGN_CORNERS)
        xB_list[1] =torch.cat([sample,xB_list[1]], 1)
        xB_list[1]=self.B1_conv_2_1(xB_list[1])
        sample=F.interpolate( xB_list[0],size=xB_list[1].shape[-2:], mode='bilinear', align_corners=self.ALIGN_CORNERS)
        xB_list[1] =torch.cat([sample,xB_list[1]], 1)
        xB_list[1]=self.B1_conv_0_1(xB_list[1])

        sample=F.interpolate( xB_list[1], size=xB_list[0].shape[-2:], mode='bilinear', align_corners=self.ALIGN_CORNERS)
        xB_list[0] =torch.cat([sample,xB_list[0]], 1)
        xB_list[0]=self.B1_conv_1_0(xB_list[0])

        #第二级
        sample=F.interpolate( xB_list[2],size=xB_list[1].shape[-2:], mode='bilinear', align_corners=self.ALIGN_CORNERS)
        xB_list[1] =torch.cat([sample,xB_list[1]], 1)
        xB_list[1]=self.B2_conv_2_1(xB_list[1])
        sample=F.interpolate( xB_list[0],size=xB_list[1].shape[-2:], mode='bilinear', align_corners=self.ALIGN_CORNERS)
        xB_list[1] =torch.cat([sample,xB_list[1]], 1)
        xB_list[1]=self.B2_conv_0_1(xB_list[1])
        outB_list.append(xB_list[1])

        sample=F.interpolate( xB_list[1], size=xB_list[0].shape[-2:], mode='bilinear', align_corners=self.ALIGN_CORNERS)
        xB_list[0] =torch.cat([sample,xB_list[0]], 1)
        xB_list[0]=self.B2_conv_1_0(xB_list[0])

        #第三级
        sample=F.interpolate( xB_list[1], size=xB_list[0].shape[-2:], mode='bilinear', align_corners=self.ALIGN_CORNERS)
        xB_list[0] =torch.cat([sample,xB_list[0]], 1)
        xB_list[0]=self.B3_conv_1_0(xB_list[0])
        outB_list.append(xB_list[0])

        # 创建 CustomDistance 实例
        custom_distance = CustomDistance(window_size=3)
        #custom_distance = CustomDistance(window_size=4)

        change_map = []
        curAB0 = torch.cat([outA_list[0], outB_list[0]], dim=1)
        curAB0 = self.euclidean_distance(outA_list[0], outB_list[0])*self.decode_layers1[0](curAB0)
        # curAB0 = custom_distance.structural_similarity_distance(outA_list[0], outB_list[0])*self.decode_layers1[0](curAB0)
        change_map.append(curAB0)

        curAB1 = torch.cat([outA_list[1], outB_list[1]], dim=1)
        curAB0=F.interpolate(curAB0, scale_factor=2, mode='bilinear', align_corners=False)
        curAB1=curAB0+self.decode_layers1[1](curAB1)
        dist1 = self.euclidean_distance(outA_list[1], outB_list[1])
        # dist1 = custom_distance.structural_similarity_distance(outA_list[1], outB_list[1])
        curAB1=dist1*self.decode_layers2[1](curAB1)
        change_map.append(curAB1)

        curAB2 = torch.cat([outA_list[2], outB_list[2]], dim=1)
        curAB1=F.interpolate(curAB1, scale_factor=2, mode='bilinear', align_corners=False)
        curAB2=curAB1+self.decode_layers1[2](curAB2)
        dist2 = custom_distance.structural_similarity_distance(outA_list[2], outB_list[2])
        curAB2=dist2*self.decode_layers2[2](curAB2)
        change_map.append(curAB2)

        curAB3 = torch.cat([outA_list[3], outB_list[3]], dim=1)
        curAB2=F.interpolate(curAB2, scale_factor=2, mode='bilinear', align_corners=False)
        curAB3=curAB2+self.decode_layers1[3](curAB3)
        dist3 = custom_distance.structural_similarity_distance(outA_list[3], outB_list[3])
        curAB3=dist3*self.decode_layers2[3](curAB3)
        change_map.append(curAB3)

        
        return change_map
   
@MODELS.register_module()
class EfficientCD(EncoderDecoder):
    def __init__(
        self,
        backbone: ConfigType = None,
        decode_head: ConfigType = None,
        neck: OptConfigType = None,
        auxiliary_head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        pretrained: Optional[str] = None,
        model_name: Optional[str] = None,
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(backbone=backbone, decode_head=decode_head, data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # self.model = CDNet(neck, model_name)
        self.model = CDNet_triangle_dense_share_ssim(neck, model_name)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        imgs1, imgs2 = inputs[:, :3, :, :], inputs[:, 3:, :, :]
        x = self.model(imgs1, imgs2)
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                        self.test_cfg)
        return seg_logits

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        losses = dict()

        imgs1, imgs2 = inputs[:, :3, :, :], inputs[:, 3:, :, :]
        change_map = self.model(imgs1, imgs2)
        # self.G_loss =  self._pxl_loss(self.G_pred1, gt) + self._pxl_loss(self.G_pred2, gt) + 0.5*(self._pxl_loss(self.G_middle1, gt)+self._pxl_loss(self.G_middle2, gt))
        loss_decode = self._decode_head_forward_train(change_map, data_samples)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(change_map, data_samples)
            losses.update(loss_aux)
        return losses

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        imgs1, imgs2 = inputs[:, :3, :, :], inputs[:, 3:, :, :]
        x = self.model(imgs1, imgs2)
        data_samples = [{}]
        data_samples[0]['img_shape'] = (256, 256)
        seg_logits = self.decode_head.predict(x, data_samples,
                                        self.test_cfg)
        return seg_logits
