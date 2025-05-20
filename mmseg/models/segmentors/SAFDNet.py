import logging
from typing import List, Optional
import kornia
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

class CustomDistance:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.padding = (window_size-1) // 2

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
    
    def mutual_info(self, img1, img2, window_size=3, eps=1e-6):
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

        # 计算相关系数的平方（避免除零）
        rho_sq = (sigma12 ** 2) / (sigma1 * sigma2 + eps)
        
        # 计算互信息（基于高斯假设）
        mi_value = -0.5 * torch.log(1 - rho_sq + eps)

        return mi_value
    
    def structural_similarity_distance(self, img1, img2):
        # 确保输入张量具有相同的形状
        assert img1.shape == img2.shape, "Input images must have the same shape"

        # 计算SSIM值
        ssim_values = self.ssim(img1, img2)

        # 在第二个维度（通道维度）上取平均
        ssim_values = torch.mean(ssim_values, dim=1, keepdim=True)

        # 计算结构相似性距离
        distances = 1 - ssim_values

        # 归一化距离
        max_distance = torch.max(distances)
        normalized_distances = torch.sigmoid(distances / max_distance)

        return normalized_distances


class style_self_muldiff_Net(nn.Module):
    def __init__(self, neck, model_name='densenet121'):
        super(style_self_muldiff_Net, self).__init__()
        '''
        swinv2_base_window8_256
        torch.Size([16, 64, 64, 128])
        torch.Size([16, 32, 32, 256])
        torch.Size([16, 16, 16, 512])
        torch.Size([16, 8, 8, 1024])
        '''
        # self.model = timm.create_model('swinv2_base_window8_256', pretrained=True, features_only=True)
        self.model = timm.create_model('densenet121', pretrained=True, features_only=True)
        # self.model = timm.create_model('efficientnet_b5', pretrained=True, pretrained_cfg_overlay=dict(file='/home/jicredt1/pretrained/efficientnet_b5/pytorch_model.bin'), features_only=True)
        self.interaction_layers = ['blocks']
        norm_cfg = dict(type='SyncBN', requires_grad=True)

        FPN_DICT = {'type': 'FPN', 'in_channels': [256, 512, 1024, 1024], 'out_channels': 128, 'num_outs': 4}
        FUSED_DICK={'type': 'FPN', 'in_channels': [128, 128, 128, 128], 'out_channels': 128, 'num_outs': 4}
        DIF_DIF_FUSED_DICK={'type': 'FPN', 'in_channels': [512, 512, 512, 512], 'out_channels': 256, 'num_outs': 4}
        self.fpn_fuse_A = MODELS.build(FUSED_DICK)
        self.fpn_fuse_B = MODELS.build(FUSED_DICK)
        self.fpn_diff = MODELS.build(DIF_DIF_FUSED_DICK)
        '''
        torch.Size([16, 64, 128, 128])
        torch.Size([16, 256, 64, 64])
        torch.Size([16, 512, 32, 32])
        torch.Size([16, 1024, 16, 16])
        torch.Size([16, 1024, 8, 8])
        '''
        self.fpnA = MODELS.build(FPN_DICT)
        self.fpnB = MODELS.build(FPN_DICT)

        self.ALIGN_CORNERS = True

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
        self.weight_layersA = nn.Sequential(
            BasicBlock(inplanes=FPN_DICT['out_channels'], planes=FPN_DICT['out_channels'], norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels'], planes=FPN_DICT['out_channels'], norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels'], planes=FPN_DICT['out_channels'], norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels'], planes=FPN_DICT['out_channels'], norm_cfg=norm_cfg)
        )
        self.weight_layersB = nn.Sequential(
            BasicBlock(inplanes=FPN_DICT['out_channels'], planes=FPN_DICT['out_channels'], norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels'], planes=FPN_DICT['out_channels'], norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels'], planes=FPN_DICT['out_channels'], norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels'], planes=FPN_DICT['out_channels'], norm_cfg=norm_cfg)
        )
        self.weight_A = nn.Parameter(torch.ones(len(self.weight_layersA)) * 0.5)  # 初始化为0.5
        self.weight_B = nn.Parameter(torch.ones(len(self.weight_layersB)) * 0.5)  # 初始化为0.5
        # 定义4个可学习权重列表
        self.weights_list = nn.ParameterList([
            nn.Parameter(torch.ones(3) * (1.0 / 3)) for _ in range(4)
        ])
        self.sigmoid = nn.Sigmoid()


    def change_feature(self, x, y):
        # 使用列表推导式创建新的列表
        x_copy = [item for item in x]
        y_copy = [item for item in y]
        i = 2
        for index in range(0, len(x), i):
            x_copy[index], y_copy[index] = y_copy[index], x_copy[index]
        return x_copy, y_copy

    def euclidean_distance(self, img1, img2):
        diff_squared = (img1 - img2) ** 2
        distances = torch.sqrt(torch.sum(diff_squared, dim=1)).unsqueeze(1)
        max_distance = torch.max(distances)
        normalized_distances = torch.sigmoid(distances / max_distance)
        return normalized_distances

    def single_histogram_equalization(self, image):
        # 获取对应特征图的Max与Min值，便于归一化
        Max = torch.max(image)
        Min = torch.min(image)
        image=(image - Min) / (Max - Min)

        # 均衡化的设置
        min_val=0.0
        max_val=1.0
        n_bins: int = 16
        b, c, h, w = image.size()
        # 计算图像的直方图和分箱中心点，kornia要求输入张量的值类型是torch.float32 或 torch.float64，范围在[0,1] 
        hist, _ = kornia.enhance.image_histogram2d(image, min = min_val, max = max_val, n_bins = n_bins)

        # 计算累积分布函数 (CDF)   归一化 CDF   调整CDF尺寸以便映射索引
        cdf = hist.cumsum(dim=-1)
        cdf_normalized = cdf / cdf[:, :, -1:]  
        cdf_normalized = cdf_normalized.view(b * c, n_bins)  # [b*c, n_bins]

        # 扁平化处理，便于进行映射
        flat_image = image.view(b * c, h * w)

        # 生成分箱区间， 需要提供n_bins+1个端点，包括两端，以便生成n_bins个区间
        bins = torch.linspace(min_val, max_val, steps=n_bins+1).to(image.device)
        # 找到每个像素属于哪个分箱，根据下界确定，所以提供nbins除去末尾值的所有区间上界值
        bin_idx = torch.bucketize(flat_image, bins[:-1])
        # 如果有任何索引为 16 的值，将其调整为 15，防止索引越界
        bin_idx[bin_idx == len(bins) - 1] = len(bins) - 2
        # 使用归一化的 CDF 映射新值，torch.gather 函数用于根据索引index从输入张量中收集元素
        new_values = torch.gather(cdf_normalized, dim = -1, index = bin_idx)

        # 将均衡化后的图像重塑回原始形状
        equalized_image = new_values.view(b, c, h, w)

        return equalized_image


    def histogram_equalization(self, imgA, imgB):
        # 输入B*C*H*W的特征图，统计出B*C*H*W的权重图
        equalized_imgA=self.single_histogram_equalization(imgA)
        equalized_imgB=self.single_histogram_equalization(imgB)

        return (equalized_imgA, equalized_imgB)

    def diff_cal(self, list_img, fused_img):
        change_map = []
        # 创建 CustomDistance 实例
        # custom_distance = CustomDistance(window_size=3)

        cur0 = torch.cat([list_img[0], fused_img[0]], dim=1)
        # dist0 = self.euclidean_distance(list_img[0], fused_img[0])
        dist0 = self.multi_dist(list_img[0], fused_img[0], 0)
        cur0 = dist0 * self.decode_layers1[0](cur0)
        # curAB0 = custom_distance.structural_similarity_distance(outA_list[0], outB_list[0])*self.decode_layers1[0](curAB0)
        change_map.append(cur0)

        cur1 = torch.cat([list_img[1], fused_img[1]], dim=1)
        cur0=F.interpolate(cur0, scale_factor=2, mode='bilinear', align_corners=False)
        cur1=cur0+self.decode_layers1[1](cur1)
        dist1 = self.multi_dist(list_img[1], fused_img[1], 1)
        # dist1 = custom_distance.structural_similarity_distance(outA_list[1], outB_list[1])
        cur1=dist1*self.decode_layers2[1](cur1)
        change_map.append(cur1)

        cur2 = torch.cat([list_img[2], fused_img[2]], dim=1)
        cur1=F.interpolate(cur1, scale_factor=2, mode='bilinear', align_corners=False)
        cur2=cur1+self.decode_layers1[2](cur2)
        dist2 = self.multi_dist(list_img[2], fused_img[2], 2)
        cur2=dist2*self.decode_layers2[2](cur2)
        change_map.append(cur2)

        cur3 = torch.cat([list_img[3], fused_img[3]], dim=1)
        cur2=F.interpolate(cur2, scale_factor=2, mode='bilinear', align_corners=False)
        cur3=cur2+self.decode_layers1[3](cur3)
        dist3 = self.multi_dist(list_img[3], fused_img[3], 3)
        cur3=dist3*self.decode_layers2[3](cur3)
        change_map.append(cur3)

        return change_map

    def structural_similarity_distance(self, img1, img2, window_size, C1=0.01**2, C2=0.03**2):
        pad_size = (window_size-1) // 2
        # 计算均值
        mu1 = F.avg_pool2d(img1, window_size, stride = 1, padding = pad_size)
        mu2 = F.avg_pool2d(img2, window_size, stride = 1, padding = pad_size)
        # 计算方差
        sigma1 = F.avg_pool2d((img1 - mu1) ** 2, window_size, stride=1, padding=pad_size)
        sigma2 = F.avg_pool2d((img2 - mu2) ** 2, window_size, stride=1, padding=pad_size)
        sigma12 = F.avg_pool2d((img1 - mu1) * (img2 - mu2), window_size, stride=1, padding=pad_size)

        # 计算SSIM
        ssim_value = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
        # 在第二个维度（通道维度）上取平均
        ssim_values = torch.mean(ssim_value, dim=1, keepdim=True)
        # 计算结构相似性距离
        distances = 1 - ssim_values
        # 归一化距离
        max_distance = torch.max(distances)
        normalized_distances = torch.sigmoid(distances / max_distance)

        return normalized_distances
    
    def mutual_infor_distance(self, img1, img2, window_size, eps=1e-6):
        pad_size = (window_size-1) // 2
        # 计算均值
        mu1 = F.avg_pool2d(img1, window_size, stride = 1, padding = pad_size)
        mu2 = F.avg_pool2d(img2, window_size, stride = 1, padding = pad_size)
        # 计算方差
        sigma1 = F.avg_pool2d((img1 - mu1) ** 2, window_size, stride=1, padding=pad_size)
        sigma2 = F.avg_pool2d((img2 - mu2) ** 2, window_size, stride=1, padding=pad_size)
        sigma12 = F.avg_pool2d((img1 - mu1) * (img2 - mu2), window_size, stride=1, padding=pad_size)

        # 计算相关系数的平方（避免除零）
        rho_sq = (sigma12 ** 2) / (sigma1 * sigma2 + eps)
        
        # 计算互信息（基于高斯假设）
        mi_value = -0.5 * torch.log(1 - rho_sq + eps)
        
        # 在第二个维度（通道维度）上取平均
        mi_values = torch.mean(mi_value, dim=1, keepdim=True)
        # 计算结构相似性距离
        distances = 1 - mi_values
        # 归一化距离
        max_distance = torch.max(distances)
        normalized_distances = torch.sigmoid(distances / max_distance)

        return normalized_distances
    
    def multi_dist(self, img1, img2, level):
        dist_1 = self.euclidean_distance(img1, img2)
        # dist_3 = self.structural_similarity_distance(img1, img2, 3)
        # dist_5 = self.structural_similarity_distance(img1, img2, 5)
        dist_3 = self.mutual_infor_distance(img1, img2, 3)
        dist_5 = self.mutual_infor_distance(img1, img2, 5)

        dist = self.weights_list[level][0] * dist_1 + self.weights_list[level][1] * dist_3 + self.weights_list[level][2] * dist_5

        return dist
   
    def forward(self, xA, xB):

        feature_A_list = self.model(xA)[1:]
        feature_B_list = self.model(xB)[1:]

        feature_A_list = list(self.fpnA(feature_A_list))[::-1]
        feature_B_list = list(self.fpnB(feature_B_list))[::-1]

        level = len(feature_A_list)

        # 这里返回与outA_list，outB_list完全一样尺寸的权重张量，用于对风格进行修正
        aligned_map=[]
        for i in range(level):
            # z=len(outA_list)-i-1
            aligned_map.append(self.histogram_equalization(feature_A_list[i], feature_B_list[i]))
        # 风格对齐 outlist[0]~[3]对应尺寸8*8~64*64
        align_A_list = [feature_A_list[i] * aligned_map[i][0] for i in range(level)]
        align_A_list = [self.weight_layersA[i](align_A_list[i]) for i in range(level)]
        outA_list = [self.weight_A[i] * outA + (1 - self.weight_A[i]) * align_A for outA, align_A in zip(feature_A_list, align_A_list)]

        align_B_list = [feature_B_list[i] * aligned_map[i][1] for i in range(level)]
        align_B_list = [self.weight_layersB[i](align_B_list[i]) for i in range(level)]
        outB_list = [self.weight_B[i] * outB + (1 - self.weight_B[i]) * align_B for outB, align_B in zip(feature_B_list, align_B_list)]

        # 融合自差异性计算
        fused_A_list,fused_B_list = self.change_feature(outA_list, outB_list)
        fused_A_list = list(self.fpn_fuse_A(fused_A_list))
        fused_B_list = list(self.fpn_fuse_B(fused_B_list))
        fused_A_list,fused_B_list = self.change_feature(fused_A_list, fused_B_list)
        # 创建 CustomDistance 实例
        # custom_distance = CustomDistance(window_size=3)
        change_map_A = self.diff_cal(outA_list, fused_A_list)
        change_map_B = self.diff_cal(outB_list, fused_B_list)

        change_map = [torch.cat([change_A, change_B], dim = 1) for change_A, change_B in zip(change_map_A, change_map_B)]
        change_map = list(self.fpn_diff(change_map))


        return change_map


@MODELS.register_module()
class SAFDNet(EncoderDecoder):
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
        self.model = style_self_muldiff_Net(neck, model_name)
        # self.model = BaseNet(neck, model_name)
        # self.model = style_match_Net(neck, model_name)
        # self.model = self_diff_Net(neck, model_name)

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
