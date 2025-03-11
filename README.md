
### mmcv=2.1.0
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
### torch=2.1.0
### cuda=11.8
pip install torch==2.1.0+cu118 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
### mmseg=0.22.0
pip install mmsegmentation
pip install -v -e .
### gdal
conda install GDAL
### ftfy and regex
pip install ftfy
pip install regex
pip install einops
pip install timm
pip install kornia

### 思路
设计了一个用于遥感图像变化检测的网络结构，主要优化方式围绕双时相图像的风格展开。首先使用一个共享参数的densenet121骨干网络进行特征提取，得到双时相四个尺度下的特征图，然后基于直方图均衡化思想分别对每个时相的特征图金字塔进行风格统一化。在这之后设计了一个融合自差异计算模块，基于自监督学习的思想出发，首先将两个时相的特征金字塔进行特征交互，然后计算三种差异，分别是每个时相融合前后的差异，以及融合后两个时相的差异，将三个差异进行可学习的加权融合。另外，差异的计算方式也有所改进，使用的是多尺度的SSIM相似性计算加权模块而不是传统的欧式距离。

### 直方图均衡化的思想
Kornia库虽然提供了直方图均衡化，但是仅针对图像变为tensor在GPU上利用矩阵计算进行，并没有适配到特征图上，所以本研究的主要工作之一是将直方图均衡化匹配到特征图层级，并以此为基本思想设计以一个特征对齐模块，用来削弱双时相特征图中伪变化的特征。

### 融合自相似模块
参考cyclegan中的循环一致性损失，构建了一种差异度量方式，除了传统的XA与XB的差异，还融入了单时相混合前混合后的差异度量，进一步削弱了风格的影响

### 多尺度差异度量
优化了之前SSIM遇到8*8等小尺寸特征图的缺陷，构建了多尺度度量的差异衡量方式