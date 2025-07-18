
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

### Hist-Equal Module
A style alignment module (Hist-Equal) inspired by histogram equalization is proposed. This module recalibrates feature maps based on their frequency information to align the styles of bitemporal images. This approach reduces false alarms caused by stylistic differences (pseudo-changes) from varying imaging conditions, without significantly increasing the model's complexity as seen in GAN-based methods

### Fusion-Difference
A novel fusion-difference calculation module (Fusion-Diff) is designed, which replaces traditional cross-temporal difference calculation with a "consistency difference" approach. It evaluates the differences for each temporal image before and after feature fusion with the other temporal image. This method effectively suppresses pseudo-changes caused by non-target objects while still capturing real changes.

### Multi-SSIM
A multi-scale Structure Similarity Index Measure (SSIM) is introduced as a difference metric, replacing conventional distance measures. This method calculates differences within local windows at multiple scales instead of pixel by pixel. This design enhances robustness against noise and improves the model's ability to detect structural differences across targets of various sizes。

## Citation 

 If you use this code for your research, please cite our papers.  

```
@ARTICLE{11050962,
  author={Fu, Siming and Dong, Sijun and Meng, Xiaoliang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Beyond Cross-Temporal Difference: Style-Aligned and Fusion-Difference Learning for Change Detection}, 
  year={2025},
  volume={63},
  number={},
  pages={1-15},
  keywords={Feature extraction;Remote sensing;Generative adversarial networks;Translation;Noise;Data mining;Accuracy;Transformers;Indexes;Histograms;Change detection;differential feature;pseudo-change;structural similarity index;style alignment},
  doi={10.1109/TGRS.2025.3583166}}

```
## Acknowledgments

 Our code is inspired and revised by [open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation),  [timm](https://github.com/huggingface/pytorch-image-models). Thanks  for their great work!!  
