# MMRSCD Configuration Tutorial

## Data Prepared
### In order to facilitate the use of relative paths, CDPATH is set in the ~/.bashrc file. Here is how to add this setting in the ~/.bashrc

After adding CDPATH as mentioned above, you can quickly navigate to the respective data path in the following way:
```bash
import os  
data_root = os.path.join(os.environ.get("CDPATH"), 'SYSU-CD')
```  

***

### Take SYSU-CD dataset as an example, here introduce the usage of the code.
use tools/general/write_path.py to generate a txt file for the dataset path. The format is as follows (for details, please refer to the code):
```bash
/home/user/dsj_files/CDdata/SYSU-CD/test/time1/03414.png  /home/user/dsj_files/CDdata/SYSU-CD/test/time2/03414.png  /home/user/dsj_files/CDdata/SYSU-CD/test/label/03414.png
/home/user/dsj_files/CDdata/SYSU-CD/test/time1/00708.png  /home/user/dsj_files/CDdata/SYSU-CD/test/time2/00708.png  /home/user/dsj_files/CDdata/SYSU-CD/test/label/00708.png
/home/user/dsj_files/CDdata/SYSU-CD/test/time1/03907.png  /home/user/dsj_files/CDdata/SYSU-CD/test/time2/03907.png  /home/user/dsj_files/CDdata/SYSU-CD/test/label/03907.png
/home/user/dsj_files/CDdata/SYSU-CD/test/time1/03107.png  /home/user/dsj_files/CDdata/SYSU-CD/test/time2/03107.png  /home/user/dsj_files/CDdata/SYSU-CD/test/label/03107.png
/home/user/dsj_files/CDdata/SYSU-CD/test/time1/02776.png  /home/user/dsj_files/CDdata/SYSU-CD/test/time2/02776.png  /home/user/dsj_files/CDdata/SYSU-CD/test/label/02776.png
/home/user/dsj_files/CDdata/SYSU-CD/test/time1/01468.png  /home/user/dsj_files/CDdata/SYSU-CD/test/time2/01468.png  /home/user/dsj_files/CDdata/SYSU-CD/test/label/01468.png
/home/user/dsj_files/CDdata/SYSU-CD/test/time1/00026.png  /home/user/dsj_files/CDdata/SYSU-CD/test/time2/00026.png  /home/user/dsj_files/CDdata/SYSU-CD/test/label/00026.png
/home/user/dsj_files/CDdata/SYSU-CD/test/time1/02498.png  /home/user/dsj_files/CDdata/SYSU-CD/test/time2/02498.png  /home/user/dsj_files/CDdata/SYSU-CD/test/label/02498.png
/home/user/dsj_files/CDdata/SYSU-CD/test/time1/02439.png  /home/user/dsj_files/CDdata/SYSU-CD/test/time2/02439.png  /home/user/dsj_files/CDdata/SYSU-CD/test/label/02439.png
/home/user/dsj_files/CDdata/SYSU-CD/test/time1/01057.png  /home/user/dsj_files/CDdata/SYSU-CD/test/time2/01057.png  /home/user/dsj_files/CDdata/SYSU-CD/test/label/01057.png
```  

***

## Environment
### First, you need to have a conda environment with python3.8 or above installed.
```bash
conda create --name mmrscd python=3.8
conda activate mmrscd
```  


### Make sure you have mmcv>=2.1.0 installed, and make sure your torch version matches mmcv. You can find version matching information from the following linked documents.
### <https://mmcv.readthedocs.io/zh-cn/latest/get_started/installation.html>

### For quick start, you can install them by the following command
```bash
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install torch==2.1.0+cu118 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```  
    
***

### Install mmsegmentation and complete compilation
#### mmseg=0.22.0
```bash
pip install mmsegmentation
pip install -v -e .
```  
  

***

### Please install the following dependencies in turn
#### gdal
```bash
conda install GDAL
```
#### ftfy, regex, einops, timm, kornia
```bash
pip install ftfy
pip install regex
pip install einops
pip install timm
pip install kornia
```
