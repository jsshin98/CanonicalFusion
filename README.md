# [ECCV 2024] CanonicalFusion: Generating Drivable 3D Human Avatars from Multiple Images

**CanonicalFusion: Generating Drivable 3D Human Avatars from Multiple Images (ECCV 2024 ACCEPTED !!)**

**Jisu Shin**, Junmyeong Lee, Seongmin Lee, Min-Gyu Park, Ju-Mi Kang, Ju Hong Yoon, and Hae-Gon Jeon

**[[PAPER]](https://arxiv.org/abs/2407.04345)**

Not a completed version. Currently on updating..

## Introduction

__CanonicalFusion.__ We propose CanonicalFusion, which takes multiple frames and generate drivable 3D avatar by integrating individual reconstruction results into the canonical space.

## Getting Started
### Prerequisites (Recommended Version)

- Python 3.9
- Pytorch 2.0.1
- Cuda 11.8

### Installation
To run the Demo, it is suggested to install the Conda environment as detailed below (check the compatibility with your version, especially for pytorch-lightning and torchmetrics):
```
conda create -n canonicalfusion python=3.9
conda activate canonicalfusion
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install scikit-image scikit-learn pillow numpy matplotlib trimesh fvcore smplx timm tensorboard pytorch_msssim loguru pyrender open3d pickle5 albumentations people-segmentation pymeshlab rembg meshzoo ninja
(Optional) apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0 libgl1-mesa-glx 
pip install opencv-python
pip install torchmetrics==0.7 pytorch-lightning==2.1
pip install git+https://github.com/tatsy/torchmcubes.git
conda install -c conda-forge pyembree
```
Install pytorch3d
```
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .
```
Install pyremesh (refer to Chupa:https://github.com/snuvclab/chupa)
```
python -m pip install --no-index --find-links ./diff_renderer/src/normal_nds/ext/pyremesh pyremesh
```
Install Flexicubes setting (Kaolin version check: https://kaolin.readthedocs.io/en/latest/notes/installation.html)
```
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu118.html
```
Install Pyopengl
```
git clone https://github.com/mcfletch/pyopengl
cd pyopengl
pip install -e .
cd accelerate
pip install -e .
```

## Dataset
Our model requires fitted SMPL-X for each image as an input. We provide some examples that are compatible to our model. Note that the pelvis of the SMPL-X should be nearly centered to generate the plausible reconstruction results since we train our model on this setting.

- Download SMPL-X and place it in resources/smpl_models
  - SMPL-X : https://smpl-x.is.tue.mpg.de/
  
## Inference
### Stage 1: Initial reconstruction from a single view
```
cd ./apps
python 01_human_recon_eval.py
```
### Stage 2: Refine canonical model with multiple frames
You need to manually set the frame number that you want to use on refinement. (Will be revised soon)
```
cd ./apps
python 02_canonical_fusion.py
```

### Citation
Will be updated soon

### Contact
If you have any question, please feel free to contact us via jsshin98@gm.gist.ac.kr.

Coming Soon! :)
