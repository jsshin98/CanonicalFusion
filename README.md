# [ECCV 2024] CanonicalFusion: Generating Drivable 3D Human Avatars from Multiple Images

**CanonicalFusion: Generating Drivable 3D Human Avatars from Multiple Images (ECCV 2024 ACCEPTED 🎉)**

**Jisu Shin**, Junmyeong Lee, Seongmin Lee, Min-Gyu Park, Ju-Mi Kang, Ju Hong Yoon, and Hae-Gon Jeon

**[[Arxiv]](https://arxiv.org/abs/2407.04345)**
**[[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-73337-6_3)**

## 📣 News & TODOs
- [x] Release arxiv paper and github page!
- [x] Release code and pretrained weights for single image-based 3D human reconstruction! (First phase)
- [x] Release canonical mesh reconstruction code with multiple images via differentiable rendering. (Second phase)

## 💡 Introduction
Our current version contains the **inference code & pretrained weights for 3D human mesh reconstruction** that takes input image and fitted SMPL-X depth map. You can use them for **single image-based 3D human reconstruction evaluation**. We also released the second phase, integrating multiple meshes on canonical space, but it is not well-organized to use so I'd like you to take this as a reference only.

## ⚙️Getting Started
### Prerequisites (Recommended Version)

- Python 3.8
- Pytorch 2.1.0
- Cuda 12.1
- Linux / Ubuntu Environment

### Environment Setting
#### 1. Clone CanonicalFusion
```
git clone https://github.com/jsshin98/CanonicalFusion.git
cd CanonicalFusion
```

#### 2. Basic Installation
You first need to build Docker image with [nvdiffrast](https://nvlabs.github.io/nvdiffrast/#linux). 
(This is not required to current evaluation code, but **required for differentiable rendering step**, which will also be published soon.)

Then, it is suggested to install the Conda environment inside the Docker as detailed below:
```
# Create conda environment inside docker
conda create -n canonicalfusion python=3.10
conda activate canonicalfusion

# Install Pytorch and other dependencies
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0 libgl1-mesa-glx 
pip install -r requirements.txt
pip install git+https://github.com/tatsy/torchmcubes.git
conda install -c conda-forge pyembree
```

#### 3. Additional Installation for pytorch3d, pyopengl, nvdiffrast, kaolin (necessary for differentiable rendering parts - second phase)
Please check the [compatibility](https://kaolin.readthedocs.io/en/latest/notes/installation.html) and modify it based on your environment (cuda & torch)
```
# Install pytorch3d
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .

# Install PyOpenGL
git clone https://github.com/mcfletch/pyopengl
cd pyopengl
pip install -e .
cd accelerate
pip install -e .

# Install pyremesh (refer to Chupa:https://github.com/snuvclab/chupa)
python -m pip install --no-index --find-links ./diff_renderer/normal_nds/ext/pyremesh pyremesh

# Install nvdiffrast and kaolin 
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu121.html 
```

## 🧰 Models & Dataset
### Download pretrained models and extra data
Currently, we provide a set of pretrained models for the inference. Models and sample dataset for inference can be downloaded [here](https://drive.google.com/drive/folders/1zK_XZ6KGb5h7UNy0gMLVHBjgJPXHl-a5?usp=sharing). **Note that we also publish the checkpoints not using SMPLX for your convenience on evaluation. Check ```DEPTH_LBS_WO_SMPLX & COLOR_WO_SMPLX``` on the above link. Of course you have to modify the network a bit not to take the SMPLX guidance as input. This will decrease the performance a bit but there should be no big difference.**
Please check the dataset tree below. For the rendering code, we provide example scripts to render images so check it if you need them (you might modify the code according to our needs.) 

Our model requires fitted SMPL-X for each image as an input. We provide some examples that are compatible to our model. Note that the pelvis of the SMPL-X should be nearly centered to the origin and height should be 180 to generate the plausible reconstruction results since we train our model on this setting. We follow the rendering process of [PIFu](https://github.com/shunsukesaito/PIFu) to render the depth map of fitted SMPL-X.

Set the dataset path on ```apps/canonfusion_eval.yaml``` data_dir argument.

### Dataset Tree
You need to organize the dataset as following:
```
YOUR DATASET PATH
├── dataset_name
│   └── IMG
│       └── 0001
│         └── 000_front.png
│         └── 030_front.png
│       └── 0002
│         └── 000_front.png
│         └── ...
│   └── MASK (optional. If you don't have mask, you can just add rembg to get one.)
│       └── 0001
│         └── 000_front.png
│         └── 030_front.png
│       └── 0002
│         └── 000_front.png
│         └── ...
│   └── DEPTH
│       └── 0001
│         └── 000_front.png
│         └── 000_back.png
│         └── 030_front.png
│         └── 030_back.png
│       └── 0002
│         └── 000_front.png
│         └── 000_back.png
│         └── ...
│   └── SMPLX
│       └── 0001
│         └── 0001.json
│         └── 0001.obj
│       └── 0002
│         └── 0002.json
│         └── 0002.obj
│       └── ...
├── resource
│   └──smpl_models
│       └── smplx
│   └──pretrained_models
│       └── lbs_ckpt
│           └── best.tar
│       └── main_ckpt
│           └── COLOR
│           └── DEPTH_LBS
│       └── Real-ESRGAN
```

## 🔎 Inference
### Stage 1: Initial reconstruction from a single view
```
cd ./apps
python 01_human_recon_eval.py
```
### Stage 2: Refine canonical model with multiple frames - Still fixing.. Not complete.
You need to manually set the frame number that you want to use on refinement. You need to know the view angle of all the frames and put the frame numbers that you want to integrate on ```apps/canonfusion_eval.yaml``` file.
```
cd ./apps
python 02_canonical_fusion.py
```

## ✏️ Citation
If you find our work meaningful, please consider the citation:
```
@inproceedings{shin2025canonicalfusion,
  title={CanonicalFusion: Generating Drivable 3D Human Avatars from Multiple Images},
  author={Shin, Jisu and Lee, Junmyeong and Lee, Seongmin and Park, Min-Gyu and Kang, Ju-Mi and Yoon, Ju Hong and Jeon, Hae-Gon},
  booktitle={European Conference on Computer Vision},
  pages={38--56},
  year={2025},
  organization={Springer}
}
```

## 📱 Contact
If you have any question, please feel free to contact us via jsshin98@gm.gist.ac.kr.
