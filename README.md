# GS-CPR: Efficient Camera Pose Refinement via 3D Gaussian Splatting
**[Changkun Liu](https://lck666666.github.io/),
[Shuai Chen](https://scholar.google.com/citations?user=c0xTh_YAAAAJ&hl=en), 
[Yash Bhalgat](https://scholar.google.com/citations?user=q0VSEHYAAAAJ&hl=en),
[Siyan Hu](https://scholar.google.com/citations?user=S56rLU4AAAAJ&hl=en), 
[Ming Cheng](https://scholar.google.com/citations?user=MPyUxv4AAAAJ&hl=en),
[Zirui Wang](https://scholar.google.com/citations?user=zCBKqa8AAAAJ&hl=en), 
[Victor Prisacariu](https://scholar.google.com/citations?user=GmWA-LoAAAAJ&hl=en) 
and [Tristan BRAUD](https://scholar.google.com/citations?user=ZOZtoQUAAAAJ&hl=en)**

**International Conference on Learning Representations (ICLR) 2025**

**[Project Page](https://xrim-lab.github.io/GS-CPR/) | [Paper](https://openreview.net/forum?id=mP7uV59iJM)**

[![GS-CPR](framework_imgs/Method.jpg)](https://arxiv.org/abs/2408.11085)
[![GS-CPR_rel](framework_imgs/Method_rel.jpg)](https://arxiv.org/abs/2408.11085)

## To Do:
- [x] Finish environment setting
- [x] Upload all scripts
- [x] Upload pre-trained models (Cambridge Landmarks)
- [ ] Upload pre-trained models (7scenes)
- [ ] Upload pre-trained models (12scenes)

will try to finish by March!

## Installation
### ACT Scaffold-GS environment
We tested our code based on CUDA 12.1, PyTorch 2.5.1, and Python 3.11+

### Install dependencies for ACT Scaffold-GS rendering

```
cd ACT_Scaffold_GS
conda create -n scaffold_act python=3.11
conda activate scaffold_act
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia 
pip install -r requirements.txt

# install Tiny-cuda-nn
pip install ninja
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# install depth rendering for 3DGS
git clone git@github.com:leo-frank/diff-gaussian-rasterization-depth.git
cd diff-gaussian-rasterization-depth
python setup.py install
```

## Datasets (pretrained 3DGS models and ACT weights)
You can download the pretrained 3DGS models from the provided [link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/cliudg_connect_ust_hk/ElfOnz0vRm9Ot6j47CDzFaoBJrGKoqKGLfb6xYSuMwf7WQ?e=Rrc98i) and unzip them in the folder `GS-CPR/ACT_Scaffold_GS/data/`. You can download pretrained ACT MLP models from the provided [link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/cliudg_connect_ust_hk/ElfOnz0vRm9Ot6j47CDzFaoBJrGKoqKGLfb6xYSuMwf7WQ?e=Rrc98i) and put them in the folder `GS-CPR/ACT_Scaffold_GS/logs/`.
```
ACT_Scaffold_GS
├── data
│   ├── cambridge
│   ├── 7scenes
│   ├── 12scenes
├── logs
|   ├──paper_models
```

For 7scenes COLMAP files, we improves the accuracy of the [sparse point cloud](https://github.com/tsattler/visloc_pseudo_gt_limitations) using dense depth maps in [HLoc](https://github.com/cvg/Hierarchical-Localization/tree/master/hloc/pipelines/7Scenes) tool box courtesy of Eric Brachmann for [DSAC*](https://github.com/vislearn/dsacstar). For 12scenes COLMAP files, we directly use SfM models provided by [link](https://github.com/tsattler/visloc_pseudo_gt_limitations). For Cambridge Landmarks, we use SfM models from [HLoc](https://github.com/cvg/Hierarchical-Localization/tree/master/hloc/pipelines/Cambridge) toolbox, courtesy of Torsten Sattler.

And then run the below command to render the synthetic images based on the `coarse_poses`.
```
# generate rendered images based on coarse poses for 7Scenes
bash script_render_pred_7s.sh
# generate rendered images based on coarse poses for 12Scenes
bash script_render_pred_12s.sh
# generate rendered images based on coarse poses for Cambridge Landmarks
bash script_render_pred_cam.sh
```


## Install dependencies for GS-CPR refinement
Create the environment as same as [MASt3R](https://github.com/naver/mast3r#demo)
```
cd GS-CPR
conda activate mast3r
```

## Datasets (raw images + poses)
This paper uses three public datasets:
- [Microsoft 7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
- [Cambridge Landmarks](https://www.repository.cam.ac.uk/handle/1810/251342/)
- [Stanford 12-Scenes](https://graphics.stanford.edu/projects/reloc/)

Following [ACE](https://github.com/nianticlabs/ace), we utilize the same scripts in the `datasets` folder to automatically download and extract the data in a consistent format.

> **Important: make sure you have checked the license terms of each dataset before using it.**

#### {7, 12}-Scenes:

You can use the `datasets/setup_{7,12}scenes.py` scripts to download the data.
As mentioned in our paper, we experimented _Pseudo Ground Truth (PGT)_ camera poses obtained after running SfM on the scenes (see the [ICCV 2021 paper](https://openaccess.thecvf.com/content/ICCV2021/html/Brachmann_On_the_Limits_of_Pseudo_Ground_Truth_in_Visual_Camera_ICCV_2021_paper.html),
and [associated code](https://github.com/tsattler/visloc_pseudo_gt_limitations/) for details).

To download and prepare the datasets using the PGT poses:

```shell
cd datasets
# Downloads the data to datasets/pgt_7scenes_{chess, fire, ...}
./setup_7scenes.py --poses pgt
# Downloads the data to datasets/pgt_12scenes_{apt1_kitchen, ...}
./setup_12scenes.py --poses pgt
``` 
You can follow [ACE](https://github.com/nianticlabs/ace) to download DSLAM poses and try.

#### Cambridge Landmarks

Simply run:

```shell
cd datasets
# Downloads the data to datasets/Cambridge_{GreatCourt, KingsCollege, ...}
./setup_cambridge.py
```

## GS-CPR refinement
```
#For 7Scenes
python gs_cpr_7s.py --pose_estimator ace --scene chess #for a specific scene
python gs_cpr_7s.py --pose_estimator ace --test_all #for the whole dataset

#For 12Scenes
python gs_cpr_12s.py --pose_estimator ace --scene apt1_kitchen
python gs_cpr_12s.py --pose_estimator ace --test_all #for the whole dataset

#For Cambridge Landmarks
python gs_cpr_cam.py --pose_estimator ace --scene ShopFacade
python gs_cpr_cam.py --pose_estimator ace --test_all #for the whole dataset
```
You can check the refined poses for each query in `txt` files and the statistic `log` results in `GS-CPR/outputs`.
## Citation
If you find our work helpful, please consider citing:

```bibtex
@inproceedings{
liu2025gscpr,
title={{GS}-{CPR}: Efficient Camera Pose Refinement via 3D Gaussian Splatting},
author={Changkun Liu and Shuai Chen and Yash Sanjay Bhalgat and Siyan HU and Ming Cheng and Zirui Wang and Victor Adrian Prisacariu and Tristan Braud},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=mP7uV59iJM}
}
```

## Acknowledgements
This project is developed based on several fantastic repos: [Scaffold-GS](https://github.com/city-super/Scaffold-GS), [MASt3R](https://github.com/naver/mast3r), [NeFeS](https://github.com/ActiveVisionLab/NeFeS), [ACE](https://github.com/nianticlabs/ace) and [Depth for 3DGS](https://github.com/leo-frank/diff-gaussian-rasterization-depth). We thank the original authors for their excellent work.