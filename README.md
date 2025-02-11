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

[![GS-CPR](imgs/Method.jpg)](https://arxiv.org/abs/2408.11085)
[![GS-CPR_rel](imgs/Method_rel.jpg)](https://arxiv.org/abs/2408.11085)

## To Do:
- [ ] Finish environment setting
- [ ] Upload all pre-trained models
- [ ] Upload all scripts

will try to finish by March!

## Installation
### ACT Scaffold-GS environment
We tested our code based on CUDA 12.1, PyTorch 2.4.1, and Python 3.11+

###Install dependencies

```
cd ACT_Scaffold_GS
conda env create --file environment.yml
conda activate scaffold_gs
```

## Datasets (pretrained 3DGS models and ACT weights)
You can download the prepared DFNet/ACE/Marepo/GLACE prediction poses [link]() and unzip them in the folder `GS-CPR/coarse_poses/`.
```
coarse_poses
├── ace
├── dfnet
├── glace
├── marepo
```
You can download the pretrained 3DGS models [link]() and unzip them in the folder `GS-CPR/ACT_Scaffold_GS/data/`. 
```
ACT_Scaffold_GS
├── data
│   ├── cambridge
│   ├── 7scenes
│   ├── 12scenes
```
And then run the below command to render the synthetic images based on the `coarse_poses`.
```
# generate rendered images based on coarse poses for 7Scenes
bash script_render_pred_7s.sh
# generate rendered images based on coarse poses for 7Scenes
bash script_render_pred_12s.sh
# generate rendered images based on coarse poses for Cambridge Landmarks
bash script_render_pred_cam.sh
```


## GS-CPR refinement environment
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

### GS-CPR refinement
```
#For 7Scenes
python gs_cpr_7s.py --pose_estimator ace --scene chess

#For 12Scenes
python gs_cpr_12s.py --pose_estimator ace --scene apt1_kitchen

#For Cambridge Landmarks
python gs_cpr_cam.py --pose_estimator ace --scene ShopFacade
```


## Acknowledgements
This project is developed based on several fantastic repos: [Scaffold-GS](https://github.com/city-super/Scaffold-GS), [MASt3R](https://github.com/naver/mast3r), [NeFeS](https://github.com/ActiveVisionLab/NeFeS), [ACE](https://github.com/nianticlabs/ace) and [Depth for 3DGS](https://github.com/leo-frank/diff-gaussian-rasterization-depth). We thank the original authors for their excellent work.