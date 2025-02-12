import torch
from scene import SceneAnchor
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_dof, render

import torchvision
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, VirtualPipelineParams2, get_combined_args
from gaussian_renderer import GaussianModel
from utils.graphics_utils import focal2fov
from PIL import Image
import numpy as np
import time
import root_file_io as fio

from scene.cameras import Camera, VirtualCamera2
import matplotlib.pyplot as plt


import logging
_logger = logging.getLogger(__name__)

def qvec2rotmat(qvec):
    # Ensure qvec is a unit quaternion
    q0, q1, q2, q3 = qvec  # quaternion components
    return np.array([
        [1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1**2 + q2**2)]
    ])


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def current_timestamp(micro_second=False):
    t = time.time()
    if micro_second:
        return int(t * 1000 * 1000)
    else:
        return int(t * 1000)
    

def render_set_virtual2(source_path, model_path, name, views, gaussians_na, pipeline, background):
    model_position_combo = model_path.split(fio.sep)
    if len(model_position_combo) < 3:
        return
    model_name = model_position_combo[-2]
    model_setting = model_position_combo[-1]
    session_dir = fio.createPath(fio.sep, [source_path, name, '_'.join([model_name, model_setting])])

    log_path = fio.createPath(fio.sep, [session_dir, 'render_log_' + str(current_timestamp()) + '.txt'])
    render_path = fio.createPath(fio.sep, [session_dir, 'render_single_view'])
    makedirs(render_path, exist_ok=True)

    device = torch.device('cuda')

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt_image_name = view.image_name
        rendering = render_dof(view, gaussians_na, pipeline, background)["render"]
        depth = render_dof(view, gaussians_na, pipeline, background)["depth"]
        gt_original_image_path = os.path.join(source_path, 'images', gt_image_name)
        save_path = fio.createPath(fio.sep, [render_path], gt_image_name)
        (savedir, savename, saveext) = fio.get_filename_components(save_path)
        fio.ensure_dir(savedir)
        torchvision.utils.save_image(rendering, os.path.join(render_path, gt_image_name))
        depth_array = depth.cpu().numpy()
        '''
        depth_min = depth_array.min()
        depth_max = depth_array.max()
        depth_normalized = (depth_array - depth_min) / (depth_max - depth_min)
        plt.axis('off')
        # 转换为 8 位图像
        depth_image = (depth_normalized * 255).astype(np.uint8)
        plt.imshow(depth_image, cmap='plasma_r')
        plt.savefig(os.path.join(render_path, gt_image_name.replace('.png','_depth.png')), bbox_inches='tight', pad_inches=0)
        plt.close() 
        '''
        np.save(os.path.join(render_path, gt_image_name.replace('.png','.npy')), depth_array)
       


def render_sets_virtual2(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        print("Cuda is avail: ", torch.cuda.is_available())
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        pretrain_source = dataset.model_path
        combo = pretrain_source.split('/')
        
        scene = SceneAnchor(dataset, gaussians, load_iteration=iteration, shuffle=False)
        focal_length_dict = {'chess':526.22, 'fire':526.903, 'heads':527.745, 'office':525.143, 'pumpkin':525.647, 'redkitchen':525.505, 'stairs':525.505}
        fl  = focal_length_dict[args.render_scene]

        #fl = 585.0 #for depth evaluation
        camera_intrin_params =  [640, 480, fl, 320, 240]
        camera_model = 'SIMPLE_PINHOLE'
        width = camera_intrin_params[0]
        height = camera_intrin_params[1]
        focal_length_x = camera_intrin_params[2]
        render_scene = args.render_scene
        
        render_pose_path = f"../coarse_poses/{args.pose_estimator}/7Scenes_pgt/poses_pgt_7scenes_{render_scene}_.txt"
        with open(render_pose_path, 'r') as file:
            lines = file.readlines()

        views = []
        for line in lines:
            image_name = line.split()[0]
            qvec =  [float(value) for value in line.split()[1:5]]
            tvec = [float(value) for value in line.split()[5:8]]
            R = np.transpose(qvec2rotmat(qvec))
            T = np.array(tvec)
            if camera_model=="SIMPLE_PINHOLE":
                FovY = focal2fov(focal_length_x, height)
                FovX = focal2fov(focal_length_x, width)
            view = VirtualCamera2(colmap_id=1, uid=0, R=R, T=T, FoVx=FovX, FoVy=FovY, width=width, height=height, image_name=image_name)
            views.append(view)
        render_set_virtual2(dataset.source_path, dataset.model_path, f"evaluate_{args.pose_estimator}", views, gaussians, pipeline, background)

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    print("model: ",model.feat_dim)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_scene", type=str)
    parser.add_argument("--pose_estimator", type=str)
    args = get_combined_args(parser)
    print("args: ",args)
    print("Rendering " + args.model_path + ' for testing set ' + args.source_path)
    print(f"----------render images for {args.pose_estimator}----------")
    virtual_pipeline = VirtualPipelineParams2()
    render_sets_virtual2(model.extract(args), args.iteration, virtual_pipeline)