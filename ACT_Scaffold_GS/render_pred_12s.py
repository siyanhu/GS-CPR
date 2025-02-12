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

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


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
    
def calculate_resolution(camera):
    orig_w, orig_h = camera.image_width, camera.image_height
    resolution_scale = 1

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    return resolution
    

def render_set_virtual2(source_path, model_path, name, views, gaussians_na, pipeline, background):

    model_position_combo = model_path.split(fio.sep)
    if len(model_position_combo) < 3:
        return
    model_name = model_position_combo[-2]
    model_setting = model_position_combo[-1]
    session_dir = fio.createPath(fio.sep, [source_path, name, '_'.join([model_name, model_setting])])

    render_path = fio.createPath(fio.sep, [session_dir, 'render_single_view'])
    makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt_image_name = view.image_name

        rendering = render_dof(view, gaussians_na, pipeline, background)["render"]
        depth = render_dof(view, gaussians_na, pipeline, background)["depth"]
        save_path = fio.createPath(fio.sep, [render_path], gt_image_name)
        (savedir, savename, saveext) = fio.get_filename_components(save_path)
        fio.ensure_dir(savedir)
        print(os.path.join(render_path, gt_image_name))
        print("render path: ",render_path)
        torchvision.utils.save_image(rendering, os.path.join(render_path, gt_image_name))
        depth_array = depth.cpu().numpy()
        '''
        depth_min = depth_array.min()
        depth_max = depth_array.max()
        depth_normalized = (depth_array - depth_min) / (depth_max - depth_min)

        # 转换为 8 位图像
        depth_image = (depth_normalized * 255).astype(np.uint8)
        plt.imshow(depth_image, cmap='viridis')
        plt.savefig(os.path.join(render_path, gt_image_name.replace('.png','_depth.png')))
        '''
        np.save(os.path.join(render_path, gt_image_name.replace('.jpg','.npy')), depth_array)
        


def render_sets_virtual2(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        print("Cuda current device: ", torch.cuda.current_device())
        print("Cuda is avail: ", torch.cuda.is_available())
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        pretrain_source = dataset.model_path
        combo = pretrain_source.split('/')
        scene = SceneAnchor(dataset, gaussians, load_iteration=iteration, shuffle=False)
        focal_length_dict = {'apt1_kitchen':1167.8, 'apt1_living':1172.29, 'apt2_bed':1166.72,'apt2_kitchen':1169.57, 'apt2_living':1166.41,'apt2_luke':1160.96,'office1_gates362':1170.08, 'office1_gates381':1168.02, 'office1_lounge':1165.19,'office1_manolis':1168.41,'office2_5a':1139.32,'office2_5b':1161.54}
        fl  = focal_length_dict[args.render_scene]
        camera_intrin_params =  [1296, 968, fl, 648, 484]
        camera_model = 'SIMPLE_PINHOLE'
        width = camera_intrin_params[0]
        height = camera_intrin_params[1]
        focal_length_x = camera_intrin_params[2]
        render_scene = args.render_scene
        with open(args.render_images + f'poses_pgt_12scenes_{render_scene}_.txt', 'r') as file:
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
    virtual_pipeline = VirtualPipelineParams2()
    print(f"----------render images for {args.pose_estimator}----------")
    render_sets_virtual2(model.extract(args), args.iteration, virtual_pipeline)