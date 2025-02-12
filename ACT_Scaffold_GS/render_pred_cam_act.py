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
import torchvision.transforms as transforms
import numpy as np
import root_file_io as fio

import time
from scene.cameras import Camera, VirtualCamera2
import matplotlib.pyplot as plt
from scene.nerfh_nff import create_nerf
from dataset_loaders.utils.color import rgb_to_yuv

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

    log_path = fio.createPath(fio.sep, [session_dir, 'render_log_' + str(current_timestamp()) + '.txt'])
    render_path = fio.createPath(fio.sep, [session_dir, 'render_single_view'])
    makedirs(render_path, exist_ok=True)

    device = torch.device('cuda')

    psnr_value = 0
    l1_loss_value = 1
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt_image_name = view.image_name.replace('_frame','/frame')
        gt_original_image_path = os.path.join(source_path, 'images', gt_image_name)
        gt_image = Image.open(gt_original_image_path)
        transform = transforms.Compose([
        transforms.ToTensor(),  # 将图片转换为张量，并且归一化到 [0, 1] 之间
    ])

    # 将图片应用转换
        gt_image = transform(gt_image).cuda()
        yuv = rgb_to_yuv(gt_image)
        y_img = yuv[0] # extract y channel only
        hist = torch.histc(y_img, bins=10, min=0., max=1.) # compute intensity histogram
        hist = hist/(hist.sum())*100 # convert to histogram density, in terms of percentage per bin
        hist = torch.round(hist).unsqueeze(0).cuda()
        rendering = render_dof(view, gaussians_na, pipeline, background)["render"]
        if args.encode_hist:
            affine_color_transform = render_kwargs_test['network_fn'].affine_color_transform
            rgb = affine_color_transform(args, rendering, hist, 1)
        depth = render_dof(view, gaussians_na, pipeline, background)["depth"]
        

        save_path = fio.createPath(fio.sep, [render_path], gt_image_name)
        (savedir, savename, saveext) = fio.get_filename_components(save_path)
        fio.ensure_dir(savedir)
        torchvision.utils.save_image(rgb.reshape(3,1080,1920), os.path.join(render_path, gt_image_name))
        #torchvision.utils.save_image(rgb.reshape(3,480,854), os.path.join(render_path, gt_image_name))
        depth_array = depth.cpu().numpy()
        '''
        depth_min = depth_array.min()
        depth_max = depth_array.max()
        depth_normalized = (depth_array - depth_min) / (depth_max - depth_min)
        #depth_normalized_inv = 1.0 - depth_normalized
        
        depth_image = (depth_normalized * 255).astype(np.uint8)
        plt.imshow(depth_image, cmap='plasma_r')
        plt.axis('off')
        plt.savefig(os.path.join(render_path, gt_image_name.replace('.png','_depth.png')), bbox_inches='tight', pad_inches=0)
        '''
        np.save(os.path.join(render_path, gt_image_name.replace('.png','.npy')), depth_array)
        


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
        pretrain_tag = '_'.join(combo[0:2])
        scene = SceneAnchor(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        camera_intrin_params =  [1920, 1080, 1673, 960, 540]
        #camera_intrin_params =  [854, 480, 836, 427, 240], resized images in ace/datasets
        camera_model = 'SIMPLE_PINHOLE'
        width = camera_intrin_params[0]
        height = camera_intrin_params[1]
        focal_length_path = f'../../ace/datasets/Cambridge_{args.render_scene}/test/calibration/'
        render_pose_path = f"../coarse_poses/{args.pose_estimator}/Cambridge/poses_Cambridge_{args.render_scene}_.txt"
        with open(render_pose_path, 'r') as file:
            lines = file.readlines()

        views = []
        for line in lines:
            image_name = line.split()[0]
            qvec =  [float(value) for value in line.split()[1:5]]
            tvec = [float(value) for value in line.split()[5:8]]
            R = np.transpose(qvec2rotmat(qvec))
            T = np.array(tvec)
            focal_length = np.loadtxt(focal_length_path + image_name.replace('.png','.txt').replace('/frame','_frame'))
            if camera_model=="SIMPLE_PINHOLE":
                FovY = focal2fov(focal_length * 2.25, height) #*1 if render image in ace preprocess size
                FovX = focal2fov(focal_length * 2.25, width)  #*1 if render image in ace preprocess size
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
    parser.add_argument("--multires_views", type=int, default=4, help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--multires", type=int, default=10, help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--i_embed", type=int, default=0, help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--reduce_embedding", type=int, default=-1, help='fourier embedding mode: -1: paper default, \
                                                                        0: reduce by half, 1: remove embedding, 2: DNeRF embedding')
    parser.add_argument("--use_viewdirs", action='store_true', default=True, help='use full 5D input instead of 3D')
    parser.add_argument("--netdepth", type=int, default=8, help='layers in network')
    parser.add_argument("--netwidth", type=int, default=128, help='channels per layer')
    parser.add_argument("--use_fusion_res", default=False, action='store_true', help='add residual connection to the fusion block')
    parser.add_argument("--no_fusion_BN", default=False, action='store_true', help='no batchnorm for fusion block')
    parser.add_argument("--multi_gpu", action='store_true', help='use multiple gpu on the server')
    parser.add_argument("--N_importance", type=int, default=64,help='number of additional fine samples per ray')
    parser.add_argument("--N_samples", type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument("--NeRFW", action='store_true', default=True, help='new implementation for NeRFW')
    parser.add_argument("--in_channels_a", type=int, default=50, help='appearance embedding dimension, hist_bin*N_a when embedding histogram')
    parser.add_argument("--in_channels_t", type=int, default=20, help='transient embedding dimension, hist_bin*N_tau when embedding histogram')
    parser.add_argument("--netchunk", type=int, default=2097152, help='number of pts sent through network in parallel, defualt is 2^21, consider reduce this if GPU OOM')
    parser.add_argument("--no_grad_update", default=False, help='do not update nerf in training')
    parser.add_argument("--basedir", type=str, default='./logs/', help='where to store ckpts and logs')
    parser.add_argument("--expname", type=str, default='paper_models', help='experiment name')
    parser.add_argument("--act_itr", type=str, default='30000', help='act_models_iteration')
    parser.add_argument("--ft_path", type=str, default=f'./logs/paper_models/', help='specific weights npy file to reload for coarse network')
    parser.add_argument("--perturb", type=float, default=1.,help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--white_bkgd", action='store_true', help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--raw_noise_std", type=float, default=0., help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--dataset_type", type=str, default='Cambridge', help='options: llff / blender / deepvoxels')
    parser.add_argument("--lindisp", action='store_true', help='sampling linearly in disparity rather than depth')
    #parser.add_argument("--datadir", type=str, default='./data/Cambridge/ShopFacade', help='input data directory')
    parser.add_argument("--df", type=float, default=2., help='image downscale factor')
    parser.add_argument("--hist_bin", type=int, default=10, help='image histogram bin size')
    parser.add_argument("--semantic", action='store_true', default=False, help='using semantic segmentation mask to remove temporal noise at training time')
    parser.add_argument("--encode_hist", default=True, action='store_true', help='encode histogram instead of frame index')
    parser.add_argument("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--ffmlp", action='store_true', default=False, help='using NeRF with tiny-cuda-nn acceleration on mlp')
    parser.add_argument("--tcnn", action='store_true', default=False, help='using NeRF with tiny-cuda-nn acceleration, with hash encoding')
    parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')

    args = get_combined_args(parser)
    print("args: ",args)
    print("Rendering " + args.model_path + ' for testing set ' + args.source_path)
    print(f"----------render images for {args.pose_estimator}----------")

    virtual_pipeline = VirtualPipelineParams2()
    
    render_sets_virtual2(model.extract(args), args.iteration, virtual_pipeline)