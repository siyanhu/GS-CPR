from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
import math
import cv2
import os

from utils.functions import *

import logging
_logger = logging.getLogger(__name__)

if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    parser = ArgumentParser(description="GS-CPR for pose estimators")
    parser.add_argument("--pose_estimator", default="ace",choices=["ace","marepo","glace","dfnet"], type=str)
    parser.add_argument("--scene", default="chess", type=str)
    parser.add_argument("--test_all", action='store_true', default=False)
    args = parser.parse_args()
    original_size = (480, 640)
    pe = args.pose_estimator
    if args.test_all:
        SCENES = ['chess','fire','heads','office','pumpkin','redkitchen','stairs']
    else:
        SCENES = [args.scene]
        
    focal_length_dict = {'chess':526.22, 'fire':526.903, 'heads':527.745, 'office':525.143, 'pumpkin':525.647, 'redkitchen':525.505, 'stairs':525.505}
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    log_path = f"./outputs/7scenes/GS_CPR_{pe}_results/" 
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        print(f"Directory {log_path} created.")
    else:
        print(f"Directory {log_path} already exists.")
    for SCENE in tqdm(SCENES):
        logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日志格式
        filename= log_path + f'logs_{SCENE}.log',  # 日志文件名
        filemode='w'  # 写入模式，'w' 表示覆盖，'a' 表示追加
        )
        _logger = logging.getLogger(__name__)

        fl = focal_length_dict[SCENE]
        # load_images can take a list of images or a directory
        query_path = f'./datasets/pgt_7scenes_{SCENE}/test/rgb/'
        rendered_path = f'./ACT_Scaffold_GS/data/7scenes/scene_{SCENE}/test/evaluate_{pe}/train_output/render_single_view/'

        predict_pose_w2c_path = f'./coarse_poses/{pe}/7Scenes_pgt/poses_pgt_7scenes_{SCENE}_.txt'
        gt_pose_c2w_path = f'./datasets/pgt_7scenes_{SCENE}/test/poses/'
        gs_depth_path = rendered_path

        gt_pose_c2w_dict = {}
        predict_pose_w2c_dict = {}
        images_list = []
        for filename in os.listdir(query_path):
            if filename.endswith('.png'):
                images_list.append(filename)

        images_list.sort()
        for img_name in images_list:
            pose_file_name = gt_pose_c2w_path + img_name.replace('.color.png','.pose.txt')
            c2w_pose = np.loadtxt(pose_file_name)
            gt_pose_c2w_dict[img_name] = c2w_pose
            if pe == 'dfnet':
                predict_w2c_ini= getPredictPos(img_name.replace('-frame','/frame'),predict_pose_w2c_path)
            else:
                predict_w2c_ini= getPredictPos(img_name,predict_pose_w2c_path)
            predict_pose_w2c_dict[img_name] = predict_w2c_ini
            
        
        results_ini = []
        results_final = []
        

        refine_results_path = log_path +  "refine_predictions/" 
        if not os.path.exists(refine_results_path):
            os.makedirs(refine_results_path)
            print(f"Directory {refine_results_path} created.")
        else:
            print(f"Directory {refine_results_path} already exists.")
        ransac_time = 0
        with open(refine_results_path + f'{pe}_refinew2c_mast3r_{SCENE}.txt', 'w') as f:
            for image in tqdm(images_list):   
                try: 
                    image1 = rendered_path + image
                    image2 = query_path + image
                    images = load_images([image1, image2], size=512)
                except:
                    image1 = rendered_path + image.replace('-frame','/frame')
                    image2 = query_path + image
                    images = load_images([image1, image2], size=512)

                images = load_images([image1, image2], size=512)
                output = inference([tuple(images)], model, device, batch_size=1, verbose=False)
                view1, pred1 = output['view1'], output['pred1']
                view2, pred2 = output['view2'], output['pred2']

                desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

                # find 2D-2D matches between the two images
                matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                            device=device, dist='dot', block_size=2**13)

                # ignore small border around the edge
                H0, W0 = view1['true_shape'][0]
                
                valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
                    matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

                H1, W1 = view2['true_shape'][0]
                valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
                    matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

                valid_matches = valid_matches_im0 & valid_matches_im1
                matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]
                scale_x = original_size[1] / W0.item()
                scale_y = original_size[0] / H0.item()
                for pixel in matches_im1:
                    pixel[0] *= scale_x
                    pixel[1] *= scale_y
                for pixel in matches_im0:
                    pixel[0] *= scale_x
                    pixel[1] *= scale_y
                try:
                    depth_map = np.load(gs_depth_path+image.replace('png','npy').replace('-frame','/frame'))
                except:
                    depth_map = np.load(gs_depth_path+image.replace('png','npy'))
                fx, fy, cx, cy = fl, fl, original_size[1]/2, original_size[0]/2  # Example values for focal lengths and principal point
                K = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
                dist_eff = np.array([0,0,0,0], dtype=np.float32)
                predict_c2w_ini = np.linalg.inv(predict_pose_w2c_dict[image])
                predict_w2c_ini = predict_pose_w2c_dict[image]
                initial_rvec, _ = cv2.Rodrigues(predict_c2w_ini[:3,:3].astype(np.float32))
                initial_tvec = predict_c2w_ini[:3,3].astype(np.float32)
                gt_c2w_pose = gt_pose_c2w_dict[image]
                K_inv = np.linalg.inv(K)
                height, width = depth_map.shape
                x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
                x_flat = x_coords.flatten()
                y_flat = y_coords.flatten()
                depth_flat = depth_map.flatten()
                x_normalized = (x_flat - K[0, 2]) / K[0, 0]
                y_normalized = (y_flat - K[1, 2]) / K[1, 1]
                X_camera = depth_flat * x_normalized
                Y_camera = depth_flat * y_normalized
                Z_camera = depth_flat
                points_camera = np.vstack((X_camera, Y_camera, Z_camera, np.ones_like(X_camera)))
                points_world = predict_c2w_ini @ points_camera
                X_world = points_world[0, :]
                Y_world = points_world[1, :]
                Z_world = points_world[2, :]
                points_3D = np.vstack((X_world, Y_world, Z_world))
                scene_coordinates_gs = points_3D.reshape(3, original_size[0], original_size[1])
                points_3D_at_pixels = np.zeros((matches_im0.shape[0], 3))
                for i, (x, y) in enumerate(matches_im0):
                    points_3D_at_pixels[i] = scene_coordinates_gs[:, y, x]

                if matches_im1.shape[0] >= 4:
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(points_3D_at_pixels.astype(np.float32), matches_im1.astype(np.float32), K, dist_eff,rvec=initial_rvec,tvec=initial_tvec, useExtrinsicGuess=True, reprojectionError=1.0,iterationsCount=2000,flags=cv2.SOLVEPNP_EPNP)
                    R = perform_rodrigues_transformation(rvec)
                    trans = -R.T @ np.matrix(tvec)
                    predict_c2w_refine = np.eye(4)
                    predict_c2w_refine[:3,:3] = R.T
                    predict_c2w_refine[:3,3] = trans.reshape(3)
                    ini_rot_error,ini_translation_error=cal_campose_error(predict_c2w_ini, gt_c2w_pose)
                    results_ini.append([ini_rot_error,ini_translation_error])
                    refine_rot_error,refine_translation_error=cal_campose_error(predict_c2w_refine, gt_c2w_pose)
                    results_final.append([refine_rot_error,refine_translation_error])
                    combined_list = [image] + rotmat2qvec(np.linalg.inv(predict_c2w_refine)[:3,:3]).tolist() + np.linalg.inv(predict_c2w_refine)[:3,3].tolist()
                    output_line = ' '.join(map(str, combined_list))
                    f.write(output_line + '\n')
                else:
                    ini_rot_error,ini_translation_error=cal_campose_error(predict_c2w_ini, gt_c2w_pose)
                    results_ini.append([ini_rot_error,ini_translation_error])
                    refine_rot_error,refine_translation_error=cal_campose_error(predict_c2w_ini, gt_c2w_pose)
                    results_final.append([refine_rot_error,refine_translation_error])
                    combined_list = [image] + rotmat2qvec(np.linalg.inv(predict_c2w_ini)[:3,:3]).tolist() + np.linalg.inv(predict_c2w_ini)[:3,3].tolist()
                    output_line = ' '.join(map(str, combined_list))
                    f.write(output_line + '\n')
                    
        median_result_ini = np.median(results_ini,axis=0)
        mean_result_ini = np.mean(results_ini,axis=0)

        median_result = np.median(results_final,axis=0)
        mean_result = np.mean(results_final,axis=0)
        pct10_5 = 0
        pct5 = 0
        pct2 = 0
        pct1 = 0
        for err in results_ini:
            r_err = err[0]
            t_err = err[1]
            if r_err < 5 and t_err < 0.1:  # 10cm/5deg
                pct10_5 += 1
            if r_err < 5 and t_err < 0.05:  # 5cm/5deg
                pct5 += 1
            if r_err < 2 and t_err < 0.02:  # 2cm/2deg
                pct2 += 1
            if r_err < 1 and t_err < 0.01:  # 1cm/1deg
                pct1 += 1
        total_frames = len(results_ini)
        pct10_5 = pct10_5 / total_frames * 100
        pct5 = pct5 / total_frames * 100
        pct2 = pct2 / total_frames * 100
        pct1 = pct1 / total_frames * 100
        _logger.info('Ini Accuracy:')
        _logger.info(f'\t10cm/5deg: {pct10_5:.1f}%')
        _logger.info(f'\t5cm/5deg: {pct5:.1f}%')
        _logger.info(f'\t2cm/2deg: {pct2:.1f}%')
        _logger.info(f'\t1cm/1deg: {pct1:.1f}%')

        pct10_5 = 0
        pct5 = 0
        pct2 = 0
        pct1 = 0
        for err in results_final:
            r_err = err[0]
            t_err = err[1]
            if r_err < 5 and t_err < 0.1:  # 10cm/5deg
                pct10_5 += 1
            if r_err < 5 and t_err < 0.05:  # 5cm/5deg
                pct5 += 1
            if r_err < 2 and t_err < 0.02:  # 2cm/2deg
                pct2 += 1
            if r_err < 1 and t_err < 0.01:  # 1cm/1deg
                pct1 += 1
        total_frames = len(results_final)
        pct10_5 = pct10_5 / total_frames * 100
        pct5 = pct5 / total_frames * 100
        pct2 = pct2 / total_frames * 100
        pct1 = pct1 / total_frames * 100
        _logger.info('After refine Accuracy:')
        _logger.info(f'\t10cm/5deg: {pct10_5:.1f}%')
        _logger.info(f'\t5cm/5deg: {pct5:.1f}%')
        _logger.info(f'\t2cm/2deg: {pct2:.1f}%')
        _logger.info(f'\t1cm/1deg: {pct1:.1f}%')

        # standard log
        _logger.info(f"--------------GS-CPR for {pe}:{SCENE}--------------")
        _logger.info("Initial Precision:")
        _logger.info('Median error {}m and {} degrees.'.format(median_result_ini[1], median_result_ini[0]))
        _logger.info('Mean error {}m and {} degrees.'.format(mean_result_ini[1], mean_result_ini[0]))

        _logger.info("After refine Precision:")
        _logger.info('Median error {}m and {} degrees.'.format(median_result[1], median_result[0]))
        _logger.info('Mean error {}m and {} degrees.'.format(mean_result[1], mean_result[0]))



