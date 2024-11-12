import sys
import os
sys.path.insert(1, './datasets')
sys.path.insert(1, './TMPI')
from spaces_dataset.code import utils
from DP import dp_utils
from spaces_dataset import get_scene_list
import torch
from tqdm import tqdm

import pandas as pd

import metric_utils

from parallax_inpainting_new import ParallaxInpainting
from depth_inference import midas_inference, dimas_inference
# from TMPI import utils, tmpi
# from TMPI.tmpi_renderer_gl import TMPIRendererGL

def get_dualpixels_dataset_infos(data_path):
    scenes_images_path = os.path.join(data_path,'scaled_images')
    scenes_list  = get_scene_list( scenes_images_path)
    data_infos_list = []
    for scene in scenes_list:
        params_path = os.path.join(data_path,'scaled_camera_pose_txt')
        scene_path = os.path.join(params_path,scene)
        views, model_json = dp_utils.ReadScene( scene_path, scene)
        data_infos_list.append((views, model_json,scene_path,scene))
    return data_infos_list

def get_spaces_dataset_infos(data_path):
    scenes_list  = get_scene_list( data_path)
    data_infos_list = []
    for scene in scenes_list:
        scene_path = data_path+scene+'/'
        a, model_json = utils.ReadScene( scene_path)
        data_infos_list.append((a,model_json,scene_path,scene))
    return data_infos_list

def compute_dqmesh_results(
        data_path,
        frames_output_path,
        gts_resized_path,
        pipeline_params,
        dataset_name='dualpixels',
        save_mesh_path=None
):
    depth_max_dim = pipeline_params['d_max_dim'] # 640
    block_size = pipeline_params['block_size'] # 16
    reference_frame = pipeline_params['reference_frame'] # 0
    inpaint = pipeline_params['inpaint']
    depth = pipeline_params['depth']
    tex_scale = pipeline_params['tex_scale']
    pipe_name = pipeline_params['pipe_name']

    if dataset_name == 'spaces':
        data_infos_list = get_spaces_dataset_infos(data_path)
    else:
        data_infos_list = get_dualpixels_dataset_infos(data_path)
    for scene_infos in tqdm(data_infos_list[:]):
        
        cameras, scene_params, scene_path, scene_name = scene_infos
        for c_i in range(len(cameras[0][:1] if dataset_name == 'spaces' else [1])):
            
            if dataset_name == 'spaces':
                mvp_list = [o.camera.c_f_w for o in cameras[c_i][:8]]
                # intrinsics_list = [o[c_i].camera.intrinsics for o in cameras[:]]
                intrinsics_list = [o.camera.intrinsics for o in cameras[c_i][:8]]
                # gt_path = [scene_path+c[c_i]['relative_path'] for c in scene_params]
                gt_path = [scene_path+c['relative_path'] for c in scene_params[c_i][:8]]
                p_ratio_list = [c['pixel_aspect_ratio'] for c in scene_params[c_i][:8]]
            else:
                mvp_list = [o.camera.w_f_c for o in cameras]
                intrinsics_list = [o.camera.intrinsics for o in cameras]
                gt_path = [c['relative_path'] for c in scene_params]
                p_ratio_list = [c['pixel_aspect_ratio'] for c in scene_params]

            # breakpoint()
            if dataset_name == 'dualpixels':
                for i, gt_name in enumerate(gt_path):
                    if 'center' in gt_name:
                        reference_frame = i
            parallax = ParallaxInpainting(
                        rgb_dir = 'images/color.png',
                        depth_max_dim=depth_max_dim,
                        block_size=block_size,
                        depth_type = depth,
                        inpaint_type = inpaint,
                        pipeline=pipe_name,
                        tex_scale=tex_scale)
            
            scene_frames_path = os.path.join(frames_output_path,scene_name)
            os.makedirs(scene_frames_path, exist_ok=True)
            camera_frames_path = os.path.join(scene_frames_path,str(c_i)) 
            os.makedirs(camera_frames_path, exist_ok=True)
            scene_gts_path = os.path.join(gts_resized_path, scene_name)
            os.makedirs(scene_gts_path, exist_ok=True)
            camera_gts_path = os.path.join(scene_gts_path,str(c_i)) 
            os.makedirs(camera_gts_path, exist_ok=True)
            
            parallax.list_generate_mesh(gt_path, reference_frame, save_gts_path=camera_gts_path)
            parallax.reproject_mesh(intrinsics_list[0])
            
            # if save_mesh_path is not None:
            # parallax.save_mesh('./')


            parallax.renderMVP(
                mvp_list, 
                intrinsics_list, 
                p_ratio=p_ratio_list,
                frames_path=camera_frames_path)
            # breakpoint()
            # break
        # break

            


def compute_metrics_dataset(frames_path, dataset_name='dualpixels', pipeline='DQMesh'):
    
    if dataset_name == 'spaces':
        data_path = './datasets/spaces_dataset/data/800/'
    else:
        data_path = './datasets/DP/test/'
    os.makedirs(frames_path, exist_ok=True)
    
    # d_max_dim = [640,640,640,1280,1280,1280]
    # block_size = [8,16,32,8,16,32]
    # d_max_dim = [640,1280]
    # block_size = [8,16]
    d_max_dim = [1280,1280,1280] if pipeline=='DQMesh' else [1280]
    block_size = [4,8,16] if pipeline=='DQMesh' else [4]
    tex_scale = [3,3,3]
    reference_frame = 0
    inpaint = ['lamaSep' for _ in range(6) ]
    depth = ['dimas' for _ in range(6)]
    final_csv = {
        'd_max_dim':[],
        'block_size':[],
        'inpaint':[],
        'depth':[],
        'psnr':[],
        'ssim':[],
        'lpips':[],
        'std_psnr':[],
        'std_ssim':[],
        'std_lpips':[],
    }

    all_metric_values = []
    for i in range(len(block_size)):
        pipeline_params = {}
        pipeline_params['d_max_dim'] = d_max_dim[i]
        pipeline_params['block_size'] = block_size[i]
        pipeline_params['reference_frame'] = 0
        pipeline_params['inpaint'] = inpaint[i]
        pipeline_params['depth'] = depth[i]
        pipeline_params['tex_scale'] = tex_scale[i]
        pipeline_params['pipe_name'] = pipeline
        

        frames_output_path = os.path.join(frames_path, 'rendered')
        os.makedirs(frames_output_path, exist_ok=True)
        gts_resized_path = os.path.join(frames_path, 'gts')
        os.makedirs(gts_resized_path, exist_ok=True)

        compute_dqmesh_results(data_path, frames_output_path, gts_resized_path, pipeline_params, dataset_name=dataset_name)
        metrics_dict, metric_values = metric_utils.compute_metrics(frames_path, pipeline, dataset_name)

        final_csv['d_max_dim'] += [pipeline_params['d_max_dim']]
        final_csv['block_size'] += [pipeline_params['block_size']]
        final_csv['inpaint'] += [pipeline_params['inpaint']]
        final_csv['depth'] += [pipeline_params['depth']]
        final_csv['psnr'] += [metrics_dict['psnr']]
        final_csv['ssim'] += [metrics_dict['ssim']]
        final_csv['lpips'] += [metrics_dict['lpips']]
        final_csv['std_psnr'] += [metrics_dict['std_psnr']]
        final_csv['std_ssim'] += [metrics_dict['std_ssim']]
        final_csv['std_lpips'] += [metrics_dict['std_lpips']]
        all_metric_values += [metric_values]
        print(metrics_dict)
    df = pd.DataFrame(final_csv)
    file_name = f'{pipeline}-{dataset_name}_metrics.csv'
    df.to_csv(file_name, sep='\t')
    return all_metric_values

import json
import numpy as np
class NumpyTypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    complete_test_dict = {}
    complete_test_dict['DQMesh_dualpixels'] = compute_metrics_dataset('./output_frames_dqmesh_dual/')
    complete_test_dict['DQMesh_spaces'] = compute_metrics_dataset('./output_frames_dqmesh_spaces/', dataset_name='spaces')
    complete_test_dict['SLIDE_dualpixels'] = compute_metrics_dataset('./output_frames_slide_dual/', pipeline='SLIDE')
    complete_test_dict['SLIDE_spaces'] = compute_metrics_dataset('./output_frames_slide_spaces/', dataset_name='spaces', pipeline='SLIDE')
    file_name = "all_tested_metrics.json"
    # Write the dictionary to a JSON file
    with open(file_name, 'w') as json_file:
        a = json.dumps(complete_test_dict, indent=1, cls=NumpyTypeEncoder)
        json.dump(a, json_file)
# def compute_tmpi_results(
#         data_path,
#         frames_output_path,
#         gts_resized_path,
#         pipeline_params,
#         dataset_name='spaces',
#         save_mesh_path=None
# ):
#     depth_max_dim = pipeline_params['d_max_dim'] # 640
#     block_size = pipeline_params['block_size'] # 16
#     render_res = pipeline_params['render_res']  # 720
#     reference_frame = pipeline_params['reference_frame'] # 0
    
#     model = tmpi.TMPI(num_planes=4)
#     model_file = 'TMPI/weights/mpti_04.pth'                                
#     model.load_state_dict( torch.load(  model_file, weights_only=False) ) 
    
#     if dataset_name == 'spaces':
#         data_infos_list = get_spaces_dataset_infos(data_path)
#     for scene_infos in tqdm(data_infos_list[:10]):
        
#         cameras, scene_params, scene_path, scene_name = scene_infos
#         # breakpoint()
#         for c_i in range(len(cameras[0][:1])):
            
#             if dataset_name == 'spaces':
#                 mvp_list = [o.camera.c_f_w for o in cameras[c_i][:8]]
#                 # intrinsics_list = [o[c_i].camera.intrinsics for o in cameras[:]]
#                 intrinsics_list = [o.camera.intrinsics for o in cameras[c_i][:8]]
#                 # gt_path = [scene_path+c[c_i]['relative_path'] for c in scene_params]
#                 gt_path = [scene_path+c['relative_path'] for c in scene_params[c_i][:8]]
#                 p_ratio_list = [c['pixel_aspect_ratio'] for c in scene_params[c_i][:8]]
            
#             if h >= w and h >= config.imgsz_max:
#                 h_scaled, w_scaled = 1024, int(1024 / h * w)
#             elif w > h and w >= config.imgsz_max:
#                 h_scaled, w_scaled = int(1024 / w * h), 1024
#             else:
#                 h_scaled, w_scaled = h, w
            
            
#             src_rgb = gt_path[reference_frame]
#             src_disp = dimas_inference(gt_path[reference_frame])
#             K = intrinsics_list[0]
#             tile_sz = int(np.clip(utils.next_power_of_two(0.125 * w_scaled - 1), a_min=64, a_max=256))
#             pad_sz = int(tile_sz * 0.125)
#             src_disp_tiles, src_rgb_tiles, K_tiles, sx, sy = metric_utils.tiles(src_disp, src_rgb, K, tile_sz, pad_sz)

#             mpis, mpi_disp = model( src_rgb_tiles, src_disp_tiles, src_rgb, src_disp)
            
#             h, w = src_rgb.shape[-2:]
#             renderer = TMPIRendererGL(h, w)
#             tgt_rgb_syn = renderer( mpis.cpu(), mpi_disp.cpu(), poses, cam_int, sx, sy)
#             breakpoint()
#             scene_frames_path = os.path.join(frames_output_path,scene_name)
#             os.makedirs(scene_frames_path, exist_ok=True)
#             camera_frames_path = os.path.join(scene_frames_path,str(c_i)) 
#             os.makedirs(camera_frames_path, exist_ok=True)
#             scene_gts_path = os.path.join(gts_resized_path, scene_name)
#             os.makedirs(scene_gts_path, exist_ok=True)
#             camera_gts_path = os.path.join(scene_gts_path,str(c_i)) 
#             os.makedirs(camera_gts_path, exist_ok=True)