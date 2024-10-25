import sys
import os
sys.path.insert(1, './../datasets')
from spaces_dataset.code import utils
from spaces_dataset import get_scene_list

import metric_utils

from parallax_inpainting_new import ParallaxInpainting

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
        dataset_name='spaces',
        save_mesh_path=None
):
    depth_max_dim = pipeline_params['d_max_dim'] # 640
    block_size = pipeline_params['block_size'] # 16
    render_res = pipeline_params['render_res']  # 720
    reference_frame = pipeline_params['reference_frame'] # 0

    if dataset_name == 'spaces':
        data_infos_list = get_spaces_dataset_infos(data_path)
    for scene_infos in data_infos_list:
        
        cameras, scene_params, scene_path, scene_name = scene_infos
        
        for c_i in range(len(cameras[0])):
            
            if dataset_name == 'spaces':
                mvp_list = [o[c_i].camera.c_f_w for o in cameras[:]]
                intrinsics_list = [o[c_i].camera.intrinsics for o in cameras[:]]
                gt_path = [scene_path+c[c_i]['relative_path'] for c in scene_params]
                p_ratio_list = [c[c_i]['pixel_aspect_ratio'] for c in scene_params]
            
            parallax = ParallaxInpainting(
                        rgb_dir = 'images/color.png',
                        depth_max_dim=depth_max_dim,
                        block_size=block_size,
                        render_res=render_res)
            
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
            
            if save_mesh_path is not None:
                parallax.save_mesh('OURS/')


            parallax.renderMVP(
                mvp_list, 
                intrinsics_list, 
                p_ratio=p_ratio_list,
                frames_path=camera_frames_path)
            break
        break

def compute_metrics_dataset(frames_path, dataset_name='spaces'):
    
    if dataset_name == 'spaces':
        data_path = './../datasets/spaces_dataset/data/800/'
    os.makedirs(frames_path, exist_ok=True)
    
    pipeline_params = {}
    pipeline_params['d_max_dim'] = 640
    pipeline_params['block_size'] = 16
    pipeline_params['render_res']  = 720
    pipeline_params['reference_frame'] = 6

    frames_output_path = os.path.join(frames_path, 'rendered')
    os.makedirs(frames_output_path, exist_ok=True)
    gts_resized_path = os.path.join(frames_path, 'gts')
    os.makedirs(gts_resized_path, exist_ok=True)

    compute_dqmesh_results(data_path, frames_output_path, gts_resized_path, pipeline_params)
    metric_utils.compute_metrics(frames_path, 'DQMesh', dataset_name)

if __name__ == "__main__":
    compute_metrics_dataset('./output_frames/')