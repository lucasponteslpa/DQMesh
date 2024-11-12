import os
import torch
import lpips
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import sys
from tqdm import tqdm

def load_image(image_path, invert=False):
    image = cv2.imread(image_path) 
    if invert:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # crop de 20% das regioes de borda (jampani et al)
    h, w, _ = image.shape
    crop_h = int(h * 0.2)
    crop_w = int(w * 0.2)
    
    image = image[crop_h:h-crop_h, crop_w:w-crop_w]
    #image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0  # normalizar para [0, 1]
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # hwc -> chw -> bchw

    #breakpoint()
    return image

    
# calcular lpips
def calculate_lpips(image1, image2, lpips_model):
    #breakpoint()
    lpips_value = lpips_model(image1, image2)
    return lpips_value.item()

# calcular ssim
def calculate_ssim(image1, image2):
    image1 = image1.squeeze().permute(1, 2, 0).numpy()
    image2 = image2.squeeze().permute(1, 2, 0).numpy()
    #print(image1.shape)
    ssim_value, _ = ssim(
        image1, 
        image2, 
        channel_axis=2, 
        full=True, 
        win_size=9, 
        data_range=1.0,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False)
    return ssim_value

# calcular psnr
def calculate_psnr(image1, image2):
    image1 = image1.squeeze().permute(1, 2, 0).numpy()
    image2 = image2.squeeze().permute(1, 2, 0).numpy()
    #breakpoint()
    psnr_value = psnr(image1, image2)
    return psnr_value

# processar um unico par de diretorios
def process_directory(original_dir, rendered_dir, lpips_model):
    lpips_values = []
    ssim_values = []
    psnr_values = []
    
    # obter todos os nomes de arquivos de imagem em ambos os diretorios
    original_scenes = sorted(os.listdir(original_dir))
    rendered_scenes = sorted(os.listdir(rendered_dir))
    
    for original_img_scene, rendered_img_scene in tqdm(zip(original_scenes, rendered_scenes)):
        original_scene_path = os.path.join(original_dir, original_img_scene)
        rendered_scene_path = os.path.join(rendered_dir, rendered_img_scene)

        original_cameras = sorted(os.listdir(original_scene_path))
        rendered_cameras = sorted(os.listdir(rendered_scene_path))
        for original_img_camera, rendered_img_camera in zip(original_cameras, rendered_cameras):
            original_camera_path = os.path.join(original_scene_path, original_img_camera)
            rendered_camera_path = os.path.join(rendered_scene_path, rendered_img_camera)
            original_images = sorted(os.listdir(original_camera_path))
            rendered_images = sorted(os.listdir(rendered_camera_path))
            # breakpoint()
            # carregar imagens
            for original_img_name, rendered_img_name in zip(original_images, rendered_images):
                original_img_path = os.path.join(original_camera_path, original_img_name)
                rendered_img_path = os.path.join(rendered_camera_path, rendered_img_name)
                original_img = load_image(original_img_path)
                rendered_img = load_image(rendered_img_path)

                lpips_values.append(calculate_lpips(original_img, rendered_img, lpips_model))
                ssim_values.append(calculate_ssim(original_img, rendered_img))
                psnr_values.append(calculate_psnr(original_img, rendered_img))
                # breakpoint()
    return lpips_values, ssim_values, psnr_values

# funcao principal para processar todo o dataset e calcular metricas medias
def process_dataset(dataset_dir,frames_name='gts',rendered_name='rendered'):
    lpips_model = lpips.LPIPS(net='alex')
    all_lpips_values = []
    all_ssim_values = []
    all_psnr_values = []

   # breakpoint()
    
    # iterar por cada subdiretorio
    # for subdir in os.listdir(dataset_dir):
    #     breakpoint()
    #     subdir_path = os.path.join(dataset_dir, subdir)
        
    #     if not os.path.isdir(subdir_path):
    #         continue
        
    original_dir = os.path.join(dataset_dir, frames_name)
    rendered_dir = os.path.join(dataset_dir, rendered_name)
    
    if os.path.exists(original_dir) and os.path.exists(rendered_dir):
        lpips_values, ssim_values, psnr_values = process_directory(original_dir, rendered_dir, lpips_model)
        all_lpips_values.extend(lpips_values)
        all_ssim_values.extend(ssim_values)
        all_psnr_values.extend(psnr_values)
    
    # calcular metricas medias
    # mean_lpips = np.mean(all_lpips_values) if all_lpips_values else float('nan')
    # mean_ssim = np.mean(all_ssim_values) if all_ssim_values else float('nan')
    # mean_psnr = np.mean(all_psnr_values) if all_psnr_values else float('nan')
    # std_lpips = np.std(all_lpips_values) if all_lpips_values else float('nan')
    # std_ssim = np.std(all_ssim_values) if all_ssim_values else float('nan')
    # std_psnr = np.std(all_psnr_values) if all_psnr_values else float('nan')

    metric_values ={
        'psnr': all_psnr_values,
        'ssim': all_ssim_values,
        'lpips': all_lpips_values
    }
    
    return metric_values

# salvar resultados em um arquivo de texto
def save_results(mean_lpips, mean_ssim, mean_psnr, mesh_type, dataset_name):
    output_file = f'{mesh_type}-{dataset_name}_metrics.txt'
    
    with open(output_file, 'w') as f:
        f.write(f'mean lpips: {mean_lpips}\n')
        f.write(f'mean ssim: {mean_ssim}\n')
        f.write(f'mean psnr: {mean_psnr} db\n')

# ponto de entrada do script
def compute_metrics(dataset_dir, mesh_type, dataset_name):
    metric_values = process_dataset(dataset_dir)
    metrics_dict = {}
    metrics_dict['psnr'] = np.mean(metric_values['psnr'])
    metrics_dict['ssim'] = np.mean(metric_values['ssim'])
    metrics_dict['lpips'] = np.mean(metric_values['lpips'])
    metrics_dict['std_psnr'] = np.std(metric_values['psnr'])
    metrics_dict['std_ssim'] = np.std(metric_values['ssim'])
    metrics_dict['std_lpips'] = np.std(metric_values['lpips'])
    # save_results(mean_lpips, mean_ssim, mean_psnr, mesh_type, dataset_name)
    return metrics_dict, metric_values

def tiles(src_disp, src_rgb, K, tile_sz, pad_sz):

    bs, _, h, w = src_disp.shape
    K_, sx_, sy_, dmap_, rgb_ = [], [], [], [], []

    sy = torch.arange(0, h, tile_sz - pad_sz)
    sx = torch.arange(0, w, tile_sz - pad_sz)

    src_disp = F.pad(src_disp, (0, tile_sz, 0, tile_sz), 'replicate') 
    src_rgb = F.pad(src_rgb, (0, tile_sz, 0, tile_sz), 'replicate') 

    K_, src_disp_, src_rgb_,  sx_, sy_ = [], [], [], [], []
    for y in sy:
        for x in sx:
            l, r, t, b = x, x + tile_sz, y, y + tile_sz
            Ki = K.clone()
            Ki[:, 0, 2] = Ki[:, 0, 2] - x
            Ki[:, 1, 2] = Ki[:, 1, 2] - y

            K_.append(Ki)
            src_disp_.append( src_disp[:, :, t:b, l:r] )
            src_rgb_.append( src_rgb[:, :, t:b, l:r] )
            sx_.append(x)
            sy_.append(y)

    src_rgb_ = torch.stack(src_rgb_, 1)
    src_disp_ = torch.stack(src_disp_, 1)
    K_ = torch.stack(K_, 1)
    sx_, sy_ = torch.tensor(sx_).unsqueeze(0).expand(bs, -1), torch.tensor(sy_).unsqueeze(0).expand(bs, -1)
    return src_disp_, src_rgb_, K_, sx_, sy_
