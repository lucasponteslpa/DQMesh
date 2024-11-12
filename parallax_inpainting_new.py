from renderTexture import TextureRenderer
from renderSLIDEMesh import SLIDEMeshRenderer
import numpy as np
import cv2
import torch
import argparse
from mesh_utils import *

from depth_inference import midas_inference, dimas_inference
from simple_lama_inpainting import SimpleLama

from background_verts_module import background_mesh_verts, background_mesh_faces
from foreground_mesh_verts_module import pforeground_mesh_verts, pforeground_mesh_faces
from soft_disocclusions_module import soft_disocclusions_gen

import glob

# some_file.py
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, './datasets')
from spaces_dataset.code import utils
from spaces_dataset import get_scene_list

from csgl.mat4 import *

from scipy import ndimage
import math
# import trimesh
import open3d as o3d

from inpainting_Inference.inpainting_inference import lamaSep_inf

def depth_based_soft_foreground_pixel_visibility_map(depth: np.ndarray) -> np.ndarray:
    """
    Uses sobel gradients to generate soft foreground pixel visibility map.

    Args:
        depth: Depth map.

    Returns:
        pixel_visibility_map: Soft foreground pixel visibility map.
    """

    sobel_gradient_x = ndimage.sobel(depth, axis=0, mode="constant")
    sobel_gradient_y = ndimage.sobel(depth, axis=1, mode="constant")
    sobel_gradient = np.hypot(sobel_gradient_x, sobel_gradient_y)
    beta = 7
    pixel_visibility_map = np.exp(-beta * np.square(sobel_gradient))
    return pixel_visibility_map


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """
    Normalize array to range [0, 1].

    Args:
        arr: Numpy array

    Returns:
        arr: Numpy array
    """

    return (arr - arr.min()) / (arr.max() - arr.min())

def get_3D_image_points(image, depth: np.ndarray) -> np.ndarray:
    """
    Get (u, v, z) coordinates of all pixels in image.

    Args:
        image: RGB image
        depth: Corresponding depth map for image

    Returns:
        points: Numpy array of shape (image_height*image_width, 3)
    """

    image_height, image_width = np.array(image).shape[:2]

    points = np.zeros((image_height * image_width, 3))

    x = np.arange(0, image_width, 1)
    y = np.arange(0, image_height, 1)

    xv, yv = np.meshgrid(x, y)
    xv = np.expand_dims(xv, 2)
    yv = np.expand_dims(yv, 2)
    grid = np.concatenate((yv, xv), axis=2)
    grid = grid.reshape((image_height * image_width, 2))
    # breakpoint()
    points[:, 0] = grid[:, 0]
    points[:, 1] = grid[:, 1]
    points[:, 2] = depth[grid[:, 0], grid[:, 1]]

    return points

def reproject_3d_int_detail(verts, W, H, k_00, k_02, k_11, k_12, w_offset, h_offset):
    sx, sy, z = W*(verts[:,0]+1.0)/2.0, H*(verts[:,1]+1.0)/2.0, verts[:,2]
    z = -1. / np.maximum(z, 0.05)
    abs_z = abs(z)
    verts[:,0] = abs_z * ((sx+0.5-w_offset) * k_00 + k_02)
    verts[:,1] = abs_z * ((sy+0.5-h_offset) * k_11 + k_12)
    verts[:,2] = abs_z
    return verts

class ParallaxInpainting:
    def __init__(self,
                 rgb_dir,
                 depth_max_dim,
                 render_res=640,
                 block_size = 16,
                 depth_type='dimas',
                 inpaint_type='lamaSep',
                 pipeline='slide',
                 debug=False,
                 tex_scale=3
                 ):
        self.rgb_dir = rgb_dir
        self.depth_max_dim = depth_max_dim
        self.render_res = render_res
        self.block_size = block_size
        self.blocks_per_dim = depth_max_dim//block_size
        self.depth_type = depth_type
        self.inpaint_type = inpaint_type
        self.pipeline = pipeline
        self.tex_scale = tex_scale
        self.debug = debug

        self.soft_dis = None

    def get_shapes(self, color):
        
        width =  color.shape[1]
        height = color.shape[0]

        ratio_w = 1.0 if width >= height else width/height
        ratio_h = 1.0 if height >= width else height/width

        self.width =  np.round(self.blocks_per_dim*self.block_size*ratio_w).astype(int)
        self.height = np.round(self.blocks_per_dim*self.block_size*ratio_h).astype(int)
        self.width += ( - self.width%self.block_size)
        self.height += ( - self.height%self.block_size)

        self.ratio_w = 1.0 if self.width >=  self.height else self.width/self.height
        self.ratio_h = 1.0 if self.height >= self.width else  self.height/self.width

        dim = self.width if self.width > self.height else self.height
        self.dim = (dim, dim)
        if self.pipeline == 'DQMesh':
            self.tex_dim = (self.tex_scale*dim,self.tex_scale*dim)
        else:
            self.tex_dim = (dim,dim)
            


    def load_depth_and_canny(self, color=None, debug=True):
        
        if color is None:
            color = cv2.imread(self.rgb_dir)
            color = cv2.cvtColor(color,cv2.COLOR_BGR2RGB)
        
        self.get_shapes(color)
        self.color_res = cv2.resize(cv2.cvtColor(color.astype(np.uint8),cv2.COLOR_RGB2BGR), dsize=self.tex_dim, interpolation=cv2.INTER_CUBIC)
        
        cv2.imwrite(self.rgb_dir, self.color_res)

        #
        if self.depth_type == 'midas':
            img_depth = midas_inference(self.rgb_dir)
            img_depth = np.repeat(img_depth, 3, axis=-1)
        elif self.depth_type == 'dimas':
            img_depth = dimas_inference(self.rgb_dir)
            img_depth = np.repeat(img_depth, 3, axis=-1)
            # breakpoint()

        img_depth = img_depth/255.0

        depth_res_c = cv2.resize((img_depth*255).astype(np.uint8), dsize=self.dim, interpolation=cv2.INTER_CUBIC)
        kernel = np.ones((3, 3), 'uint8')
        # b_d  = cv2.erode(depth_res_c, kernel, iterations=2)
        b_d = cv2.GaussianBlur(depth_res_c, (3,3),0)
        # depth_res_c = b_d


        """=======================SOFT DISOCCLUSION=================="""
        self.soft_dis = np.zeros_like(depth_res_c)[:, :, 0].astype(np.uint8)
        d_in = depth_res_c[:, :, 0].astype(np.uint8)

        soft_disocclusions_gen(d_in, 30, 0.75, 1.0, self.soft_dis)
        self.soft_dis = 255*self.soft_dis
        """=========================================================="""


        img_depth_canny = cv2.Canny(b_d, 30, 70, L2gradient=True)
        pre_filter_canny = img_depth_canny
        img_depth_canny = self.filter_canny(img_depth_canny)



        # dilated_canny = cv2.dilate(img_depth_canny, kernel, iterations=1)

        # dilated_d = cv2.dilate(depth_res_c, kernel, iterations=2)
        # erode_d = cv2.erode(depth_res_c, kernel, iterations=2)
        # depth_res_c[dilated_canny==255] = erode_d[dilated_canny==255]
        # depth_res_c[img_depth_canny==255] = dilated_d[img_depth_canny==255]
        # dilated_d = cv2.dilate(depth_res_c, kernel, iterations=1)
        # # erode_d  = cv2.erode(dilated_d, kernel, iterations=1)
        # depth_res_c = dilated_d

        if debug:
            cv2.imwrite('depth_res_c.png', cv2.applyColorMap(depth_res_c, cv2.COLORMAP_INFERNO))
            cv2.imwrite('canny.png', img_depth_canny)
        img_depth = depth_res_c[::self.block_size,::self.block_size,:]/255
        self.img_depth_vert = img_depth
        self.img_depth = depth_res_c
        self.canny = img_depth_canny


    def filter_canny(self, canny):
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(canny, 15, cv2.CV_32S)
        filtered_canny = np.zeros_like(canny)
        n_pixels = np.prod(canny.shape)
        clip = 50
        for i in range(1,n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if (area >= clip):
                current = (labels == i).astype("uint8")
                filtered_canny = cv2.bitwise_or(filtered_canny, current)

        return 255*filtered_canny


    def halide_mesh(self, color=None):
        self.load_depth_and_canny(color=color, debug=True)
        
        canny_buf = self.canny
        assert canny_buf.ndim == 2
        assert canny_buf.dtype == np.uint8

        depth_buf = (self.img_depth[:,:,0]).astype(np.uint8)
        assert depth_buf.ndim == 2
        assert depth_buf.dtype == np.uint8

        num_blocks_W = canny_buf.shape[1] // self.block_size
        num_blocks_H = canny_buf.shape[0] // self.block_size

        total_blocks = num_blocks_H * num_blocks_W

        num_verts_W = num_blocks_W + 1
        num_verts_H = num_blocks_H + 1

        num_verts_BG = 2 * num_verts_H * num_verts_W

        num_faces_BG = 2 * num_blocks_W * num_blocks_H

        max_verts_block = (self.block_size + 1) * (self.block_size + 1)
        max_faces_block = 2 * max_verts_block

        """==========BACKGROUND MESH============="""
        IB_verts_shape    = (3, num_verts_W, num_verts_H)
        IB_uvs_shape      = (3, num_verts_W, num_verts_H)
        P_faces_shape     = (3, num_faces_BG)
        I_merge_mask      = np.empty(canny_buf.shape, dtype=canny_buf.dtype)
        IB_verts_buf      = np.empty(IB_verts_shape, dtype=np.float32)
        IB_uvs_buf        = np.empty(IB_uvs_shape, dtype=np.float32)
        back_faces_buf    = np.empty(P_faces_shape, dtype=np.int32)

        back_verts_shape  = (3, num_verts_BG)
        back_uvs_shape    = (3, num_verts_BG)
        back_verts_buf    = np.empty(back_verts_shape, dtype=np.float32)
        back_uvs_buf      = np.empty(back_uvs_shape, dtype=np.float32)

        background_mesh_verts(canny_buf, depth_buf, self.block_size, I_merge_mask, IB_verts_buf, IB_uvs_buf)
        background_mesh_faces(IB_verts_buf, IB_uvs_buf, back_verts_buf, back_uvs_buf, back_faces_buf)
        
        """=============FOREGROUND MESH==========="""
        out_f_shape     = (total_blocks, max_faces_block, 3)
        out_Nf_shape    = (total_blocks)
        out_limg_shape  = canny_buf.shape
        P_faces_buf     = np.zeros(out_f_shape, dtype=np.int32)
        P_Nfaces_buf    = np.zeros(out_Nf_shape, dtype=np.int32)
        I_Vlabels_buf   = np.zeros(out_limg_shape, dtype=np.int32)

        # if self.block_size == 4:
        #     from mmodules.foreground_mesh_gen_split4.foreground_mesh_verts_module import pforeground_mesh_verts, pforeground_mesh_faces
        

        pforeground_mesh_verts(canny_buf, depth_buf, self.block_size, P_faces_buf, P_Nfaces_buf, I_Vlabels_buf)

        NV = (I_Vlabels_buf>0).sum()
        NF = P_Nfaces_buf.sum()
        fore_verts_buf = np.zeros((NV,3), dtype=np.float32)
        fore_uvs_buf = np.zeros((NV,3), dtype=np.float32)
        fore_faces_buf = np.zeros((NF,3), dtype=np.int32)
        pforeground_mesh_faces(depth_buf, P_faces_buf, I_Vlabels_buf, back_verts_buf.shape[1], self.block_size, fore_verts_buf, fore_uvs_buf, fore_faces_buf)
        
        back_verts_buf = back_verts_buf.T
        back_verts_buf[:,-1] = back_verts_buf[:,-1] - 0.01
        back_uvs_buf = back_uvs_buf.T
        back_faces_buf = back_faces_buf.T
    
        self.all_verts = np.concatenate((back_verts_buf, fore_verts_buf), axis=0)
        self.all_uvs = np.concatenate((back_uvs_buf[:,:2], fore_uvs_buf[:,:2]), axis=0)
        self.all_uvs[:,0] /=self.all_uvs[:,0].max() 
        self.all_uvs[:,1] /=self.all_uvs[:,1].max()
        self.all_uvs[:,1] = 1.0 - self.all_uvs[:,1]
        self.fore_faces = fore_faces_buf[~(fore_faces_buf < 0).any(axis=1)]
        self.back_faces = back_faces_buf[~(back_faces_buf < 0).any(axis=1)]
        """=============TEXTURES=============="""
        mask_res = I_merge_mask

        kernel = np.ones((3, 3), 'uint8')
        dq_mask = mask_res

        inp_M = (self.soft_dis>0)*(dq_mask>0)
        inp_M = 255*inp_M.astype(np.uint8)
        back_mask = 255*(dq_mask>0).astype(np.uint8)
        inp_M = cv2.resize(np.repeat(np.expand_dims(inp_M, -1), 3, axis=-1), dsize=self.tex_dim, interpolation = cv2.INTER_NEAREST)
        back_mask = cv2.resize(np.repeat(np.expand_dims(back_mask, -1), 3, axis=-1), dsize=self.tex_dim, interpolation = cv2.INTER_NEAREST)
       
        inp_M = cv2.dilate(inp_M, kernel, iterations=4)

        # DO INPAITING W/ inp_M
        if self.inpaint_type == 'lama' :
            inpaint_func = SimpleLama(device="cuda")
            img_inp = inpaint_func(self.color_res, inp_M[:,:,0])
        else:
            img_inp = lamaSep_inf(self.color_res, inp_M[:,:,:1])

        img_inp = np.array(img_inp)
        cv2.imwrite("big_mask.png", inp_M)
    
        back_tex = ((inp_M/255)*img_inp + (1 - (inp_M/255))*self.color_res).astype(np.uint8)
        # back_tex = ((back_mask/255)*back_tex).astype(np.uint8)
        self.back_tex_path = 'images/back_tex.png'

        alpha = 255*np.ones_like(back_tex)[:, :, :1]
        back_tex = np.concatenate((cv2.cvtColor(back_tex,cv2.COLOR_RGB2BGR), alpha), axis=-1)
        # back_tex = np.concatenate((back_tex, alpha), axis=-1)
        cv2.imwrite(self.back_tex_path, back_tex)

        depth_buf = cv2.dilate(depth_buf,kernel, iterations=1)
        alpha = depth_based_soft_foreground_pixel_visibility_map(depth_buf/depth_buf.max())
        alpha = cv2.erode(alpha,kernel,iterations=1)
        alpha = (alpha>0.1).astype('int32')
        alpha = (255*((alpha-alpha.min())/(alpha.max()-alpha.min()))).astype(np.uint8)
        
        alpha = cv2.resize(alpha, dsize=self.tex_dim, interpolation=cv2.INTER_CUBIC)
        alpha = np.expand_dims(alpha, axis=-1)
        fore_tex = np.concatenate((self.color_res, alpha), axis=-1)
        self.fore_tex_path = 'images/fore_tex.png'
        fore_tex = np.concatenate((cv2.cvtColor(self.color_res,cv2.COLOR_RGB2BGR), alpha), axis=-1)
        # fore_tex = np.concatenate((self.color_res, alpha), axis=-1)
        cv2.imwrite(self.fore_tex_path, fore_tex)

    def direct_mesh(self, color=None):
        self.load_depth_and_canny(color=color, debug=True)

        """=============TEXTURES=============="""
        self.img_depth = cv2.resize(self.img_depth, dsize=self.tex_dim)
        kernel = np.ones((3, 3), 'uint8')

        inp_M = (self.soft_dis>0)
        inp_M = 255*inp_M.astype(np.uint8)
        inp_M = cv2.resize(np.repeat(np.expand_dims(inp_M, -1), 3, axis=-1), dsize=self.tex_dim, interpolation = cv2.INTER_NEAREST)
       
        inp_M = cv2.dilate(inp_M, kernel, iterations=4)
        cv2.imwrite("big_mask.png", inp_M)

        # DO INPAITING W/ inp_M
        if self.inpaint_type == 'lama' :
            inpaint_func = SimpleLama(device="cuda")
            img_inp = np.array(inpaint_func(self.color_res, inp_M[:,:,0]))
            depth_inp = np.array(inpaint_func(self.img_depth, inp_M[:,:,0]))
        else:
            img_inp = lamaSep_inf(self.color_res, inp_M[:,:,:1])
            depth_inp = lamaSep_inf(self.img_depth, inp_M[:,:,0])
    

        back_tex = ((inp_M/255)*img_inp + (1 - (inp_M/255))*self.color_res).astype(np.uint8)
        back_alpha = 255*np.ones_like(back_tex)[:, :, :1]
        # back_tex = np.concatenate((cv2.cvtColor(back_tex,cv2.COLOR_RGB2BGR), back_alpha), axis=-1)
        back_tex = np.concatenate((back_tex, back_alpha), axis=-1)
        self.back_tex_path = 'images/back_tex.png'
        cv2.imwrite(self.back_tex_path, back_tex)

        alpha = depth_based_soft_foreground_pixel_visibility_map(self.img_depth[:,:,0]/self.img_depth.max())
        # alpha = cv2.erode(alpha,kernel,iterations=2)
        alpha = (255*((alpha-alpha.min())/(alpha.max()-alpha.min()))).astype(np.uint8)
   
        alpha = cv2.resize(alpha, dsize=self.tex_dim, interpolation=cv2.INTER_CUBIC)
        alpha = np.expand_dims(alpha, axis=-1)
        fore_tex = np.concatenate((self.color_res, alpha), axis=-1)
        self.fore_tex_path = 'images/fore_tex.png'
        # fore_tex = np.concatenate((cv2.cvtColor(self.color_res,cv2.COLOR_RGB2BGR), alpha), axis=-1)
        fore_tex = np.concatenate((self.color_res, alpha), axis=-1)
        cv2.imwrite(self.fore_tex_path, fore_tex)


        
        """===============================Background Mesh================================"""
        points = get_3D_image_points(back_tex, depth_inp[:,:,0]/self.img_depth.max())
        back_points_uv = points[:, :2].astype(np.int32).copy()
        back_colors = np.asarray(back_tex)[back_points_uv[:, 0], back_points_uv[:, 1]]
        # alpha = back_alpha[back_points_uv[:, 0], back_points_uv[:, 1]]/255

        # Normalize points and colors.
        back_points = np.zeros_like(points)
        back_points[:, 0] = 2*normalize_array(points[:, 1]) - 1
        back_points[:, 1] = 2*normalize_array(points[:, 0]) - 1
        back_points[:, 2] = normalize_array(points[:, 2]) - 0.1
        back_points = back_points.astype('float32')
        back_colors = normalize_array(back_colors).astype('float32')
        back_faces = []
        v_idx = 0
        for i in range(self.img_depth.shape[0]-1):
            for j in range(self.img_depth.shape[1]-1):
                v0 = i*self.img_depth.shape[1] + j + v_idx
                v1 = i*self.img_depth.shape[1] + (j+1) + v_idx
                v2 = (i+1)*self.img_depth.shape[1] + j+ v_idx
                v3 = (i+1)*self.img_depth.shape[1] + (j+1) + v_idx
                back_faces.append([v0,v1,v3])
                back_faces.append([v0,v2,v3])

        """===============================Foreground Mesh================================"""
        points = get_3D_image_points(self.color_res, self.img_depth[:,:,0]/self.img_depth.max())
        fore_points_uv = points[:, :2].astype(np.int32).copy()
        fore_colors = np.asarray(fore_tex)[fore_points_uv[:, 0], fore_points_uv[:, 1]]
        # alpha = alpha[fore_points_uv[:, 0], fore_points_uv[:, 1]]/255

        # Normalize points and colors.
        fore_points = np.zeros_like(points)
        fore_points[:, 0] = 2*normalize_array(points[:, 1])-1
        fore_points[:, 1] = 2*normalize_array(points[:, 0])-1
        fore_points[:, 2] = normalize_array(points[:, 2])
        fore_points = fore_points.astype('float32')
        fore_colors = normalize_array(fore_colors).astype('float32')
        fore_faces = []
        v_idx = back_points.shape[0]
        for i in range(self.img_depth.shape[0]-1):
            for j in range(self.img_depth.shape[1]-1):
                v0 = i*self.img_depth.shape[1] + j + v_idx
                v1 = i*self.img_depth.shape[1] + (j+1) + v_idx
                v2 = (i+1)*self.img_depth.shape[1] + j+ v_idx
                v3 = (i+1)*self.img_depth.shape[1] + (j+1) + v_idx
                fore_faces.append([v0,v1,v3])
                fore_faces.append([v0,v2,v3])

        self.all_verts = np.concatenate((back_points, fore_points), axis=0)
        self.all_colors = np.concatenate((back_colors, fore_colors), axis=0)
        self.fore_faces = np.array(fore_faces, dtype='int32')
        self.back_faces = np.array(back_faces, dtype='int32')
        self.all_uvs = None

        # total_faces = np.concatenate((self.back_faces, self.fore_faces), axis=0)
        # mesh = o3d.geometry.TriangleMesh()
        # mesh.vertices = o3d.utility.Vector3dVector(torch.from_numpy(self.all_verts))
        # mesh.vertex_colors = o3d.utility.Vector3dVector(torch.from_numpy(self.all_colors[:,:-1]))
        # mesh.triangles = o3d.utility.Vector3iVector(torch.from_numpy(self.fore_faces))
        # o3d.io.write_triangle_mesh("slide_mesh.obj", mesh)
    
    def save_mesh(self, out_dir):
        total_faces = np.concatenate((self.back_faces, self.fore_faces), axis=0)
        
        os.makedirs(out_dir, exist_ok=True)

        # Create a Trimesh object
        # mesh = trimesh.Trimesh(vertices=self.all_verts, uvs=self.all_uvs, faces=total_faces)

        # Define a texture image file (you can load a specific texture image here)
        texture_image_path = 'texture.jpg'  # Change to your texture path

        # Save as a .obj file with texture
        write_obj('',
            v_pos=torch.from_numpy(self.all_verts),
            v_tex=torch.from_numpy(self.all_uvs),
            t_pos_idx=torch.from_numpy(self.back_faces),
            t_tex_idx=torch.from_numpy(self.back_faces),
            file_name=f'{out_dir}/back_mesh.obj')
        write_obj('',
            v_pos=torch.from_numpy(self.all_verts),
            v_tex=torch.from_numpy(self.all_uvs),
            t_pos_idx=torch.from_numpy(self.fore_faces),
            t_tex_idx=torch.from_numpy(self.fore_faces),
            file_name=f'{out_dir}/fore_mesh.obj')
        back_mesh = o3d.io.read_triangle_mesh(f'{out_dir}/back_mesh.obj')
        fore_mesh = o3d.io.read_triangle_mesh(f'{out_dir}/fore_mesh.obj')
        # if not mesh.has_vertex_normals():
        #     mesh.compute_vertex_normals()
        #Load the texture image
        fore_texture_image = o3d.io.read_image(self.fore_tex_path)
        back_texture_image = o3d.io.read_image(self.back_tex_path)

        #Assign the texture image to the mesh
        back_mesh.textures = [back_texture_image]
        fore_mesh.textures = [fore_texture_image]

        #Save the textured mesh to an output file (ex: .obj format)
        o3d.io.write_triangle_mesh("back_mesh.obj", back_mesh)
        o3d.io.write_triangle_mesh("fore_mesh.obj", fore_mesh)

        # mesh.export('textured_mesh.obj', include_texture=True, texture=self.fore_tex_path)


        
    def reproject_mesh(self, intrinsic, w_offset=0, h_offset=0):
        W_2 = 2*intrinsic[0,2]
        H_2 = 2*intrinsic[1,2]
        inv_intrinsic = intrinsic.getI()
        self.all_verts = reproject_3d_int_detail(
            self.all_verts,
            W_2,
            H_2,
            inv_intrinsic[0,0],
            inv_intrinsic[0,2],
            inv_intrinsic[1,1],
            inv_intrinsic[1,2],
            w_offset=w_offset,
            h_offset=h_offset
        )
        self.all_verts[:,-1] *= -1
        if self.pipeline == 'DQMesh':
            self.all_verts[:,0] *= -1
        else:
            self.all_verts[:,1] *= -1

    def list_generate_mesh(self, gt_list, ref_frame_index: int = 0, save_gts_path=None):
        
        img = np.array(cv2.imread(gt_list[ref_frame_index]))
        if self.pipeline == 'DQMesh':
            self.halide_mesh(color=img)
        else:
            self.direct_mesh(color=img)

        self.save_gts_path = save_gts_path
        self.gt_list = gt_list

    
    def renderMVP(self, poses, intrinsics, p_ratio=None, frames_path=None):
        if frames_path is not None:
            get_screenshot = True
        else:
            get_screenshot = False
        W_2 = int(2*intrinsics[0][0,2])
        H_2 = int(2*intrinsics[0][1,2])
        if self.save_gts_path is not None:
            for i,gt_path in enumerate(self.gt_list):
                color = cv2.imread(gt_path)
                self.get_shapes(color)
                gt_img = cv2.resize(color, dsize=(W_2,H_2), interpolation=cv2.INTER_CUBIC)
                frame_name = str(i)
                gt_rs_path = os.path.join(self.save_gts_path, (4-len(frame_name))*"0"+frame_name+'.jpg')
                cv2.imwrite(gt_rs_path, gt_img)
        if self.pipeline=='DQMesh':
            render = TextureRenderer(
                                height=int(H_2),
                                width= int(W_2),
                                vertices=self.all_verts.reshape(-1),
                                uv_coords=self.all_uvs.reshape(-1),
                                indices=self.fore_faces.reshape(-1),
                                indices_back=self.back_faces.reshape(-1),
                                img_file=self.fore_tex_path,
                                img_file_back=self.back_tex_path,
                                texture_dims=self.tex_dim,
                                frames_path=frames_path)
            render.runMVP(
                poses,
                intrinsics,
                p_ratio=1.0 if p_ratio is None else p_ratio, 
                get_screenshot=get_screenshot
            )
        else:
            total_faces = np.concatenate((self.back_faces, self.fore_faces), axis=0)
            render = SLIDEMeshRenderer(
                             height=int(H_2),
                             width=int(W_2),
                             vertices=self.all_verts.reshape(-1),
                             colors=self.all_colors.reshape(-1),
                             indices=total_faces.reshape(-1),
                             frames_path=frames_path)
            render.runMVP(
                poses,
                intrinsics,
                p_ratio=1.0 if p_ratio is None else p_ratio, 
                get_screenshot=get_screenshot
            )


def render_path(num_frames=900, r_x=0.3, r_y=0.3, r_z=0.3):
    t = torch.arange(num_frames) / (num_frames - 1)
    poses = torch.eye(4).repeat(num_frames, 1, 1)
    poses[:, 0, 3] = r_x * torch.sin(2. * math.pi * t)
    poses[:, 1, 3] = r_y * torch.cos(2. * math.pi * t)
    poses[:, 2, 3] = r_z * (torch.cos(2. * math.pi * t) - 1.)
    return poses

if __name__ == "__main__":
    rgb_dir = 'images/color.png'
    inpaint_dir = 'images/inpaint_curr.png'
    depth_dir = None

    script_dir = os.path.dirname(os.path.abspath(__file__))
    rgb_dir = os.path.join(script_dir, 'images', 'moon.jpg')
    inpaint_dir = os.path.join(script_dir, 'images', 'inpaint_curr.png')

    # dataset_dir = sys.argv[1]
    # output_dir = sys.argv[2]
    try:
        reference_frame_index = int(sys.argv[3])
    except:
        reference_frame_index = 0
    data_path = sys.path[0]+'/datasets/spaces_dataset/data/800/'
    scenes_list  = get_scene_list( data_path)
    scene_path = data_path+scenes_list[4]+'/'
    a, model_json = utils.ReadScene( scene_path)

    mvp_ref = a[reference_frame_index][0].camera.w_f_c
    mvp_list = [o[0].camera.c_f_w for o in a[:]]
    for i,mvp in enumerate(mvp_list):
        mvp_list[i] = mvp
    intrinsics_list = [o[0].camera.intrinsics for o in a[:]]
    gt_path = [scene_path+c[0]['relative_path'] for c in model_json]
    p_ratio_list = [c[0]['pixel_aspect_ratio'] for c in model_json]

    parallax = ParallaxInpainting(
                # rgb_dir,
                gt_path[0],
                depth_max_dim=640,
                block_size=16,
                render_res=720)
    parallax.halide_mesh()
    # parallax.reproject_mesh(intrinsics_list[0])
    # parallax.save_mesh('./')
    # breakpoint() 
    poses = render_path()
    # parallax.list_generate_mesh(gt_path, reference_frame_index)
    parallax.reproject_mesh(intrinsics_list[0])
    parallax.save_mesh('./')
    parallax.renderMVP(poses, intrinsics_list, p_ratio=p_ratio_list)
    




