#! /usr/bin/env python
""" Various Utilities
"""
from __future__ import print_function

from OpenGL.GL import *
from csgl import *

import glfw
import math as mathf

from PIL import Image
from PIL.Image import open as pil_open
import imageio
import glob
import collections

import cv2
import numpy as np
import os

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)

def read_cameras_text(path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(
                    id=camera_id,
                    model=model,
                    width=width,
                    height=height,
                    params=params,
                )
    return cameras

def screenshot(width,height, file_name_out=None):
    data = glReadPixels(0,0,width,height,GL_RGB,GL_UNSIGNED_BYTE,outputType=None)
    image = Image.frombytes(mode="RGB", size=(width, height), data=data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save(file_name_out)
    return image

def gen_video(frames_in_path, video_out_path, mp4save_fn):
    writer = imageio.get_writer(f'{video_out_path}/{mp4save_fn}', mode='I', fps=30, codec='libx264', bitrate='16M')
    frame_names = sorted(glob.glob(frames_in_path+'*.png'))
    for fn in frame_names:
        f = imageio.imread(fn)
        writer.append_data(f)
    writer.close()

def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)

def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg



def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    # convert to homogeneous coordinate for faster computation
    pose_avg_homo[:3] = pose_avg
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]),
                       (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row],
                       1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(
        pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, np.linalg.inv(pose_avg_homo)


def get_poses(root_dir):
    poses_bounds = np.load(os.path.join(root_dir,'poses_bounds.npy'))
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    bounds = poses_bounds[:, -2:]  # (N_images, 2)
    H, W, focal = poses[0, :, -1]
    ratio = H/W
    # focal *= ratio
    poses = np.concatenate(
            [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    poses, pose_avg = center_poses(poses)

    near_original = bounds.min()
    scale_factor = near_original*0.75

    near = near_original * 0.9 / scale_factor
    far = bounds.max() * 1

    bounds /= scale_factor
    poses[..., 3] /= scale_factor
    return poses, bounds, focal, near, far, ratio, scale_factor

def unnormalize_intrinsics(intrinsics, h, w):
    intrinsics[0] *= w
    intrinsics[1] *= h
    return intrinsics

def camera_line( entry, c2w=False):
    fx, fy, cx, cy = entry[1:5]
    intrinsics = np.array([[fx, 0, cx, 0],
                                [0, fy, cy, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
    w2c_mat = np.array(entry[7:]).reshape(3, 4)
    w2c_mat_4x4 = np.eye(4)
    w2c_mat_4x4[:3, :] = w2c_mat
    w2c_mat = w2c_mat_4x4
    if c2w:
        c2w_mat = np.linalg.inv(w2c_mat_4x4)
        return c2w_mat, intrinsics
    else:
        return w2c_mat, intrinsics
    

def loadRealStateCameraPoses(filepath, dims=None, inverse_cam=False, split=None):

    text_data = open(filepath, "r") 
    lines = text_data.readlines()

    scale = 0.1
    origin = [0,0,0]
    cam_matrix_list = []
    intrisics_list = []
    for i, line in enumerate(lines):
        if i == 0:
            # 0th line is Youtube-URL
            continue

        # values = [float(v) for j, v in enumerate(line.split(' ')) if j > 6]
        entry = [float(x) for x in line.split()]

        # R = np.array([[values[0], values[1], values[2]],
        #               [values[4], values[5], values[6]],
        #               [values[8], values[9], values[10]]])
        # t = np.array([values[3],values[7],values[11]])
        w2c, intrisics = camera_line(entry)

        if dims is not None:
            intrisics = unnormalize_intrinsics(intrisics, dims[0], dims[1])
        
        if split is None:
            if i == 1:
                transform_matrix0 = w2c
        else:
            if (i%split) == 0:
                transform_matrix0 = w2c
        
        if inverse_cam:
            c2w = np.linalg.inv(w2c)
            t_inv = np.linalg.inv(transform_matrix0)
            cam_matrix_list.append(c2w@t_inv)
            # R_inv = np.linalg.inv(R)
            # T = -R_inv @ t
            # transform_matrix = np.array([[R_inv[0,0], R_inv[0,1], R_inv[0,2], T[0]],
            #                              [R_inv[1,0], R_inv[1,1], R_inv[1,2], T[1]],
            #                              [R_inv[2,0], R_inv[2,1], R_inv[2,2], T[2]],
            #                              [0.0,       0.0,       0.0,       1.0]])
        else:
            cam_matrix_list.append(w2c@transform_matrix0)
            # transform_matrix = np.array([[R[0,0], R[0,1], R[0,2], t[0]],
            #                                 [R[1,0], R[1,1], R[1,2], t[1]],
            #                                 [R[2,0], R[2,1], R[2,2], t[2]],
            #                                 [0.0,    0.0,    0.0,    1.0]])
        intrisics_list.append(intrisics)
        
    cam_matrix = np.stack(cam_matrix_list, axis=0)
    intrisics_matrix = np.stack(intrisics_list, axis=0)
    #     inv_transform_matrix = np.array([[R_inv[0,0], R_inv[0,1], R_inv[0,2], T[0]],
    #                                      [R_inv[1,0], R_inv[1,1], R_inv[1,2], T[1]],
    #                                      [R_inv[2,0], R_inv[2,1], R_inv[2,2], T[2]],
    #                                      [0.0,       0.0,       0.0,       1.0]])

    #     inv_transform_matrix = transform_matrix0 @ inv_transform_matrix


    return cam_matrix, intrisics_matrix

def loadRealStateImages(filesPath):
    img_names = sorted(glob.glob(os.path.join(filesPath,'*.jpg')))
    img_list=[]
    names_list = []
    for name in img_names:
        i_name = name.split('/')[-1].split('.')[0]
        img = cv2.imread(name)
        img_list.append(img)
        names_list.append(i_name)
    #breakpoint()
    all_images = np.stack(img_list, axis=0)
    return all_images, names_list

def selectRealStateScene(datasetPath, scene_name, inverse_cam=False):
    camera_path = os.path.join(datasetPath,'test/'+scene_name+'.txt')
    imgs_path = os.path.join(datasetPath,'test_images/test/'+scene_name)

    imgs,_ = loadRealStateImages(imgs_path)
    poses, intrisics = loadRealStateCameraPoses(camera_path, dims=(imgs[0].shape[0], imgs[0].shape[1]), inverse_cam=inverse_cam)
    return imgs, poses, intrisics

def show_seg_mask(anns):
    if len(anns) == 0:
        return
    
    img = np.ones((anns[0].shape[0], anns[0].shape[1], 3))
    for ann in anns:
        m = ann
        color_mask = np.concatenate([np.random.random(3)])
        img[m>0] = color_mask
    # cv2.imwrite("img_mask.png",255*img[:,:,:-1])
    return img




