import os
import sys

def get_scene_list(path):
    l = sorted(os.listdir(path))
    return l
