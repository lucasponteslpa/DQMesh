import numpy as np
import os
import torch
import cv2


def write_obj(folder,
              v_pos=None,
              v_nrm=None,
              v_tex=None,
              t_pos_idx=None,
              t_nrm_idx=None,
              t_tex_idx=None,
              save_material=True,
              file_name = 'mesh.obj'):
    obj_file = os.path.join(folder, file_name)
    print("Writing mesh: ", obj_file)
    with open(obj_file, "w") as f:
        #f.write("mtllib mesh.mtl\n")
        #f.write("g default\n")

        v_pos = v_pos.detach().cpu().numpy() if v_pos is not None else None
        v_nrm = v_nrm.detach().cpu().numpy() if v_nrm is not None else None
        v_tex = v_tex.detach().cpu().numpy() if v_tex is not None else None

        t_pos_idx = t_pos_idx.detach().cpu().numpy() if t_pos_idx is not None else None
        t_nrm_idx = t_nrm_idx.detach().cpu().numpy() if t_nrm_idx is not None else None
        t_tex_idx = t_tex_idx.detach().cpu().numpy() if t_tex_idx is not None else None

        print("    writing %d vertices" % len(v_pos))
        for v in v_pos:
            f.write('v {} {} {} \n'.format(v[0], v[1], v[2]))

        if v_tex is not None:
            print("    writing %d texcoords" % len(v_tex))
            #assert(len(t_pos_idx) == len(t_tex_idx))
            for v in v_tex:
                #f.write('vt {} {} \n'.format(v[0], 1.0 - v[1]))
                f.write('vt {} {} \n'.format(v[0], v[1]))

        # faces
        f.write("s 1 \n")
        f.write("g pMesh1\n")
        f.write("usemtl defaultMat\n")

        # Write faces
        print("    writing %d faces" % len(t_pos_idx))
        for i in range(len(t_pos_idx)):
            f.write("f")
            for j in range(3):
                # Handle cases where UVs or normals are None
                pos_idx = str(t_pos_idx[i][j] + 1)  # Vertex index (1-based)
                tex_idx = str(t_tex_idx[i][j] + 1) if v_tex is not None else ''
                nrm_idx = str(t_nrm_idx[i][j] + 1) if v_nrm is not None else ''

                if tex_idx and nrm_idx:
                    f.write(f" {pos_idx}/{tex_idx}/{nrm_idx}")
                elif tex_idx:  # Only texture coordinates
                    f.write(f" {pos_idx}/{tex_idx}")
                elif nrm_idx:  # Only normals
                    f.write(f" {pos_idx}//{nrm_idx}")
                else:  # Only vertex indices
                    f.write(f" {pos_idx}")
            f.write("\n")


    print("Done exporting mesh")
