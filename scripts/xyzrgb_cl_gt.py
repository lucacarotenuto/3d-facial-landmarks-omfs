### OBSOLETE (xyzrgb_cl.py)

import io
import math
import glob
import os
import numpy as np
from pathlib import Path
import pathlib
import numpy as np
import pickle
import torch
import potpourri3d as pp3d

rootdir = '/Users/carotenuto/Master Radboud/MscProj/gt_mesh_50kf/'

# Set True if single point colorization or False if heatmap colorization
POINT_PREDS = False
# Set True if pointcloud or false if mesh
IS_PCL = False

searchpath = 'test/*/13*.txt' if IS_PCL else 'test/*/13*.obj'
for filepath in glob.iglob(rootdir + searchpath):
    if IS_PCL:
        # process pointcloud file
        lines = open(filepath, 'r').read().split('\n')[:-1]
        verts = [[float(l) for l in lines[j].split(',')[:3]] for j in range(len(lines))]
    else:
        # process mesh
        verts, faces = pp3d.read_mesh(filepath)


    # find folder num
    folder_num = Path(filepath).parts[-2]
    folder_num_int = int(folder_num)

    # open gt label
    with open("{}test/{}/hmap_per_class.pkl".format(rootdir, str(folder_num)), 'rb') as f:
        labels = pickle.load(f)
        #buffer = io.BytesIO(f.read())
        #preds = torch.load(f, map_location=torch.device('cpu'))
    #preds = torch.load(buffer, map_location=torch.device('cpu'))

    # restore sparse representation
    labels_sparse = []
    labels_sparse = np.zeros((68, len(verts)))
    for j in range(len(labels)):
        for k in range(len(labels[j])):
            pos = labels[j][k, 0]
            if POINT_PREDS:
                act = 0 if labels[j][k, 1] < 1 else 1
            else:
                act = labels[j][k, 1]
            labels_sparse[j, int(pos)] = act

    preds = labels_sparse

    # make negative predictions zero
    #preds_float = [float(x) for x in preds]
    preds[preds < 0] = 0

    # go through each landmark (class) in preds array, save only the maximum activation among all classes and save
    # the class with the maximum activation
    outp_mask = np.zeros((preds.shape[1], 2)) # shape (vertices, 2)
    for i in range(len(preds)):
        max_act = np.amax(preds, axis=0) # shape (vertices,)
        outp_mask[:,0] = max_act
        max_act_cl = np.argmax(preds, axis=0)
        outp_mask[:,1] = max_act_cl

    # make vis dir if not exists
    pathlib.Path(rootdir + '/gt_vis').mkdir(parents=True, exist_ok=True)

    # create new xyzrgb with intensity and alternate between rgb channel
    if POINT_PREDS:
        f = open(rootdir + "gt_vis/xyzrgb_pt" + str(folder_num) + ".txt", "w+")
    else:
        f = open(rootdir + "gt_vis/xyzrgb" + str(folder_num) + ".txt", "w+")
    for i, el in enumerate(outp_mask):
        if el[1] % 2 == 0:
            if IS_PCL:
                f.write(str(verts[i])[1:-1] + ', ' + str(el[0]) + ', 0.0, 0.0\n')
            else:
                f.write(', '.join(str(e) for e in verts[i]) + ', ' + str(el[0]) + ', 0.0, 0.0\n')
        elif el[1] % 2 == 1:
            if IS_PCL:
                f.write(str(verts[i])[1:-1] + ', 0.0, ' + str(el[0]) + ', 0.0\n')
            else:
                f.write(', '.join(str(e) for e in verts[i]) + ', 0.0, ' + str(el[0]) + ', 0.0\n')
    f.close()

#save points in file
# f = open(os.path.dirname(filepath) + "/hmap.txt", "w+")
# for e in output:
#    f.write(str(e) + '\n')
# f.close()
