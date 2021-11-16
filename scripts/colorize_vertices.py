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
from tqdm import tqdm
import potpourri3d as pp3d

# Set directory with 'test' folder and 'preds' folder (if visualizing predictions)
rootdir = '/Users/carotenuto/Master Radboud/MscProj/preds_pcl_all_c256_l10/'

# Set true if single point colorization or False if heatmap colorization
POINT_PREDS = True
# Set true if pointcloud or false if mesh
IS_PCL = True
# Set true if you are visualizing ground truth or false if visualizing predictions
IS_GT = True

LANDMARK_INDICES = [8, 27, 30, 33, 36, 39, 45, 42, 60, 64]  # e.g. nosetip 31 has index 30

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

    # open pred pkl
    if IS_GT:
        # open gt label
        with open('{}test/{}/hmap_per_class.pkl'.format(rootdir, str(folder_num)), 'rb') as f:
            labels = pickle.load(f)
        # only keep selected landmarks
        labels = [item for pos, item in enumerate(labels) if pos in LANDMARK_INDICES]
        # restore sparse representation
        labels_sparse = np.zeros((len(LANDMARK_INDICES), len(verts)))
        for j in range(len(labels)):
            for k in range(len(labels[j])):
                pos = labels[j][k, 0]
                if POINT_PREDS:
                    act = 0 if labels[j][k, 1] < 1 else 1
                else:
                    act = labels[j][k, 1]
                labels_sparse[j, int(pos)] = act
        preds = labels_sparse
    else:
        # open predictions
        with open(rootdir + 'preds/hmap_per_class' + str(folder_num) + '.pkl', 'rb') as f:
            preds = pickle.load(f)
        # only keep selected landmarks
        #preds = [item for pos, item in enumerate(preds) if pos in LANDMARK_INDICES]
        #preds = preds[:, LANDMARK_INDICES]
        # restore original shape
        preds = np.transpose(preds)
        # make negative predictions zero

    preds[preds < 0] = 0

    if POINT_PREDS:
        # only keep highest activation
        preds_pt = np.zeros_like(preds)
        preds_pt[np.arange(len(preds)), preds.argmax(1)] = 1

        preds = preds_pt

    # go through each landmark (class) in preds array, save only the maximum activation among all classes and save
    # the class with the maximum activation
    outp_mask = np.zeros((preds.shape[1], 2)) # shape (vertices, 2)
    for i in tqdm(range(len(preds))):
        max_act = np.amax(preds, axis=0) # shape (vertices,)
        outp_mask[:,0] = max_act
        max_act_cl = np.argmax(preds, axis=0)
        outp_mask[:,1] = max_act_cl

    # make vis dir if not exists
    if IS_GT:
        pathlib.Path(rootdir + '/gt_vis').mkdir(parents=True, exist_ok=True)
    else:
        pathlib.Path(rootdir + '/preds/vis').mkdir(parents=True, exist_ok=True)

    # create new xyzrgb with intensity and alternate between rgb channel
    if POINT_PREDS:
        if IS_GT:
            f = open(rootdir + 'gt_vis/gt_pt_' + str(folder_num) + '.txt', 'w+')
        else:
            f = open(rootdir + 'preds/vis/pred_pt_' + str(folder_num) + '.txt', 'w+')
    else:
        if IS_GT:
            f = open(rootdir + 'gt_vis/gt_' + str(folder_num) + '.txt', 'w+')
        else:
            f = open(rootdir + 'preds/vis/pred_' + str(folder_num) + '.txt', 'w+')
    for i, el in tqdm(enumerate(outp_mask)):
        if IS_PCL:
            if IS_GT:
                f.write(str(verts[i])[1:-1] + ', 0.0, {}, 0.0\n'.format(el[0]))
            else:
                f.write(str(verts[i])[1:-1] + ', {}, 0.0, 0.0\n'.format(el[0]))
        else:
            if IS_GT:
                f.write(', '.join(str(e) for e in verts[i]) + ', 0.0, {}, 0.0\n'.format(el[0]))
            else:
                f.write(', '.join(str(e) for e in verts[i]) + ', {}, 0.0, 0.0\n'.format(el[0]))
    f.close()