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
from sympy import Plane, Point3D

# Directory should contain 'test' folder and 'preds' folder (if visualizing predictions)
#rootdir = 'C:\\Users\\Luca\\Documents\\GitHub\\3d-facial-landmarks-omfs\\diffusion-net\\experiments\\refine_ldmks\\refined_141_manual_inference\\'
rootdir = '/Users/carotenuto/Documents/GitHub/3d-facial-landmarks-omfs/diffusion-net/experiments/headspace_ldmks/no_op/'
#rootdir = '/Users/carotenuto/Master Radboud/MscProj/manual_results/pcl_196_30k/'
#rootdir = '/Volumes/Extreme SSD/MscProject/vis_face'

# Set true if single point colorization or False if heatmap colorization
POINT_PREDS = True
# Set true if pointcloud or false if mesh
IS_PCL = True
# Set true if you are visualizing ground truth or false if visualizing predictions
IS_GT = False
# Set true if segmentation predictions
IS_SEG = False
# Set true if refinement predictions
IS_REFINED = False
# Test if True, Train if False
IS_TEST = True
NO_OP = True
PER_CLASS_VIS = True
# Set true if preds in 'wrong' half of the face should be removed
REMOVE_SYMMETRICAL_PREDS = True

LANDMARK_INDICES = [8, 27, 30, 31, 33, 35, 36, 39, 42, 45, 60, 64]  # e.g. nosetip 31 has index 30
#LANDMARK_INDICES = [8, 27, 30, 33, 36, 39, 42, 45, 60, 64]  
#LANDMARK_INDICES = [30, 39, 42, 60, 64]
#LANDMARK_INDICES = [30]

#LANDMARK_INDICES = [30]  # e.g. nosetip 31 has index 30

if not IS_REFINED:
    if not NO_OP:
        searchpath = 'test/*/13*.txt' if IS_PCL else 'test/*/13*.obj'
    else:
        searchpath = ('test' if IS_TEST else 'train') + '/*.txt'
else:
    searchpath = ('test' if IS_TEST else 'train') + '/*/*/13*.txt'

for filepath in glob.iglob(os.path.join(rootdir, searchpath)):
    fname = Path(filepath).parts[-1]
    if fname == '695__1479_al_cent.txt':
        if IS_PCL:
            # process pointcloud file
            lines = open(filepath, 'r').read().split('\n')[:-1]
            verts = [[float(l) for l in lines[j].split(',')[:3]] for j in range(len(lines))]
        else:
            # process mesh
            verts, faces = pp3d.read_mesh(filepath)

        # find folder num
        if not IS_REFINED:
            if not NO_OP:
                folder_num = Path(filepath).parts[-2]
            else: 
                folder_num = Path(filepath).parts[-1]
        else:
            folder_num = Path(filepath).parts[-3]
            folder_num_ldmk = Path(filepath).parts[-2]

        #if int(folder_num_ldmk) < 7:
        #    continue

        # open pred pkl
        if IS_GT:
            # open gt label
            if not IS_REFINED:
                with open(os.path.join(rootdir, 'test', str(folder_num), 'hmap_per_class.pkl'), 'rb') as f:
                    labels = pickle.load(f)
                # only keep selected landmarks
                #labels = [item for pos, item in enumerate(labels) if pos in LANDMARK_INDICES]
                #labels = [item for pos, item in enumerate(labels)]
                # restore sparse representation
                labels_sparse = np.zeros((len(LANDMARK_INDICES), len(verts)))
                for j in range(len(labels)):
                    if j in LANDMARK_INDICES:
                        for k in range(len(labels[j])):
                            pos = labels[j][k, 0]

                            if POINT_PREDS:
                                act = 0 if labels[j][k, 1] < 1 else 1
                            else:
                                act = labels[j][k, 1]
                            #if not np.isnan(pos):
                            labels_sparse[LANDMARK_INDICES.index(j), int(pos)] = act
                preds = labels_sparse
            else:
                with open(os.path.join(rootdir, 'test' if IS_TEST else 'train', str(folder_num), folder_num_ldmk, 'hmap_per_class.pkl'), 'rb') as f:
                    labels = pickle.load(f)
                    # only keep selected landmarks
                    #labels = [item for pos, item in enumerate(labels) if pos in LANDMARK_INDICES]
                    # restore sparse representation
                    labels_sparse = np.zeros((1,len(verts)))
                    for j in range(len(labels)):
                        act = labels[j][1]
                        pos = labels[j][0]
                        labels_sparse[0,int(pos)] = act
                    preds = labels_sparse
        else:
            # open predictions
            #with open(rootdir + 'preds\\hmap_per_class' + str(folder_num) + ('_{}'.format(folder_num_ldmk)\
            #                                                if IS_REFINED else '') + '.pkl', 'rb') as f:
            
            
            with open(os.path.join(rootdir, 'preds', 'hmap_per_class' + str(folder_num) + ('_{}'.format(folder_num_ldmk) if IS_REFINED else '') + '.pkl'), 'rb') as f:
                preds = pickle.load(f)
            
            
            
            # only keep selected landmarks
            #preds = [item for pos, item in enumerate(preds) if pos in LANDMARK_INDICES]
            #preds = preds[:, LANDMARK_INDICES]
            # restore original shape
            preds = np.transpose(preds)
            # make negative predictions zero

        preds[preds < 0] = 0

        if REMOVE_SYMMETRICAL_PREDS:
            # get nasion, pronasale and subnasale coordinates
            ns = verts[np.argmax(preds[1,:])]
            prn = verts[np.argmax(preds[2,:])]
            sn = verts[np.argmax(preds[4,:])]

            # create a plane
            plane = Plane(Point3D(ns), Point3D(prn), Point3D(sn))

            # check on which side of the plane prediction points are
            # to reduce computational complexity, convert to numpy and only check predictions with activation higher than threshold
            #verts_np = np.array(verts)
            #verts_np = preds[3,:]>0.3
            for i, point in enumerate(tqdm(verts)):
                if preds[3, i] > 0.2:
                    if plane.equation(point[0], point[1], point[2]) > 0:
                        preds[3, i] = 0
                if preds[5, i] > 0.2:
                    if plane.equation(point[0], point[1], point[2]) < 0:
                        preds[5, i] = 0
    
        if POINT_PREDS:
            # only keep highest activation
            preds_pt = np.zeros_like(preds)
            preds_pt[np.arange(len(preds)), preds.argmax(1)] = 1

            preds = preds_pt

        # Visualize per class
        if PER_CLASS_VIS:
            for it in range(preds.shape[0]):
                f = open(rootdir + 'preds/vis/pred_' + str(folder_num) + ('_{}'.format(folder_num_ldmk
                                                                        if IS_REFINED else '')) + '_CLASS{}.txt'.format(it), 'w+')
                for i in range(preds.shape[1]):
                    f.write(str(verts[i])[1:-1] + ', {}, 0.0, 0.0\n'.format(preds[it][i]))

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
            pathlib.Path(os.path.join(rootdir, 'gt_vis')).mkdir(parents=True, exist_ok=True)
        else:
            pathlib.Path(os.path.join(rootdir, 'preds', 'vis')).mkdir(parents=True, exist_ok=True)

        # create new txt file
        if POINT_PREDS:
            if IS_GT:
                f = open(os.path.join(rootdir, 'gt_vis, gt_pt_' + str(folder_num) +  ('_{}'.format(folder_num_ldmk
                                                                        if IS_REFINED else '')) +'.txt'), 'w+')
            else:
                f = open(os.path.join(rootdir, 'preds', 'vis', 'pred_pt_' + str(folder_num) + ('_{}'.format(folder_num_ldmk
                                                                if IS_REFINED else '')) + '.txt'), 'w+')
        else:
            if IS_GT:
                f = open(os.path.join(rootdir, 'gt_vis', 'gt_' + str(folder_num) + ('_{}'.format(folder_num_ldmk)
                                                                    if IS_REFINED else '') + '.txt'), 'w+')
            else:
                f = open(rootdir + 'preds/vis/pred_' + str(folder_num) + ('_{}'.format(folder_num_ldmk
                                                                if IS_REFINED else '')) + '.txt', 'w+')
        # write intensity
        for i, el in tqdm(enumerate(outp_mask)):
            if IS_PCL:
                if IS_GT:
                    f.write(str(verts[i])[1:-1] + ', 0.0, {}, 0.0\n'.format(el[0]))
                else:
                    if IS_SEG:
                        f.write(str(verts[i])[1:-1] + ', {}, 0.0, 0.0\n'.format(1.0 if el[1] > 0 else 0.0))
                    else:
                        #f.write(str(verts[i])[1:-1] + ', {}, 0.0, 0.0\n'.format(1.0 if el[1] == 4.0 and el[0] > 0.2 else 0.0))
                        f.write(str(verts[i])[1:-1] + ', {}, 0.0, 0.0\n'.format(el[0]))
            else:
                if IS_GT:
                    f.write(', '.join(str(e) for e in verts[i]) + ', 0.0, {}, 0.0\n'.format(el[0]))
                else:
                    f.write(', '.join(str(e) for e in verts[i]) + ', {}, 0.0, 0.0\n'.format(el[0]))
        f.close()