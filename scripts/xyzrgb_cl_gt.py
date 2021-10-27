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

rootdir = '/Users/carotenuto/clones/diffusion-net/experiments/headspace_ldmks/headspace_pcl_hmap100_3k_noweights/test/00019/'

# Set True if single point colorization or False if heatmap colorization
POINT_PREDS = True

for filepath in glob.iglob(rootdir + '13*.txt'):
    # process pointcloud file
    lines = open(filepath, 'r').read().split('\n')[:-1]
    pcl = [[float(l) for l in lines[j].split(',')[:3]] for j in range(len(lines))]

    # find folder num
    folder_num = Path(filepath).parts[-2]
    folder_num_int = int(folder_num)

    # open pred pkl
    with open(rootdir + "/hmap_per_class.pkl", 'rb') as f:
        labels = pickle.load(f)
        #buffer = io.BytesIO(f.read())
        #preds = torch.load(f, map_location=torch.device('cpu'))
    #preds = torch.load(buffer, map_location=torch.device('cpu'))

    # restore sparse representation
    labels_sparse = []
    labels_sparse = np.zeros((68, len(pcl)))
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
            f.write(str(pcl[i])[1:-1] + ', ' + str(el[0]) + ', 0.0, 0.0\n')
        elif el[1] % 2 == 1:
            f.write(str(pcl[i])[1:-1] + ', 0.0, ' + str(el[0]) + ', 0.0\n')
        #elif el[1] % 3 == 2:
        #    f.write(str(pcl[i])[1:-1] + ', 0.0, 0.0, ' + str(el[0]) + '\n')
    f.close()

#save points in file
# f = open(os.path.dirname(filepath) + "/hmap.txt", "w+")
# for e in output:
#    f.write(str(e) + '\n')
# f.close()
