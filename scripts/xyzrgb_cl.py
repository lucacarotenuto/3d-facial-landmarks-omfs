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

rootdir = '/Users/carotenuto/clones/diffusion-net/experiments/headspace_ldmks/headspace_pcl_hmap100_3k/'
POINT_PREDS = True

for filepath in glob.iglob(rootdir + 'test/*/13*.txt'):
    # process pointcloud file
    lines = open(filepath, 'r').read().split('\n')[:-1]
    pcl = [[float(l) for l in lines[j].split(',')[:3]] for j in range(len(lines))]

    # find folder num
    folder_num = Path(filepath).parts[-2]
    folder_num_int = int(folder_num)

    # open pred pkl
    with open(rootdir + "preds/hmap_per_class" + str(folder_num_int) + ".pkl", 'rb') as f:
        preds = pickle.load(f)
        #buffer = io.BytesIO(f.read())
        #preds = torch.load(f, map_location=torch.device('cpu'))
    #preds = torch.load(buffer, map_location=torch.device('cpu'))
    # restore original shape
    preds = np.transpose(preds)
    #preds = np.reshape(preds, (68, int(len(preds) / 68)))

    # make negative predictions zero
    #preds_float = [float(x) for x in preds]
    preds[preds < 0] = 0

    if POINT_PREDS:
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
    pathlib.Path(rootdir + '/preds/vis').mkdir(parents=True, exist_ok=True)

    # create new xyzrgb with intensity and alternate between rgb channel
    if POINT_PREDS:
        f = open(rootdir + "preds/vis/xyzrgb_pt" + str(folder_num) + ".txt", "w+")
    else:
        f = open(rootdir + "preds/vis/xyzrgb" + str(folder_num) + ".txt", "w+")
    for i, el in tqdm(enumerate(outp_mask)):
        if el[1] % 2 == 0:
            f.write(str(pcl[i])[1:-1] + ', ' + str(el[0]) + ', 0.0, 0.0\n')
        elif el[1] % 2 == 1:
            f.write(str(pcl[i])[1:-1] + ', 0.0, ' + str(el[0]) + ', 0.0\n')
        #elif el[1] % 3 == 2:
        #    f.write(str(pcl[i])[1:-1] + ', 0.0, 0.0, ' + str(el[0]) + '\n')
    f.close()

# save points in file
#f = open(os.path.dirname(filepath) + "/hmap.txt", "w+")
#for e in output:
#    f.write(str(e) + '\n')
#f.close()
