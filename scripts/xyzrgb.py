import math
import glob
import os
import numpy as np
from pathlib import Path
import pathlib
import numpy as np

rootdir = '/Users/carotenuto/clones/diffusion-net/experiments/headspace_ldmks/headspace_pcl_hmap100/drive-download-20211020T150402Z-001/'

for filepath in glob.iglob(rootdir + 'test/*/13*.txt'):
    # process pointcloud file
    lines = open(filepath, 'r').read().split('\n')[:-1]
    pcl = [[float(l) for l in lines[j].split(',')[:3]] for j in range(len(lines))]

    # find folder num
    folder_num = Path(filepath).parts[-2]
    folder_num_int = int(folder_num)

    # open pred
    f = open(rootdir + "preds/hmap" + str(folder_num_int) + ".txt", "r")
    preds = f.read().split('\n')[:-1]
    f.close()

    # make negative predictions zero
    preds_float = [float(x) for x in preds]
    preds_np = np.asarray(preds_float)
    preds_np[preds_np < 0] = 0

    # make vis dir if not exists
    pathlib.Path(rootdir + '/preds/vis').mkdir(parents=True, exist_ok=True)

    # create new xyzrgb with intensity
    f = open(rootdir + "preds/vis/xyzrgb" + str(folder_num) + ".txt", "w+")
    for i, el in enumerate(preds_np):
        f.write(str(pcl[i])[1:-1] + ', ' + str(el) + ', 0.0, 0.0\n')
    f.close()

# save points in file
# f = open(os.path.dirname(filepath) + "/hmap.txt", "w+")
# for e in output:
#    f.write(str(e) + '\n')
# f.close()
