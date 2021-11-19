import io
import math
import glob
import os
import numpy as np
from pathlib import Path
import pathlib
import numpy as np
import pickle
from tqdm import tqdm
froms cipy.spatial import ConvexHull


# Set directory with 'test' folder and 'preds' folder (if visualizing predictions)
rootdir = '/Users/carotenuto/Master Radboud/MscProj/preds_pcl_all_c256_l10/'


LANDMARK_INDICES = [8, 27, 30, 33, 36, 39, 45, 42, 60, 64]  # e.g. nosetip 31 has index 30

searchpath = 'test/*/13*.txt'
for filepath in glob.iglob(rootdir + searchpath):
    # find folder num
    folder_num = Path(filepath).parts[-2]
    folder_num_int = int(folder_num)

    # process pointcloud file
    lines = open(filepath, 'r').read().split('\n')[:-1]
    verts = [[float(l) for l in lines[j].split(',')[:3]] for j in range(len(lines))]
    verts = np.array(verts)

    # open predictions
    with open(rootdir + 'preds/hmap_per_class' + str(folder_num) + '.pkl', 'rb') as f:
        preds = pickle.load(f)
    # go through each landmark (each channel)
    #  create a region for each landmark
    for i in range(preds.shape[1]):
        # only take vertices where prediction in that landmark channel is bigger than value
        verts_refined = verts[preds[:,i] > 0.1]
        print('test')

        # how to get the refined region?
        #for i in range(3):
        #    print(verts[:, i].min(), verts[:, 1].max())
        # gives a 3d quad, so too many points ouside there approx circle
        # convex hull?

        hull = ConvexHull(verts)

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0