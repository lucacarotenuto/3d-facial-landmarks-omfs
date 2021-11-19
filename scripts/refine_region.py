import io
import math
import glob
import os
import numpy as np
from pathlib import Path
import pathlib
import numpy as np
import pickle
from scipy.spatial import Delaunay
import potpourri3d as pp3d
import pptk # requires python 3.6!

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0

def main():
    # Directory should contain 'test' folder and 'preds' folder
    PREDS_DIR = '/Users/carotenuto/Master Radboud/MscProj/preds_pcl_all_c256_l10/'

    # High res .obj directory
    HRES_DIR = '/Users/carotenuto/Master Radboud/MscProj/pcl_testset_fullres'

    LANDMARK_INDICES = [8, 27, 30, 33, 36, 39, 45, 42, 60, 64]  # e.g. nosetip 31 has index 30

    searchpath = 'test/*/13*.txt'
    for fp_pred in glob.iglob(PREDS_DIR + searchpath):
        # find folder num
        folder_num = Path(fp_pred).parts[-2]
        folder_num_int = int(folder_num)

        # process pointcloud file
        lines = open(fp_pred, 'r').read().split('\n')[:-1]
        verts = [[float(l) for l in lines[j].split(',')[:3]] for j in range(len(lines))]
        verts = np.array(verts)

        # open predictions
        with open(PREDS_DIR + 'preds/hmap_per_class' + str(folder_num) + '.pkl', 'rb') as f:
            preds = pickle.load(f)
        # go through each landmark (each channel)
        #  create a region for each landmark
        for i in range(preds.shape[1]):
            print('landmark {}'.format(i))
            # only take vertices where prediction in that landmark channel is bigger than value
            verts_refined = verts[preds[:, i] > 0.1]
            print('test')

            # find high res .obj file
            for j, fp_hres in enumerate(glob.iglob(os.path.join(HRES_DIR, folder_num, '13*.obj'))):
                assert j == 0, "More than one highres .obj file found"

            high_res_points, _ = pp3d.read_mesh(fp_hres)

            # take points from high-resolution point cloud that are in low-res preds hull
            print('preds points higher 0.1 {}'.format(len(verts_refined)))
            k = 0
            if not isinstance(verts_refined, Delaunay):
                hull = Delaunay(verts_refined)
            high_res_region = []
            for coords in high_res_points:
                if hull.find_simplex(coords)>=0:
                    k+=1
                    high_res_region.append(coords)
            print('highres points in hull {}'.format(k))

            high_res_region_np = np.array(high_res_region)

            # todo: overlay pictures high res low res to debug
            v = pptk.viewer(high_res_region_np, show_axis=False)
            v.set(point_size=0.1, show_axis=False, show_info=True)

if __name__ == "__main__":
    main()

