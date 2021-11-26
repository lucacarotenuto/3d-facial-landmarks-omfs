# Cut region from high-resolution based on low-resolution predictions, and visualize or save pcl (for test set)

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
from tqdm import tqdm
import utils

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

    # Save dir
    SAVE_DIR = '/Users/carotenuto/Master Radboud/MscProj/refined_sets/refined_subnasal6_test'

    VISUALIZE = False

    #LANDMARK_INDICES = [8, 27, 30, 33, 36, 39, 45, 42, 60, 64]  # e.g. nosetip 31 has index 30
    #LANDMARK_INDICES = [33]
    #searchpath = 'test/*/13*.txt'
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
        preds = np.transpose(preds)
        # go through each landmark (class) in preds array, save only the maximum activation among all classes and save
        # the class with the maximum activation
        preds_sparse = np.zeros((preds.shape[1], 2))  # shape (vertices, 2)
        for i in tqdm(range(len(preds))):
            max_act = np.amax(preds, axis=0)  # shape (vertices,)
            preds_sparse[:, 0] = max_act
            max_act_cl = np.argmax(preds, axis=0)
            preds_sparse[:, 1] = max_act_cl

        # go through each landmark (each channel)
        #  create a region for each landmark
        #for i in range(preds.shape[1]):
        for i in range(3,4):
            print('landmark {}'.format(i))
            # only take vertices where prediction in that landmark channel is bigger than value
            verts_refined_mask = (preds_sparse[:, 1] == i) & (preds_sparse[:, 0] > 0.1)
            verts_refined = verts[verts_refined_mask]

            # # visualize preds region
            # v = pptk.viewer(verts_refined, show_axis=False)
            # v.set(point_size=0.4, show_axis=False, show_info=True)

            # # visualize overlayed preds region on low res image
            # color = np.zeros((len(verts),3))
            # color[verts_refined_mask] = [255,0,0]
            # v = pptk.viewer(verts, color, show_axis=False)
            # v.set(point_size=0.4, show_axis=False, show_info=True)
            # todo: find clusters in preds to remove outliers

            # find high res .obj file
            for j, fp_hres in enumerate(glob.iglob(os.path.join(HRES_DIR, folder_num, '13*.obj'))):
                assert j == 0, "More than one highres .obj file found"

            high_res_points, _ = pp3d.read_mesh(fp_hres)
            print('preds points higher 0.1 {}'.format(len(verts_refined)))

            if len(verts_refined) == 0:
                print('exception: not enough confident predictions in highres')

            method = 'radius'
            high_res_region = []
            high_res_region_mask = np.full((len(high_res_points)), False, dtype=bool)
            if method == 'delaunay':
                # take points from high-resolution point cloud that are in low-res preds hull
                if not isinstance(verts_refined, Delaunay):
                    hull = Delaunay(verts_refined)
                for k, coords in enumerate(high_res_points):
                    if hull.find_simplex(coords)>=0:
                        high_res_region.append(coords)
                        high_res_region_mask[k] = True
                print('highres points in hull {}'.format(len(high_res_region)))

            elif method == 'minmaxxyz':
                xmin, xmax, ymin, ymax, zmin, zmax = verts_refined[:,0].min(), verts_refined[:,0].max(),\
                                                    verts_refined[:, 1].min(), verts_refined[:, 1].max(),\
                                                    verts_refined[:, 2].min(), verts_refined[:, 2].max()
                for k, coords in enumerate(high_res_points):
                    if coords[0] > xmin and coords[0] < xmax and\
                                coords[1] > ymin and coords[1] < ymax and\
                                coords[2] > zmin and coords[2] < zmax:
                        high_res_region.append(coords)
                        high_res_region_mask[k] = True
            elif method == 'radius':
                high_res_region = []
                point_max = preds[i,:].argmax()
                coords_max = verts[point_max]

                for k, coords in enumerate(high_res_points):
                    dist = utils.eucl_dist(coords_max[0],coords_max[1],coords_max[2],
                                           coords[0],coords[1],coords[2],)
                    if dist < 25:
                        high_res_region.append(high_res_points[k])


            high_res_region_np = np.array(high_res_region)

            # c_highres_mask = np.full(len(high_res_points), False, dtype=bool)
            # c_highres_mask[point_max] = True
            # c_highres = np.zeros((len(high_res_points),3))
            # c_highres[point_max] = [255,0,0]
            # overlay = pptk.viewer(high_res_points, c_highres, show_axis=True)
            # overlay.set(point_size=0.4, show_info=True)



            # # visualize high res points with
            # color = np.zeros((len(high_res_points), 3))
            # color[high_res_region_mask] = [255,0,0]
            # overlay = pptk.viewer(high_res_points, color, show_axis=False)
            # overlay.set(point_size=0.4, show_axis=False, show_info=True)

            # todo: overlay pictures high res low res to debug
            #v = pptk.viewer(high_res_region_np, col, show_axis=False)
            #v.set(point_size=0.1, show_axis=False, show_info=True)
            if VISUALIZE:
                # visualize high res region on low res points
                c_verts_refined = np.zeros((len(verts),3))
                c_verts_refined[verts_refined_mask] = [255,0,0]
                v_overlay = np.concatenate((verts, high_res_region_np))
                c_highres_region_mask = np.full(len(high_res_region_np), True, dtype=bool)
                c_highres_region = np.zeros((len(c_highres_region_mask),3))
                c_highres_region[c_highres_region_mask] = [0,255,0]
                c_overlay = np.concatenate((c_verts_refined,
                                            c_highres_region))
                overlay = pptk.viewer(v_overlay, c_overlay, show_axis=True)
                overlay.set(point_size=0.4, show_info=True)

                # overlayed picture
                #v_overlay = verts
                #c_overlay = np.zeros((v_overlay.shape[0],3))
                #c_overlay[verts_refined_mask] = [255,0,0]
                #overlay = pptk.viewer(v_overlay, c_overlay, show_axis=False)
                #overlay.set(point_size=0.4, show_axis=False, show_info=True)

            # save
            dir = os.path.join(SAVE_DIR, folder_num, str(i))
            if not os.path.exists(dir):
                os.makedirs(dir)
            np.savetxt(os.path.join(dir, os.path.basename(fp_pred)[:-3] + 'txt'), X=high_res_region_np, fmt='%10.7f',
                       delimiter=',')


if __name__ == "__main__":
    main()

