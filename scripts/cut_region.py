import glob
import os.path
import pickle
import numpy as np
import potpourri3d as pp3d
from pathlib import Path
from utils import eucl_dist
from utils import dist_between_points


from tqdm import tqdm

TRAIN_DIR = '/Users/carotenuto/Master Radboud/MscProj/subjects_1-150_train'
LOWRES_DIR = '/Users/carotenuto/Master Radboud/MscProj/headspace_pcl_all'
SAVE_DIR = '/Users/carotenuto/Master Radboud/MscProj/refined_sets/refined1'
LANDMARK_INDICES = [8, 27, 30, 33, 36, 39, 45, 42, 60, 64]
LDMKS = np.load('/Users/carotenuto/Master Radboud/MscProj/headspace_pcl_all/ldmks.pkl',
                allow_pickle=True)  # shape (samples, landmarks + 1, 3)
LDMKS = LDMKS[:,np.concatenate(([0], LANDMARK_INDICES)),:]

for filepath in glob.glob(os.path.join(LOWRES_DIR, '*', '*.txt')):
    folder_num = Path(filepath).parts[-2]
    folder_num_int = int(folder_num)

    # REMOVE SELECTION LATER
    if folder_num_int > 100 or (folder_num_int >= 9 and folder_num_int <= 38):
        continue

    targets = np.load(os.path.join(LOWRES_DIR, folder_num, 'hmap_per_class.pkl'), allow_pickle=True)

    # process lowres pointcloud file
    lines = open(filepath, 'r').read().split('\n')[:-1]
    verts = [[float(l) for l in lines[j].split(',')[:3]] for j in range(len(lines))]
    verts = np.array(verts)

    # only keep selected landmarks (remove later)
    landmark_indices = [8, 27, 30, 33, 36, 39, 45, 42, 60, 64]  # e.g. nosetip 31 has index 30
    targets = [item for pos, item in enumerate(targets) if pos in landmark_indices]

    # find highres obj file
    for i, obj_path in enumerate(glob.iglob(os.path.join(TRAIN_DIR, folder_num, '*.obj'))):
        if i >= 1:
            raise RuntimeError('more than one obj found')

    # process high res .obj file to pcl
    pcl_hres, _ = pp3d.read_mesh(obj_path)

    # identify landmark coordinates for file
    for i in range(LDMKS.shape[0]):
        if int(LDMKS[i, 0, 0]) == int(folder_num):
            ldmks_idx = i
            break
    ldmks_per_file = LDMKS[ldmks_idx, 1:, :]  # shape (landmarks, 3)

    point_list = np.zeros((len(LANDMARK_INDICES)))

    # take the vertices from lowres pcl that has activation and save coords
    for m, target in enumerate(targets):
        coords = verts[[int(k) for k in target[:,0]]]

        # get min, max for x, y, z
        xmin, xmax, ymin, ymax, zmin, zmax = coords[:, 0].min(), coords[:, 0].max(), \
                                             coords[:, 1].min(), coords[:, 1].max(), \
                                             coords[:, 2].min(), coords[:, 2].max()



        # get points from highres pcl within criteria
        d = 3 # delta
        pcl_hres_region = pcl_hres[(pcl_hres[:, 0] > xmin - d) & (pcl_hres[:, 0] < xmax + d) & \
                                    (pcl_hres[:, 1] > ymin - d) & (pcl_hres[:, 1] < ymax + d) & \
                                    (pcl_hres[:, 2] > zmin - d) & (pcl_hres[:, 2] < zmax + d)]
        #print(pcl_hres_region.shape)

        # save highres pcl for each landmark
        dir = os.path.join(SAVE_DIR, folder_num, str(m))
        if not os.path.exists(dir):
            os.makedirs(dir)
        np.savetxt(os.path.join(dir, os.path.basename(filepath)[:-3] + 'txt'), X = pcl_hres_region, fmt = '%10.7f', delimiter = ',')

        # get points with shortest distance to actual landmark
        shortest_dist = 99999999
        target_coords = ldmks_per_file[m]
        for j in range(len(pcl_hres_region)):
            # calc distance
            orig_coords = pcl_hres_region[j]
            dist = eucl_dist(orig_coords[0], orig_coords[1], orig_coords[2], target_coords[0], target_coords[1],
                             target_coords[2])
            if dist < shortest_dist:
                shortest_dist = dist
                pt_shortest_dist = j
        point_list[m] = pt_shortest_dist

        # go through each point and create activation depending on proximity to landmark point (heatmap)
        ldmk_point = point_list[m]
        output = np.array([])
        for n, point in enumerate(pcl_hres_region):
            dist = dist_between_points(pcl_hres_region, n, int(ldmk_point))

            # calculate activation
            if ldmk_point == n:
                activation = 1
            elif dist <= 1 and n != m:
                activation = 0.75
            elif dist <= 2 and n != m:
                activation = 0.5
            elif dist <= 3 and n != m:
                activation = 0.25
            else:
                continue

            # if array does not exist create new with else concatenate
            if output.size == 0:
                output = np.array([[n, activation]])
            else:
                output = np.append(output, np.array([[n, activation]]), axis=0)

        with open(os.path.join(SAVE_DIR, folder_num, str(m), 'hmap_per_class.pkl'), 'wb') as f:
            pickle.dump(output, f)