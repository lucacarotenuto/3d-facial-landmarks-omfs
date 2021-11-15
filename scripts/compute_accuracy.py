# Compute the distance of each predicted landmark to their corresponding target

import pickle
import glob
import os
import numpy as np
import potpourri3d as pp3d
from tqdm import tqdm
from utils import eucl_dist

ROOTDIR = '/Users/carotenuto/Documents/GitHub/3d-facial-landmarks-omfs/diffusion-net/experiments/headspace_ldmks/headspace_pcl_all'
LANDMARK_INDICES = [8, 27, 30, 33, 36, 39, 42, 45, 60, 64]  # e.g. nosetip 31 has index 30

# determine total prediction number and num landmarks
total_preds = 0
for path in glob.iglob(os.path.join(ROOTDIR, 'preds/hmap_per_class*.pkl')):
    total_preds += 1
with open(path, 'rb') as f:
    pred = pickle.load(f)
num_ldmks = pred.shape[1]
# num_ldmks = 10 # define landmarks manually if predictions include more landmarks

# error matrix (ldmks, preds)
error_matrix = np.zeros((num_ldmks, total_preds))

# for each prediction
for pred_idx, path in tqdm(enumerate(glob.iglob(os.path.join(ROOTDIR, 'preds/hmap_per_class*.pkl')))):
    with open(path, 'rb') as f:
        pred = pickle.load(f)
    # only keep selected landmarks e.g. if predictions include more landmarks than necessary
    #ldmk_indcs = [8, 27, 30, 33, 36, 39, 42, 45, 60, 64]
    #pred = pred[:, ldmk_indcs]

    # get the num part e.g. '00004'
    folder_num = path[-9:-4]

    # load target
    with open(os.path.join(ROOTDIR, 'test/{}/hmap_per_class.pkl'.format(folder_num)), 'rb') as f:
        target = pickle.load(f)
    # only keep selected landmarks
    target = [item for pos, item in enumerate(target) if pos in LANDMARK_INDICES]

    # make point activations (pred is non-sparse representation (ldmks, verts))
    pred = np.transpose(pred)
    pred_pt = np.zeros_like(pred)
    pred_pt[np.arange(len(pred)), pred.argmax(1)] = 1

    # get vertex xyz coordinate
    for file in os.listdir(os.path.join(ROOTDIR, 'test/{}'.format(folder_num))):
        # .txt extension for pcl
        if file.endswith('.txt'):
            verts_filepath = os.path.join(ROOTDIR, 'test/{}'.format(folder_num), file)

            # todo make helper function to load .txt into pcl array
            lines = open(verts_filepath, 'r').read().split('\n')[:-1]
            verts = [[float(l) for l in lines[j].split(',')[:3]] for j in range(len(lines))]
            break

        elif file.endswith('.obj'): # .obj extension for mesh
            verts_filepath = os.path.join(ROOTDIR, 'test/{}'.format(folder_num), file)
            verts, _ = pp3d.read_mesh(verts_filepath)
            break

    # pred indices with activation 1
    indcs = np.where(pred_pt == 1)[1]

    # for each landmark
    print('')
    for i in range(len(indcs)):

        # prediction coords
        coords_pred = verts[indcs[i]]

        # target coords by looking for activation 1 in each landmark channel
        ind = int(np.where(target[i] == 1)[0])
        point = int(target[i][ind][0])
        coords_target = verts[point]

        dist = eucl_dist(coords_target[0],coords_target[1],coords_target[2],
                        coords_pred[0],coords_pred[1],coords_pred[2])
        #print(str(i) + " " + str(dist)
        error_matrix[i, pred_idx] = dist
meanax0 = error_matrix.mean(axis=0)
meanax1 = error_matrix.mean(axis=1)
print(error_matrix.mean())

# write to txt
with open(os.path.join(ROOTDIR, 'error.txt'), 'w') as f:
    f.write('Total mean: {:.3f}'.format(error_matrix.mean()))
    f.write('\n\nSample mean:\n')
    f.write('\n'.join(['{} {:.3f}'.format(i, e) for i, e in enumerate(meanax0)]))
    f.write('\n\nLandmark mean:\n')
    f.write('\n'.join(['{} {:.3f}'.format(i, e) for i, e in enumerate(meanax1)]))