# Compute the distance of each predicted landmark to their corresponding target

import pickle
import glob
import os
import numpy as np
import potpourri3d as pp3d
from tqdm import tqdm
from utils import eucl_dist
from sympy import Plane, Point3D


def main():
    ROOTDIR = '/Users/carotenuto/Documents/GitHub/3d-facial-landmarks-omfs/diffusion-net/experiments/refine_ldmks/refined_500_mult_25_3'
    LANDMARK_INDICES = [8, 27, 30, 31, 33, 35, 36, 39, 42, 45, 60, 64]  # e.g. nosetip 31 has index 30
    #LANDMARK_INDICES = [8, 27, 30, 33, 36, 39, 42, 45, 60, 64]
    #LANDMARK_INDICES = [8, 27, 30, 33, 36, 39, 42, 45, 60, 64]
    IS_REFINED = True # are predictions refined?
    #LDMKS = np.load('C:\\Users\\Luca\\Documents\\headspace_pcl_all\\ldmks.pkl',
    #                allow_pickle=True)  # shape (samples, landmarks + 1, 3)
    LDMKS = np.load('/Users/carotenuto/Master Radboud/MscProj/manual_results/subjects_196_labelled/ldmks.pkl',
                    allow_pickle=True)  # shape (samples, landmarks + 1, 3)
    REMOVE_SYMMETRICAL_PREDS = False

    # determine total prediction number and num landmarks
    total_preds = 0
    for path in glob.iglob(os.path.join(ROOTDIR, 'preds', 'hmap_per_class*.pkl')):
        total_preds += 1
    with open(path, 'rb') as f:
        pred = pickle.load(f)

    if not IS_REFINED:
        num_ldmks = pred.shape[1]
    else:
        num_ldmks = 12   
    #num_ldmks = 10 # define landmarks manually if predictions include more landmarks

    if not IS_REFINED:
        # error matrix (ldmks, preds)
        error_matrix = np.zeros((num_ldmks, total_preds))
    else:
        error_matrix = np.zeros((num_ldmks, int(int(total_preds)/num_ldmks)))

    arr_folder_nums = []
    # for each prediction
    for pred_idx, path in tqdm(enumerate(glob.iglob(os.path.join(ROOTDIR, 'preds', 'hmap_per_class*.pkl')))):
        with open(path, 'rb') as f:
            pred = pickle.load(f)
        # only keep selected landmarks e.g. if predictions include more landmarks than necessary
        #ldmk_indcs = [8, 27, 30, 33, 36, 39, 42, 45, 60, 64]
        #pred = pred[:, ldmk_indcs]

        if IS_REFINED:
            # get the num part e.g. '00004'
            folder_num = os.path.basename(path).split('hmap_per_class')[1][:5]
            folder_num_ldmk = os.path.basename(path).split('hmap_per_class')[1].split('.')[0].split('_')[-1]
        else:
            folder_num = os.path.basename(path).split('.')[0][-5:]
        if folder_num not in arr_folder_nums:
            arr_folder_nums.append(folder_num)


        # load target
        if IS_REFINED:
            # choose target by proximation of closest point
            #path = os.path.join('test', folder_num + '_0', folder_num_ldmk, 'hmap_per_class.pkl')
            #with open(os.path.join(ROOTDIR, path), 'rb') as f:
            #    target = pickle.load(f)

            # take actual landmark coordinate target
            # identify landmark coordinates for file
            for i in range(LDMKS.shape[0]):
                if int(LDMKS[i, 0, 0]) == int(folder_num):
                    ldmks_idx = i
                    break
            ldmks_per_file = LDMKS[ldmks_idx, 1:, :]  # shape (landmarks, 3)
            target = ldmks_per_file
        else:
            # choose target by proximation of closest point
            #path = os.path.join('test',folder_num, 'hmap_per_class.pkl')
            #with open(os.path.join(ROOTDIR, path), 'rb') as f:
            #    target = pickle.load(f)

            # take actual landmark coordinate target
            for i in range(LDMKS.shape[0]):
                if int(LDMKS[i, 0, 0]) == int(folder_num):
                    ldmks_idx = i
                    break
            ldmks_per_file = LDMKS[ldmks_idx, 1:, :]  # shape (landmarks, 3)
            target = ldmks_per_file
        # only keep selected landmarks
        target = [item for pos, item in enumerate(target) if pos in LANDMARK_INDICES]

        # get vertex xyz coordinate
        for file in os.listdir(os.path.join(ROOTDIR, 'test', folder_num + '', folder_num_ldmk if IS_REFINED else '')):
            # .txt extension for pcl
            if file.endswith('.txt'):
                verts_filepath = os.path.join(ROOTDIR, 'test', folder_num + '', folder_num_ldmk if IS_REFINED else '', file)

                # todo make helper function to load .txt into pcl array
                lines = open(verts_filepath, 'r').read().split('\n')[:-1]
                verts = [[float(l) for l in lines[j].split(',')[:3]] for j in range(len(lines))]
                break

            elif file.endswith('.obj'): # .obj extension for mesh
                verts_filepath = os.path.join(ROOTDIR, 'test/{}'.format(folder_num), file)
                verts, _ = pp3d.read_mesh(verts_filepath)
                break

        pred = np.transpose(pred)
        
        if REMOVE_SYMMETRICAL_PREDS:
            # get nasion, pronasale and subnasale coordinates
            ns = verts[np.argmax(pred[1,:])]
            prn = verts[np.argmax(pred[2,:])]
            sn = verts[np.argmax(pred[3,:])]

            # create a plane
            plane = Plane(Point3D(ns), Point3D(prn), Point3D(sn))

            # as computation is expensive, sort activations descending for each channel, if point found at the right side, set activation 1.0 and leave loop
            right = [3, 6, 7, 10] if len(LANDMARK_INDICES) == 12 else [4, 5, 8]
            left = [5, 8, 9, 11] if len(LANDMARK_INDICES) == 12 else [6, 7, 9]
            sorted_indices_r = [np.argsort(pred[x,:])[::-1] for x in right]
            sorted_indices_l = [np.argsort(pred[x,:])[::-1] for x in left]
            # check on which side of the plane prediction points are by checking if plane equation is:
            #   smaller 0, then point is at right side of mid-face plane (subjects perspective)
            #   bigger 0, then point is at left side of mid-face plane (subjects perspective)
            for i, channel in enumerate(sorted_indices_r):
                for j, ind in enumerate(channel):
                    if plane.equation(verts[ind][0], verts[ind][1], verts[ind][2]) < 0:
                        pred[right[i], ind] = 1.0
                        print(i, j)
                        break
                    if j == 80:
                        break
            for i, channel in enumerate(sorted_indices_l):
                for j, ind in enumerate(channel):
                    if plane.equation(verts[ind][0], verts[ind][1], verts[ind][2]) > 0:
                        pred[left[i], ind] = 1.0
                        print(i, j)
                        break
                    if j == 80:
                        break

        # make point activations (pred is non-sparse representation (ldmks, verts))
        pred_pt = np.zeros_like(pred)
        pred_pt[np.arange(len(pred)), pred.argmax(1)] = 1

        # pred indices with activation 1
        indcs = np.where(pred_pt == 1)[1]

        # for each landmark
        print('')
        for i in range(len(indcs)):

            # prediction coords
            coords_pred = verts[indcs[i]]

            # target coords by looking for activation 1 in each landmark channel
            if not IS_REFINED:
                #ind = int(np.where(target[i] == 1)[0])
                #point = int(target[i][ind][0])
                #coords_target = verts[point]
                coords_target = target[i]
            else:
                coords_target = ldmks_per_file[LANDMARK_INDICES[int(folder_num_ldmk)]]

            dist = eucl_dist(coords_target[0],coords_target[1],coords_target[2],
                            coords_pred[0],coords_pred[1],coords_pred[2])
            #print(str(i) + " " + str(dist)
            if not IS_REFINED:
                error_matrix[i, pred_idx] = dist
            else:
                error_matrix[int(folder_num_ldmk),arr_folder_nums.index(folder_num)] = dist
    meanax0 = error_matrix.mean(axis=0)
    meanax1 = error_matrix.mean(axis=1)
    print(error_matrix.mean())
    stdax1 = error_matrix.std(axis=1)
    maxax1 = error_matrix.max(axis=1)
    print(error_matrix[error_matrix == 0.0])


    # write to txt
    with open(os.path.join(ROOTDIR, 'error.txt'), 'w') as f:
        f.write('Total mean: {:.3f}'.format(error_matrix.mean()))
        f.write('\n\nSample mean:\n')
        f.write('\n'.join(['{} {:.3f}'.format(arr_folder_nums[i], e) for i, e in enumerate(meanax0)]))
        f.write('\n\nLandmark mean:\n')
        f.write('\n'.join(['{} {:.3f}'.format(i, e) for i, e in enumerate(meanax1)]))
    with open(os.path.join(ROOTDIR, 'error_std.txt'), 'w') as f:
        f.write('Total std: {:.3f}'.format(error_matrix.std()))
        f.write('\n\nLandmark std:\n')
        f.write('\n'.join(['{} {:.3f}'.format(i, e) for i, e in enumerate(stdax1)]))
    with open(os.path.join(ROOTDIR, 'error_max.txt'), 'w') as f:
        f.write('Total max: {:.3f}'.format(error_matrix.max()))
        f.write('\n\nLandmark max:\n')
        f.write('\n'.join(['{} {:.3f}'.format(i, e) for i, e in enumerate(maxax1)]))
    np.savetxt(os.path.join(ROOTDIR, 'errormatrix.txt'), error_matrix, fmt='%.3f')

if __name__ == "__main__":
    main()