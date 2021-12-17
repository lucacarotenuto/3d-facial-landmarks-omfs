import math
import glob
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from utils import eucl_dist
import pickle


def dist_between_points(pointcl, idx_origin, idx_target):
    """

    Args:
        pointcl: point cloud represented as list of points with shape (points, 3)
        idx_origin: index of origin point
        idx_target: index of target point

    Returns:
        distance: distance between two points in a point cloud, where points are given as indices in the point cloud
    """
    origin = pointcl[idx_origin]
    target = pointcl[idx_target]
    distance = eucl_dist(origin[0], origin[1], origin[2], target[0], target[1], target[2])
    return distance


def closest_ldmk_dist(pcl, ldmks):
    """

    Args:
        pointcl: pointcloud shape (points, 3)
        ldmks: landmark point list

    Returns:
        closest_distance: closest distance between two landmark points in the entire point cloud
    """
    closest_distance = 999999
    for idx_ldmk_1 in range(len(ldmks)):
        for idx_ldmk_2 in range(len(ldmks)):
            if idx_ldmk_1 != idx_ldmk_2:
                dist = dist_between_points(pcl, idx_ldmk_1, idx_ldmk_2)
                if dist < closest_distance:
                    closest_distance = dist
    return closest_distance


def main():
    # ldmks = np.load('/Users/carotenuto/Master Radboud/MscProj/headspace_pcl_all/ldmks.pkl',
    #                 allow_pickle=True)  # shape (samples, landmarks + 1, 3)
    # rootdir = '/Users/carotenuto/Master Radboud/MscProj/headspace_pcl_testset_fullres/'
    ldmks = np.load('/Users/carotenuto/Master Radboud/MscProj/manual_labels/headspace0-180_ownlabels/ldmks.pkl',
                    allow_pickle=True)  # shape (samples, landmarks + 1, 3)
    rootdir = '/Users/carotenuto/Master Radboud/MscProj/manual_labels/headspace_pcl141_30k/'

    # per file
    #   per landmark
    #       take point in point cloud (origin) with smallest distance to landmark point (target)
    #   save points in file

    for filepath in tqdm(glob.iglob(rootdir + '*/13*.txt')):
        print(filepath)

        # process pointcloud file
        lines = open(filepath, 'r').read().split('\n')[:-1]
        pcl = [[float(l) for l in lines[j].split(',')[:3]] for j in range(len(lines))]

        # find folder num
        folder_num = Path(filepath).parts[-2]

        # identify landmark coordinates for file
        for i in range(ldmks.shape[0]):
            if int(ldmks[i, 0, 0]) == int(folder_num):
                ldmks_idx = i
                break
        ldmks_per_file = ldmks[ldmks_idx, 1:, :]  # shape (landmarks, 3)

        LANDMARK_INDICES = [8, 27, 30, 31, 33, 35, 36, 39, 42, 45, 60, 64]
        ldmks_per_file = ldmks_per_file[LANDMARK_INDICES]
        # get mean dist between points
        """
        a = len(pcl)
        dist_list = np.zeros((a))
        for j in tqdm(range(a)):
            shortest_dist = 99999999
            for i in range(len(pcl)):
                if j != i:
                    orig_coords = pcl[j]
                    target_coords= pcl[i]
                    dist = eucl_dist(orig_coords[0], orig_coords[1], orig_coords[2], target_coords[0], target_coords[1],
                                     target_coords[2])
                    if dist < shortest_dist:
                        shortest_dist = dist
            dist_list[j] = shortest_dist
        print(np.mean(dist_list))
        """

        print("closest landmark distance: ", closest_ldmk_dist(pcl, ldmks_per_file))

        point_list = np.zeros((len(ldmks_per_file)))

        # per landmark n (starting from 0)
        for n, target_coords in enumerate(ldmks_per_file):
            shortest_dist = 99999999

            for j in range(len(pcl)):
                # calc distance
                orig_coords = pcl[j]
                dist = eucl_dist(orig_coords[0], orig_coords[1], orig_coords[2], target_coords[0], target_coords[1],
                                 target_coords[2])
                if dist < shortest_dist:
                    shortest_dist = dist
                    pt_shortest_dist = j
            point_list[n] = pt_shortest_dist

            # print(folder_num, n, target_coords, pt_shortest_dist, shortest_dist)
        # print(np.unique(point_list), len(point_list), max(point_list), len(pcl))

        # go through each point and create activation depending on proximity to landmark point (heatmap)


        print(len(point_list))
        output_list = []
        for i, ldmk_point in enumerate(point_list):
            output = np.array([])
            for n, point in enumerate(pcl):
                dist = dist_between_points(pcl, n, int(ldmk_point))

                # calculate activation
                if ldmk_point == n:
                    activation = 1
                elif dist <= 3 and n != i:
                    activation = 0.75
                elif dist <= 4.5 and n != i:
                    activation = 0.5
                elif dist <= 6 and n != i:
                    activation = 0.25
                else:
                    continue

                # if array does not exist create new with else concatenate
                if output.size == 0:
                    output = np.array([[n, activation]])
                else:
                    output = np.append(output, np.array([[n, activation]]), axis=0)
            print(np.unique(output[:,1], return_counts = True))
            output_list.append(output)

        f = open(os.path.dirname(filepath) + '/hmap_per_class.pkl', 'wb')
        pickle.dump(output_list, f)


if __name__ == '__main__':
    main()