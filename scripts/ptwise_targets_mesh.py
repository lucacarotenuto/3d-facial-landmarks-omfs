# Create point-wise targets for meshes

import math
import glob
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pickle

ldmks = np.load('/Users/carotenuto/Master Radboud/MscProj/headspace_pcl_all/ldmks.pkl',
                allow_pickle=True)  # shape (samples, landmarks + 1, 3)

rootdir = '/Users/carotenuto/Master Radboud/MscProj/headspace_mesh_all_5kf/'
target_dir = '/Users/carotenuto/Master Radboud/MscProj/headspace_mesh_all_5kf/' # copies obj and hmap_per_class to target dir

def eucl_dist(orig_x, orig_y, orig_z, target_x, target_y, target_z):
    """

    Returns:
        distance: distance between origin and target in 3 dimensional cartesian coordinate system
    """
    distance = math.sqrt(
        math.pow(target_x - orig_x, 2) + math.pow(target_y - orig_y, 2) + math.pow(target_z - orig_z, 2))
    return distance


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


def obj_data_to_mesh3d(odata):
    # odata is the string read from an obj file
    vertices = []
    faces = []
    lines = odata.splitlines()

    for line in lines:
        slist = line.split()
        if slist:
            if slist[0] == 'v':
                vertex = np.array(slist[1:], dtype=float)
                vertices.append(vertex)
            elif slist[0] == 'f':
                face = []
                for k in range(1, len(slist)):
                    face.append([int(s) for s in slist[k].replace('//', '/').split('/')])
                if len(face) > 3:  # triangulate the n-polyonal face, n>3
                    faces.extend(
                        [[face[0][0] - 1, face[k][0] - 1, face[k + 1][0] - 1] for k in range(1, len(face) - 1)])
                else:
                    faces.append([face[j][0] - 1 for j in range(len(face))])
            else:
                pass

    return np.array(vertices), np.array(faces)

# per file
#   per landmark
#       take point in point cloud (origin) with smallest distance to landmark point (target)
#   save points in file

# per file
for filepath in tqdm(glob.iglob(rootdir + '*/13*.obj')):
    print(filepath)

    with open(filepath, 'r') as file:
        obj_data = file.read()

    vertices, faces = obj_data_to_mesh3d(obj_data)
    pcl = vertices
    # find folder num
    folder_num = Path(filepath).parts[-2]

    # identify landmark coordinates for file
    for i in range(ldmks.shape[0]):
        if int(ldmks[i, 0, 0]) == int(folder_num):
            ldmks_idx = i
            break
    ldmks_per_file = ldmks[ldmks_idx, 1:, :]  # shape (landmarks, 3)

    print("closest landmark distance: ", closest_ldmk_dist(pcl, ldmks_per_file))

    point_list = np.zeros((68))
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

    point_list_ints = [int(l) for l in list(point_list)]

    # go through each point and create activation depending on proximity to landmark point (heatmap)
    #print(len(point_list))
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
        #print(np.unique(output[:,1], return_counts = True))
        output_list.append(output)

    directory = os.path.join(target_dir, folder_num)
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = open(os.path.join(target_dir, folder_num, 'hmap_per_class.pkl'), 'wb')
    pickle.dump(output_list, f)