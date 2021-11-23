import numpy as np
import math

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

#WRONG
def sparse_to_orig(labels):
    labels_sparse = np.zeros((labels.shape[0], len(labels)))
    for j in range(len(labels)):
        for k in range(len(labels[j])):
            pos = labels[j][k, 0]
            if POINT_PREDS:
                act = 0 if labels[j][k, 1] < 1 else 1
            else:
                act = labels[j][k, 1]
            labels_sparse[j, int(pos)] = act
    return labels_sparse

#WRONG
def to_sparse(ldmks):
    ndarr = np.zeros((len(ldmks),69,3))
    for i in range(len(ldmks)):
        ndarr[i,0,0] = ldmks[i][0]
        for j in range(1,69):
            ndarr[i,j,0] = ldmks[i][j][0]
            ndarr[i, j, 1] = ldmks[i][j][1]
            ndarr[i, j, 2] = ldmks[i][j][2]