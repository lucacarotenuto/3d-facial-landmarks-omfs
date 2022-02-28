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