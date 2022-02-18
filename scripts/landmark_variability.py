import numpy as np
import pptk
import potpourri3d as pp3d
import csv
import glob
import os
from utils import eucl_dist
from pathlib import Path

# Calculate variability between Headspace labels and manual labels

rootdir = '/Volumes/Extreme SSD/MscProject/annotations_manual/annotations_luc_har_raw'
distances = np.array([])
for filepath in glob.iglob(os.path.join(rootdir, '*', '*.obj')):
    #verts, _ = pp3d.read_mesh(filepath)
    folder_num = Path(filepath).parts[-2]
    if os.path.exists(os.path.join(*Path(filepath).parts[:-1], 'ldmks.csv')):
        if len(glob.glob(os.path.join(*Path(filepath).parts[:-1], 'ldmks*.txt'))) == 1:
            verts, _ = pp3d.read_mesh(filepath)
            rows = []
            with open(os.path.join(rootdir, folder_num, 'ldmks.csv'), 'r') as f:
                csvreader = csv.reader(f)
                for row in csvreader:
                    rows.append(row)

            ldmks = rows[7:19]
            ldmks_own = [row[1:4] for row in ldmks]

            # only keep selected landmarks
            LANDMARK_INDICES = [8, 27, 30, 33, 36, 39, 42, 45, 60, 64, 31, 35]
            for filepath in glob.iglob(os.path.join(rootdir, folder_num, 'ldmks13*.txt')):
                ldmks_hs = np.loadtxt(filepath)
            ldmks_hs = ldmks_hs.astype(int)

            ldmks_hs = ldmks_hs[LANDMARK_INDICES]

            sample_distances = np.zeros((len(ldmks_hs)))
            for i, ldmk_idx in enumerate(ldmks_hs):
                coords_hs_ldmk = verts[ldmk_idx-1]
                dist = eucl_dist(coords_hs_ldmk[0], coords_hs_ldmk[1], coords_hs_ldmk[2], float(ldmks_own[i][0]),
                                 float(ldmks_own[i][1]), float(ldmks_own[i][2]))
                sample_distances[i] = dist
            sample_dist_mean = np.mean(sample_distances)
            print(str(sample_dist_mean))
            distances = np.append(distances, sample_dist_mean)
print("TOTAL: {}".format(np.mean(distances)))