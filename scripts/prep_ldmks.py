# Prepare landmarks file ldmks.pkl that summarizes 68 landmarks in coordinate form from headspace labels or
# alternatively from manual labelling performed by custom analysis in 3DMedX

import os
import glob
import numpy as np
import potpourri3d as pp3d
import csv
import pickle
from tqdm import tqdm
from pathlib import Path

def main():
    # Rootdir of headspace dataset
    # rootdir = '/Users/carotenuto/Master Radboud/MscProj/subjects_1-150/'
    ROOTDIR = '/Volumes/Extreme SSD/MscProject/no-op/no_op_manual_labels'
    # Set True if manual labels ldmks.csv created by 3DMedX, False if headspace labels landmarks13*.txt
    MANUAL_LABELS = True

    searchpath = os.path.join('*', 'ldmks.csv' if MANUAL_LABELS else 'ldmks*.txt')
    ldmks = []
    for filepath in tqdm(glob.iglob(os.path.join(ROOTDIR, searchpath))):
        ldmks_per_file = [Path(filepath).parts[-2]]
        folder_number = ldmks_per_file[0]

        if MANUAL_LABELS:
            # load ldmks.csv from own labels
            lines = []
            with open(os.path.join(ROOTDIR, folder_number, 'ldmks.csv'), 'r') as f:
                csvreader = csv.reader(f)
                for row in csvreader:
                    lines.append(row)
            lines = lines[7:19]
            lines = [row[1:4] for row in lines]
        else:
            # load headspace labels ldmks*.txt
            lines = open(filepath, 'r').read().split('\n')[:68]

        for obj_fpath in glob.iglob(os.path.join(os.path.dirname(filepath), '*.obj')):
            verts, _ = pp3d.read_mesh(obj_fpath)

        if MANUAL_LABELS:
            # always create 68 landmark fields (optionally empty to ensure compatability)
            LANDMARK_INDICES = [8, 27, 30, 33, 36, 39, 42, 45, 60, 64, 31, 35]

            for i in range(68):
                if i in LANDMARK_INDICES:
                    coords = lines[LANDMARK_INDICES.index(i)]
                else:
                    coords = [0, 0, 0]
                ldmks_per_file.append([coords[0], coords[1], coords[2]])
        else:
            for l in lines:
                coords = verts[int(l)-1]
                ldmks_per_file.append([coords[0], coords[1], coords[2]])

        ldmks.append(ldmks_per_file)

    # make list to numpy array to save pickle file
    ndarr = np.zeros((len(ldmks), 69, 3))
    for i in range(len(ldmks)):
        ndarr[i, 0, 0] = ldmks[i][0]
        for j in range(1, 69):
            ndarr[i, j, 0] = ldmks[i][j][0]
            ndarr[i, j, 1] = ldmks[i][j][1]
            ndarr[i, j, 2] = ldmks[i][j][2]
    with open(os.path.join(ROOTDIR, 'ldmks.pkl'), 'wb') as f:
        pickle.dump(ndarr, f)


if __name__ == "__main__":
    main()
