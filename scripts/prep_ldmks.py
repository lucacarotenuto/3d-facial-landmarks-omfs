# Prepare landmarks for headspace dataset

import os
import glob
import pymeshlab
from pathlib import Path
import numpy as np
import pickle

# Rootdir of headspace dataset
#rootdir = '/Users/carotenuto/Master Radboud/MscProj/subjects_1-150/'
#rootdir = '/Users/carotenuto/Master Radboud/MscProj/headspace100_hs_struc'

rootdir = '/Volumes/T7/headspace-full/headspace/headspaceOnline/subjects/'
# Set true if rootdir has prepared headspace structure with train/test folders, false if original headspace folder struc
has_prep_struc = False

searchpath = '*/*/ldmks*.txt' if has_prep_struc else '*/ldmks*.txt'
ldmks = []
for filepath in glob.iglob(rootdir + searchpath):
    lines = open(filepath, 'r').read().split('\n')[:68]
    ldmks_per_file = [Path(filepath).parts[-3]] if has_prep_struc else [Path(filepath).parts[-2]]
    ms = pymeshlab.MeshSet()
    for obj_fpath in glob.iglob(os.path.dirname(filepath) + '/*.obj'):
        ms.load_new_mesh(obj_fpath)
        for l in lines:
            coord = ms.current_mesh().vertex_matrix()[int(l)-1]
            ldmks_per_file.append([coord[0], coord[1], coord[2]])
        ldmks.append(ldmks_per_file)

ndarr = np.zeros((len(ldmks),69,3))
for i in range(len(ldmks)):
    ndarr[i,0,0] = ldmks[i][0]
    for j in range(1,69):
        ndarr[i,j,0] = ldmks[i][j][0]
        ndarr[i, j, 1] = ldmks[i][j][1]
        ndarr[i, j, 2] = ldmks[i][j][2]
with open(rootdir + 'ldmks.pkl', 'wb') as f:
    pickle.dump(ndarr, f)