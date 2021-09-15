# Prepare landmarks for headspace dataset
# execute prep_hs_struc.ipynb first and use output as rootdir

import os
import glob
import pymeshlab
from pathlib import Path
import csv
import numpy as np
import pickle

rootdir = '/Users/carotenuto/Master Radboud/MscProj/headspace10_hs_struc/'

ldmks = []
for filepath in glob.iglob(rootdir + '*/*/ldmks*.txt'):
    lines = open(filepath, 'r').read().split('\n')[:68]
    ldmks_per_file = [Path(filepath).parts[-3]]
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(os.path.dirname(filepath) + '/' + os.path.basename(filepath)[5:-4] + '.obj')
    for l in lines:
        coord = ms.current_mesh().vertex_matrix()[int(l)-1]
        ldmks_per_file.append([coord[0], coord[1], coord[2]])
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
"""with open(rootdir + 'ldmk_coords.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for row in ldmks:
        wr.writerow(row)"""