# prepare point clouds

import pymeshlab
import glob
from pathlib import Path

rootdir = '/Users/carotenuto/Master Radboud/MscProj/headspace-orig copy/'

ms = pymeshlab.MeshSet()
for filepath in glob.iglob(rootdir + '*/*.obj'):
    print("loading " + filepath)
    ms.load_new_mesh(filepath)
    ms.texel_sampling(recovercolor=True)
    ms.point_cloud_simplification(samplenum=8000)
    #ms.save_current_mesh(filepath[:-4] + '_simplfd.obj', save_textures=False)
    print("saving " + filepath)
    ms.save_current_mesh(filepath[:-4] + '.xyz', save_textures=True)

    p = Path(filepath[:-4] + '.xyz')
    p.rename(p.with_suffix('.txt'))

    with open(filepath[:-4] + '.txt', mode='r+') as f:
        out = ""
        lines = f.readlines()
        for i in range(len(lines)):
            out += ",".join(lines[i].split(" "))[:-2] + "\n"
        x = open(filepath[:-4] + '_new.txt', "w+")
        x.write(out)
