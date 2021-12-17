# prepare point clouds

import pymeshlab
import glob
from pathlib import Path
import os
from tqdm import tqdm

#rootdir = '/Users/carotenuto/Master Radboud/MscProj/subjects_1-150 copy/'
rootdir = '/Users/carotenuto/Master Radboud/MscProj/manual_labels/headspace141-180_ownlabels/'
new_folder = '/Users/carotenuto/Master Radboud/MscProj/headspace_pcl141-180_30k/'
j= 0
ms = pymeshlab.MeshSet()
for filepath in tqdm(glob.iglob(os.path.join(rootdir, '*/*.obj'))):
    for File in os.listdir(os.path.dirname(filepath)):
        # if all([File.endswith(".txt") and File.startswith("ldmks")]):
        if all([File.endswith(".csv") and File.startswith("ldmks")]):
            j += 1
            print("loading " + filepath)
            print(str(j))
            ms.load_new_mesh(filepath)
            ms.texel_sampling(recovercolor=True)
            ms.point_cloud_simplification(samplenum=30000)
            #ms.save_current_mesh(filepath[:-4] + '_simplfd.obj', save_textures=False)
            print("saving " + filepath)
            ms.save_current_mesh(filepath[:-4] + '.xyz', save_textures=True)

            p = Path(filepath[:-4] + '.xyz')
            p.rename(p.with_suffix('.txt'))
            folder_num = Path(filepath).parts[-2]

            Path(new_folder + folder_num +'/').mkdir(parents=True, exist_ok=True)
            new_filepath = new_folder + folder_num +'/' + Path(filepath).parts[-1][:-3] + 'txt'
            with open(filepath[:-4] + '.txt', mode='r+') as f:
                out = ""
                lines = f.readlines()
                for i in range(len(lines)):
                    out += ",".join(lines[i].split(" "))[:-2] + "\n"
                os.makedirs(os.path.dirname(new_filepath), exist_ok=True)
                x = open(new_filepath, "w+")
                x.write(out)
            ms.clear() # frees up from memory
