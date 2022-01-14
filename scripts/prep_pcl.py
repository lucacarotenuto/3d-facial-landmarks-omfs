# prepare point clouds

import pymeshlab
import glob
from pathlib import Path
import os
from tqdm import tqdm


def main():
    # Rootdir that contains the full resolution meshes, only meshes with manual labelled .csv files are prepared
    ROOTDIR = '/Volumes/Extreme SSD/MscProject/no-op/no-op_2'
    # New folder where simplified point cloud should be saved
    NEW_FOLDER = '/Volumes/Extreme SSD/MscProject/no-op/no-op_2/pcl2'
    NO_OP = True
    j= 0
    ms = pymeshlab.MeshSet()
    if not NO_OP:
        p = os.path.join(ROOTDIR, '*', '*.obj')
    else:
        p = os.path.join(ROOTDIR, '*.obj')
    for filepath in tqdm(glob.iglob(p)):
        for File in os.listdir(os.path.dirname(filepath)):
            # use this for headspace landmarks
            #if all([File.endswith(".txt") and File.startswith("ldmks")]) and os.path.exists(filepath[:-4] + '.bmp'):
            # use this for manual landmarks
            #if all([File.endswith(".csv") and File.startswith("ldmks")]) and os.path.exists(filepath[:-4] + '.bmp'):
                j += 1
                print("loading " + filepath)
                print(str(j))
                ms.load_new_mesh(filepath)
                try:
                    ms.texel_sampling(recovercolor=True)
                except:
                    pass
                ms.point_cloud_simplification(samplenum=60000)
                #ms.save_current_mesh(filepath[:-4] + '_simplfd.obj', save_textures=False)
                print("saving " + filepath)
                ms.save_current_mesh(filepath[:-4] + '.obj', save_textures=True)

                with open(filepath[:-4] + '.obj', mode='r+') as f:
                    verts = ""
                    lines = f.readlines()
                    for i in range(len(lines)):
                        if lines[i].startswith('v '):
                            verts += lines[i][2:]
                    verts = verts.replace(' ', ',')
                    folder_num = Path(filepath).parts[-2]
                    new_filepath = os.path.join(NEW_FOLDER, folder_num, Path(filepath).parts[-1][:-3] + 'txt')

                    os.makedirs(os.path.dirname(new_filepath), exist_ok=True)
                    x = open(new_filepath, "w+")
                    x.write(verts)

                # p = Path(filepath[:-4] + '.xyz')
                #
                # if os.path.exists(p.with_suffix(('.txt'))):
                #     os.remove(p.with_suffix('.txt'))
                #
                # p.rename(p.with_suffix('.txt'))
                # folder_num = Path(filepath).parts[-2]
                #
                # #Path(NEW_FOLDER + folder_num +'\\').mkdir(parents=True, exist_ok=True)
                # new_filepath = os.path.join(NEW_FOLDER, folder_num, Path(filepath).parts[-1][:-3] + 'txt')
                # with open(filepath[:-4] + '.txt', mode='r+') as f:
                #     out = ""
                #     lines = f.readlines()
                #     for i in range(len(lines)):
                #         out += ",".join(lines[i].split(" "))[:-2] + "\n"
                #     os.makedirs(os.path.dirname(new_filepath), exist_ok=True)
                #     x = open(new_filepath, "w+")
                #     x.write(out)
                ms.clear() # frees up from memory


if __name__ == '__main__':
    main()