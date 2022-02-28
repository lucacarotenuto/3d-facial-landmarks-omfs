# Remove non manifold geometry

import pymeshlab
import glob
import os

rootdir = '/Volumes/T7/headspace-full/headspace/headspaceOnline/subjects/'
target_dir = '/Users/carotenuto/Master Radboud/MscProj/headspace_mesh_all_5kf/'

ms = pymeshlab.MeshSet()
for filepath in glob.iglob(rootdir + '*/*.obj'):
    folder_num = os.path.basename(os.path.dirname(filepath))
    filename = os.path.basename(filepath)

    # skip file if landmark does not exist
    if not os.path.exists(os.path.join(rootdir, folder_num, 'ldmks{}.txt'.format(os.path.splitext(filename)[0]))):
        continue

    print("loading " + filepath)
    ms.load_new_mesh(filepath)
    print("applying filters " + filepath)
    print("sqecd")
    ms.simplification_quadric_edge_collapse_decimation(targetfacenum=50000)
    print("remove")
    ms.remove_isolated_pieces_wrt_face_num(mincomponentsize=1000)
    print("repair")
    ms.repair_non_manifold_edges_by_removing_faces()

    ms.repair_non_manifold_edges_by_splitting_vertices()
    ms.repair_non_manifold_vertices_by_splitting()
    print("select")
    ms.select_non_manifold_vertices()
    print("delete")
    ms.delete_selected_vertices()
    print("close")
    ms.close_holes(maxholesize=3)
    #ms.save_current_mesh(filepath[:-4] + '_simplfd.obj', save_textures=False)
    print("saving " + filepath)

    if not os.path.exists(os.path.join(target_dir, filename)):
        os.makedirs(os.path.join(target_dir, folder_num))
    ms.save_current_mesh(os.path.join(target_dir, folder_num, filename), save_textures=False)

    # free up memory
    ms.clear()

    # remove .mtl file (for some reason meshlab still creates this despite save_textures=False)
    os.remove(os.path.join(target_dir, folder_num, '{}.mtl'.format(filename)))
