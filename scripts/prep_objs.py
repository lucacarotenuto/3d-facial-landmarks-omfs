# Remove non manifold geometry and reduce face count for headspace dataset
# Execute prep_hs_struc first to prepare headspace folder structure and remove objects without landmarks

import pymeshlab
import glob

rootdir = '/Users/carotenuto/Master Radboud/MscProj/headspace10_cleaned_/'

for filepath in glob.iglob(rootdir + '*/*/*.obj'):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(filepath)
    ms.simplification_quadric_edge_collapse_decimation(targetfacenum=8000)
    ms.remove_isolated_pieces_wrt_face_num(mincomponentsize=1000)
    #ms.repair_non_manifold_edges_by_removing_faces()
    #ms.save_current_mesh(filepath[:-4] + '_simplfd.obj', save_textures=False)
    ms.save_current_mesh(filepath, save_textures=False)