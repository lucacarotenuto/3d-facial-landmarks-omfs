# Remove non manifold geometry and reduce face count for headspace dataset
# Execute prep_hs_struc first to prepare headspace folder structure and remove objects without landmarks

import pymeshlab
import glob

rootdir = '/Users/carotenuto/Master Radboud/MscProj/headspace100_3k_cl/'

ms = pymeshlab.MeshSet()
for filepath in glob.iglob(rootdir + '*/*.obj'):
    print("loading " + filepath)
    ms.load_new_mesh(filepath)
    print("applying filters " + filepath)
    print("sqecd")
    ms.simplification_quadric_edge_collapse_decimation(targetfacenum=3000)
    """
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
    """
    ms.save_current_mesh(filepath, save_textures=False)

