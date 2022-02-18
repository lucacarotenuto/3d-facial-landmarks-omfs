import numpy as np
import tqdm
import glob
import os
from pathlib import Path
import pickle


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def main():
    LDMKS = np.load('/Volumes/Extreme SSD/MscProject/headspace_pcl_all/ldmks.pkl',#'/Volumes/Extreme SSD/MscProject/headspace_pcl_all_face/ldmks.pkl',
                    allow_pickle=True)  # shape (samples, landmarks + 1, 3)
    ROOTDIR = '/Volumes/Extreme SSD/MscProject/headspace_pcl_all'
    RESULT_DIR = '/Volumes/Extreme SSD/MscProject/headspace_pcl_all_f9cm'
    # No_op has different folder structure than headspace and landmarks have to be saved manually
    NO_OP = False
    # Also transform landmarks for headspace
    TRANSFORM_LDMKS = True

    if NO_OP:
        p = os.path.join(ROOTDIR, '*.txt')
    else:
        p = os.path.join(ROOTDIR, '*', '*.txt')

    if TRANSFORM_LDMKS:
        # count files
        i = 0
        for filepath in glob.iglob(p):
            i += 1
        transf_ldmks_all = np.zeros((i, 69, 3))

    for n, filepath in enumerate(tqdm.tqdm(glob.iglob(p))):
        fname = Path(filepath).parts[-1]
        if not NO_OP:
            folder_num = Path(filepath).parts[-2]
        features = np.loadtxt(
            filepath,
            delimiter=',')
        pcl = features[:, :3]
        normals = features[:, 3:]

        if NO_OP:
            if Path(filepath).parts[-1] == '2_13_598.txt':
                exr = [-70.296661, 11.683527, 59.297222]
                exl = [14.411863, 5.917605, 63.373680]
                pg = [-30.880222, -74.321335, 38.278488]
                ns = [-29.507540, -1.344554, 80.940674]
            elif Path(filepath).parts[-1] == '695__1479.txt':
                exr = [-54.337875, 13.274704, 39.221195]
                exl = [42.285194, 15.905481, 42.271881]
                pg = [-18.148575, -93.151115, 52.592976]
                ns = [-10.912676, 18.351852, 70.476601]
        else:
            # identify landmark coordinates for file
            for i in range(LDMKS.shape[0]):
                if int(LDMKS[i, 0, 0]) == int(folder_num):
                    ldmks_idx = i
                    break
            ldmks_per_file = LDMKS[ldmks_idx, 1:, :]  # shape (landmarks, 3)
        
            exr = ldmks_per_file[36]  # vec origin
            exl = ldmks_per_file[45]  # vec target
            pg = ldmks_per_file[8]  # vec origin
            ns = ldmks_per_file[27]  # vec target

        # set nasion as origin
        pcl[:,0] -= ns[0]
        pcl[:,1] -= ns[1]
        pcl[:,2] -= ns[2]
        
        # align x with exr - exl
        R_X = rotation_matrix_from_vectors(np.array([exl[0] - exr[0], exl[1] - exr[1], exl[2] - exr[2]]),
                                        np.array([1, 0, 0]))
        pcl_transf = np.dot(pcl, R_X.T)
        normals_transf = np.dot(normals, R_X.T)

        # align z with pg - nasion
        R_Z = rotation_matrix_from_vectors(np.array([ns[0] - pg[0], ns[1] - pg[1], ns[2] - pg[2]]),
                                        np.array([0, 0, 1]))
        pcl_transf = np.dot(pcl_transf, R_Z.T)
        normals_transf = np.dot(normals_transf, R_Z.T)

        # remove points behind face (only keep y axis smaller 4cm)
        pcl_face = pcl_transf[pcl_transf[:,1] < 90]

        # do alignment and centering for landmarks
        if TRANSFORM_LDMKS:
            ldmks_transf = ldmks_per_file
            # center
            ldmks_transf[:,0] -= ns[0]
            ldmks_transf[:,1] -= ns[1]
            ldmks_transf[:,2] -= ns[2]
            # align
            ldmks_transf = np.dot(ldmks_transf, R_X.T)
            ldmks_transf = np.dot(ldmks_transf, R_Z.T)

            transf_ldmks_all[n, 0, :] = [int(folder_num), 0, 0]
            transf_ldmks_all[n, 1:, :] = ldmks_transf

        try:
            os.makedirs(os.path.join(RESULT_DIR, folder_num))
        except:
            print('folder could not be created')
        np.savetxt(os.path.join(RESULT_DIR, folder_num, fname), pcl_face, fmt='%10.5f', delimiter=',')
        #np.savetxt(os.path.join(RESULT_DIR, folder_num, fname), np.concatenate((pcl_face, normals_transf), axis=1), fmt='%10.5f', delimiter=',')

    #np.save(os.path.join(RESULT_DIR, 'ldmks.pkl'), transf_ldmks_all, allow_pickle=True)
    if TRANSFORM_LDMKS:
        with open(os.path.join(RESULT_DIR, 'ldmks.pkl'), 'wb') as f:
            pickle.dump(transf_ldmks_all, f)


if __name__ == '__main__':
    main()



