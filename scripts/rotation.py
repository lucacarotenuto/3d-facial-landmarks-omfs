
import numpy as np
import pptk


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
    return rotation_cmatrix


def main():
    LDMKS = np.load('/Users/carotenuto/Downloads/ldmks_manual_196.pkl',
                    allow_pickle=True)  # shape (samples, landmarks + 1, 3)

    pcl = np.loadtxt(
        '/Users/carotenuto/Master Radboud/MscProj/manual_results/pcl_196_30k/test/00164/130929164827.txt',
        delimiter=',')
    pcl = pcl[:, :3]

    # identify landmark coordinates for file
    for i in range(LDMKS.shape[0]):
        if int(LDMKS[i, 0, 0]) == int('00164'):
            ldmks_idx = i
            break
    ldmks_per_file = LDMKS[ldmks_idx, 1:, :]  # shape (landmarks, 3)
    
    # align x with exr - exl
    exr = ldmks_per_file[36]  # vec origin
    exl = ldmks_per_file[45]  # vec target
    np.savetxt('orig00164.txt', pcl, fmt='%10.5f', delimiter=',')


    R = rotation_matrix_from_vectors(np.array([exl[0] - exr[0], exl[1] - exr[1], exl[2] - exr[2]]),
                                     np.array([1, 0, 0]))

    d = np.dot(pcl, R.T)
    np.savetxt('alignedx00164.txt', d, fmt='%10.5f', delimiter=',')
    v = pptk.viewer(d, show_axis=False)
    v.set(point_size=0.5, show_axis=True, show_info=True)

    # align z with pg - nasion
    pg = ldmks_per_file[8]  # vec origin
    ns = ldmks_per_file[27]  # vec target
    np.savetxt('orig00164.txt', pcl, fmt='%10.5f', delimiter=',')


    R = rotation_matrix_from_vectors(np.array([ns[0] - pg[0], ns[1] - pg[1], ns[2] - pg[2]]),
                                     np.array([0, 0, 1]))

    d = np.dot(d, R.T)
    np.savetxt('alignedz00164.txt', d, fmt='%10.5f', delimiter=',')

    # mirror
    e = d.copy()
    e[:, 0] = -e[:, 0]

    f = np.concatenate((e,d))

    v = pptk.viewer(f, show_axis=False)
    v.set(point_size=0.5, show_axis=True, show_info=True)

    print('end')

    np.savetxt('mirrored00164.txt', e, fmt='%10.5f', delimiter=',')

if __name__ == '__main__':
    main()



