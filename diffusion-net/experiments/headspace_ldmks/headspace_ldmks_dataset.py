import os
import sys
import numpy as np
import pandas as pd
import torch
import glob
import pickle
from torch.utils.data import Dataset
import potpourri3d as pp3d

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net

from pathlib import Path


class HeadspaceLdmksDataset(Dataset):
    def __init__(self, root_dir, train, data_format, num_landmarks, k_eig=128, use_cache=True, op_cache_dir=None):
        self.use_cache = use_cache
        self.train = train  # bool
        self.k_eig = k_eig
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, "cache", "train" if self.train else "test")
        self.op_cache_dir = op_cache_dir
        self.data_format = data_format
        self.num_landmarks = num_landmarks
        self.augment_mirror = False
        self.no_op = False

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.folder_num_list = []

        self.num_samples = 0

        mesh_files = []

        if not self.no_op:
            filepattern = '/*/13*.txt' if data_format == 'pcl' else '/*/13*.obj'
        else:
            filepattern = '/*.txt' if data_format == 'pcl' else '/*.obj'
        for filepath in glob.iglob(os.path.join(self.root_dir, 'train' if self.train else 'test') + filepattern):
            self.num_samples += 1
            mesh_files.append(filepath)
        print("loading {} meshes".format(len(mesh_files)))

        # Load the actual files
        for iFile in range(len(mesh_files)):
            print("loading mesh " + str(mesh_files[iFile]))
            if data_format == 'mesh':
                verts, faces = pp3d.read_mesh(mesh_files[iFile])
            else: # 'pcl'
                # todo: remove pandas dependency
                verts = pd.read_csv(mesh_files[iFile], sep=",", header=None)
                rgb = verts.to_numpy()[:, 3:]
                verts = verts.to_numpy()[:, :3]
                faces = np.array([])
            if not self.no_op:
                folder_num = Path(mesh_files[iFile]).parts[-2]
            else:
                folder_num = Path(mesh_files[iFile]).parts[-1]

            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))

            # center and unit scale
            verts = diffusion_net.geometry.normalize_positions(verts)

            self.folder_num_list.append(folder_num)

            # create sparse labels
            landmark_indices = {8,27,30,31,33,35,36,39,45,42,60,64} # indices start with 1
            #landmark_indices = {8,27,30,33,36,39,42,45,60,64} # indices start with 1
            #landmark_indices = {30, 39, 42, 60, 64}
            #landmark_indices = {30}

            #if :
            with open(os.path.join(self.root_dir, 'train' if self.train else 'test', folder_num,
                                'hmap_per_class.pkl'), 'rb') as fpath:
                labels_sparse = pickle.load(fpath)
            #labels_sparse = [item for pos, item in enumerate(labels_sparse) if pos in landmark_indices]
            labels = self.labels_from_sparse(verts, labels_sparse)
            #else:
            #    labels = np.array([])

            # if this file is not cached, populate
            if not os.path.isfile(os.path.join(self.cache_dir, '{}.pt'.format(folder_num))):
                # Precompute operators
                diffusion_net.utils.ensure_dir_exists(self.cache_dir)
                frames, massvec, L, evals, evecs, gradX, gradY = diffusion_net.geometry.populate_cache(
                    verts, faces, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)
                torch.save((verts, faces, rgb, labels, frames, massvec, L,
                            evals, evecs, gradX, gradY), os.path.join(self.cache_dir, folder_num + ".pt"))
            if self.train:
                if self.augment_mirror:
                    self.num_samples += 1

                    # create mirrored pcl and mirrored landmark heatmaps
                    verts = diffusion_net.geometry.mirror(verts, labels)


                

    def __len__(self):
        return self.num_samples


    def __getitem__(self, idx):
        folder_num = self.folder_num_list[idx]
        path_cache = os.path.join(self.cache_dir, folder_num)
        verts, faces, rgb, labels, frames, massvec, L, evals, evecs, gradX, gradY = torch.load(path_cache + ".pt")

        return verts, faces, rgb, frames, massvec, L, evals, evecs, gradX, gradY, labels, folder_num


    def labels_from_sparse(self, verts, labels_sparse):
        # create labels from sparse representation
        labels = torch.zeros((self.num_landmarks, len(verts)))
        for j in range(len(labels_sparse)):
            for k in range(len(labels_sparse[j])):
                pos = labels_sparse[j][k, 0]
                act = labels_sparse[j][k, 1]
                labels[j, int(pos)] = act
        return labels

