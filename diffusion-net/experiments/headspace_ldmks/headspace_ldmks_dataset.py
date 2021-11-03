import shutil
import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import potpourri3d as pp3d

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from diffusion_net.utils import toNP

import pickle
from pathlib import Path
import glob

class HeadspaceLdmksDataset(Dataset):
    def __init__(self, root_dir, train, data_format, k_eig=128, use_cache=True, op_cache_dir=None):
        self.use_cache = use_cache
        self.train = train  # bool
        self.k_eig = k_eig
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, "cache", "train" if self.train else "test")
        self.op_cache_dir = op_cache_dir
        self.n_class = 3 * 68
        self.data_format = data_format

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.folder_num_list = []

        self.num_samples = 0

        mesh_files = []


        filepattern = '/*/13*.txt' if data_format.equals('pcl') else '/*/13*.obj'
        for filepath in glob.iglob(os.path.join(self.root_dir, 'train' if self.train else 'test') + filepattern):
            self.num_samples += 1
            mesh_files.append(filepath)
        print("loading {} meshes".format(len(mesh_files)))

        # Load the actual files
        for iFile in range(len(mesh_files)):
            print("loading mesh " + str(mesh_files[iFile]))
            if data_format.equals('mesh'):
                verts, faces = pp3d.read_mesh(mesh_files[iFile])
            else: # 'pcl'
                verts = pd.read_csv(mesh_files[iFile], sep=",", header=None)
                verts = verts.to_numpy()[:, :3]
            faces = np.array([])
            folder_num = Path(mesh_files[iFile]).parts[-2]

            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))

            # center and unit scale
            verts = diffusion_net.geometry.normalize_positions(verts)

            self.folder_num_list.append(folder_num)

            # if this file is not cached, populate
            if not os.path.isfile(os.path.join(self.cache_dir, folder_num, '.pt')):
                # Precompute operators
                diffusion_net.utils.ensure_dir_exists(self.cache_dir)
                frames, massvec, L, evals, evecs, gradX, gradY = diffusion_net.geometry.populate_cache(
                    verts, faces, self.cache_dir, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)
                torch.save((verts, faces, frames, massvec, L,
                            evals, evecs, gradX, gradY), os.path.join(self.cache_dir, folder_num + ".pt"))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        folder_num = self.folder_num_list[idx]
        path_cache = os.path.join(self.cache_dir, folder_num)
        verts, faces, frames, massvec, L, evals, evecs, gradX, gradY = torch.load(path_cache + ".pt")

        # create sparse labels
        with open(os.path.join(self.root_dir, 'train' if self.train else 'test', folder_num, \
                               'hmap_per_class.pkl'), 'rb') as fpath:
            labels = pickle.load(fpath)
        sparse_labels = self.createSparseLabels(verts, labels)

        return verts, faces, frames, massvec, L, evals, evecs, gradX, gradY, sparse_labels, folder_num

    def createSparseLabels(self, verts, labels):
        # create labels sparse list from labels list
        labels_sparse = torch.zeros((68, len(verts)))
        for j in range(len(labels)):
            for k in range(len(labels[j])):
                pos = labels[j][k, 0]
                act = labels[j][k, 1]
                labels_sparse[j, int(pos)] = act
        return labels_sparse