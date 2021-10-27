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
    """Human segmentation dataset from Maron et al (not the remeshed version from subsequent work)"""

    def __init__(self, root_dir, train, k_eig=128, use_cache=True, op_cache_dir=None):

        self.train = train  # bool
        self.k_eig = k_eig
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir
        self.n_class = 3 * 68

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.labels_list = []  # per-face labels!!
        self.folder_num_list = []

        # self.labels = np.load(self.root_dir + '/ldmks.pkl', allow_pickle=True)

        # check the cache
        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            load_cache = train_cache if self.train else test_cache
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                self.verts_list, self.faces_list, self.frames_list, self.massvec_list, self.L_list, self.evals_list, \
                self.evecs_list, self.gradX_list, self.gradY_list, self.labels_list, self.folder_num_list = torch.load(
                    load_cache)

                # create labels sparse list from labels list
                self.createSparseLabels()
                return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes & labels

        # Get all the files
        mesh_files = []
        label_files = []

        # Train test split
        if self.train:
            for filepath in glob.iglob(self.root_dir + '/train/*/13*.txt'):
                mesh_files.append(filepath)
        else:
            for filepath in glob.iglob(self.root_dir + '/test/*/13*.txt'):
                mesh_files.append(filepath)

        print("loading {} meshes".format(len(mesh_files)))



        # Load the actual files
        for iFile in range(len(mesh_files)):
            print("loading mesh " + str(mesh_files[iFile]))

            # verts, faces = pp3d.read_mesh(mesh_files[iFile])
            verts = pd.read_csv(mesh_files[iFile], sep=",", header=None)
            verts = verts.to_numpy()[:, :3]
            faces = np.array([])
            folder_num = int(Path(mesh_files[iFile]).parts[-2])

            with open('/'.join(Path(mesh_files[iFile]).parts[:-1]) + '/hmap_per_class.pkl', 'rb') as fp:
                labels = pickle.load(fp)
            #labels = np.loadtxt("/".join(Path(mesh_files[iFile]).parts[:-1]) + '/hmap.txt')

            # for i in range(self.labels.shape[0]):
            #    if int(self.labels[i, 0, 0]) == folder_num:
            #        ldmks_idx = i
            #        break
            # labels = self.labels[ldmks_idx, 1:, :]
            # labels = labels.flatten()  # (204,)

            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))
            #labels = torch.tensor(np.ascontiguousarray(labels))

            # center and unit scale
            verts = diffusion_net.geometry.normalize_positions(verts)

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.labels_list.append(labels)
            self.folder_num_list.append(folder_num)
        for ind, labels in enumerate(self.labels_list):
            self.labels_list[ind] = labels

        # Precompute operators
        self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list = diffusion_net.geometry.get_all_operators(
            self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

        # create sparse labels
        self.createSparseLabels()

        # save to cache
        if use_cache:
            diffusion_net.utils.ensure_dir_exists(self.cache_dir)
            torch.save((self.verts_list, self.faces_list, self.frames_list, self.massvec_list, self.L_list,
                        self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.labels_list,
                        self.folder_num_list), load_cache)

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.frames_list[idx], self.massvec_list[idx], self.L_list[
            idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], \
               self.labels_sparse_list[idx], self.folder_num_list[idx]

    def createSparseLabels(self):
        # create labels sparse list from labels list
        self.labels_sparse_list = []
        for verts in self.verts_list:
            self.labels_sparse_list.append(torch.zeros((68, len(verts))))
        for i in range(len(self.labels_sparse_list)):
            for j in range(len(self.labels_list[i])):
                for k in range(len(self.labels_list[i][j])):
                    pos = self.labels_list[i][j][k, 0]
                    act = self.labels_list[i][j][k, 1]
                    self.labels_sparse_list[i][j, int(pos)] = act