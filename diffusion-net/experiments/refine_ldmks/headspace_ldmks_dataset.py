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
    def __init__(self, root_dir, train, data_format, num_landmarks, test_without_score, k_eig=128, use_cache=True, op_cache_dir=None):
        self.use_cache = use_cache
        self.train = train  # bool
        self.k_eig = k_eig
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, "cache", "train" if self.train else "test")
        self.op_cache_dir = op_cache_dir
        self.data_format = data_format
        self.num_landmarks = num_landmarks
        self.test_without_score = test_without_score

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.folder_num_list = []
        self.folder_num_ldmk_list = []
        self.num_samples = 0

        mesh_files = []


        filepattern = '/*/*/13*.txt' if data_format == 'pcl' else '/*/13*.obj'
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
                verts = np.loadtxt(mesh_files[iFile], delimiter = ',')
                #lines = open(mesh_files[iFile], 'r').read().split('\n')[:-1]
                #verts = [[float(l) for l in lines[j].split(',')[:3]] for j in range(len(lines))]
                #verts = np.array(verts)
                #verts = pd.read_csv(mesh_files[iFile], sep=",", header=None)
                #verts = verts.to_numpy()[:, :3]
                faces = np.array([])
            folder_num = Path(mesh_files[iFile]).parts[-3]
            folder_num_lmkd = Path(mesh_files[iFile]).parts[-2]
            if len(verts) == 0:
                raise RunTimeError('verts len is 0')
            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))

            # center and unit scale
            verts = diffusion_net.geometry.normalize_positions(verts)

            self.folder_num_list.append(folder_num)
            self.folder_num_ldmk_list.append(folder_num_lmkd)

            # if this file is not cached, populate
            if not os.path.isfile(os.path.join(self.cache_dir, '{}_{}.pt'.format(folder_num, folder_num_lmkd))):
                # Precompute operators
                diffusion_net.utils.ensure_dir_exists(self.cache_dir)
                frames, massvec, L, evals, evecs, gradX, gradY = diffusion_net.geometry.populate_cache(
                    verts, faces, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)
                torch.save((verts, faces, frames, massvec, L,
                            evals, evecs, gradX, gradY), os.path.join(self.cache_dir, '{}_{}.pt'.format(folder_num,\
                                                                                                folder_num_lmkd)))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        folder_num = self.folder_num_list[idx]
        folder_num_ldmk = self.folder_num_ldmk_list[idx]
        path_cache = os.path.join(self.cache_dir, folder_num, folder_num_ldmk)
        verts, faces, frames, massvec, L, evals, evecs, gradX, gradY = torch.load(os.path.join(self.cache_dir,\
                                                                '{}_{}.pt'.format(folder_num, folder_num_ldmk)))

        # create sparse labels
        #landmark_indices = {8,27,30,33,36,39,45,42,60,64} # indices start with 1
        landmark_indices = {33}  # indices start with 1
        if not self.train and self.test_without_score:
            labels = np.array([])
        else:
            with open(os.path.join(self.root_dir, 'train' if self.train else 'test', folder_num, folder_num_ldmk,\
                                   'hmap_per_class.pkl'), 'rb') as fpath:
                labels_sparse = pickle.load(fpath)
            #labels_sparse = [item for pos, item in enumerate(labels_sparse) if pos in landmark_indices]
            labels = self.labelsFromSparse(verts, labels_sparse)


        return verts, faces, frames, massvec, L, evals, evecs, gradX, gradY, labels, folder_num, folder_num_ldmk

    def labelsFromSparse(self, verts, labels_sparse):
        # create labels from sparse representation
        labels = torch.zeros((len(verts)))
        for k in range(len(labels_sparse)):
            pos = labels_sparse[k, 0]
            act = labels_sparse[k, 1]
            labels[int(pos)] = act
        return labels

    