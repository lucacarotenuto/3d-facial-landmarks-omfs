import os
import torch
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad
from models.layers.mesh import Mesh
import pathlib
import numpy as np

class ClassificationData(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot)
        self.classes, self.class_to_idx = self.find_classes(self.dir)
        self.paths = self.make_dataset_by_class(self.dir, self.class_to_idx, opt.phase)
        self.nclasses = len(self.classes)
        self.size = len(self.paths)
        self.get_mean_std()
        # modify for network later.
        opt.nclasses = self.nclasses
        opt.input_nc = self.ninput_channels

    def __getitem__(self, index):
        path = self.paths[index][0]
        #label = self.paths[index][1]
        label_index = int(pathlib.Path(self.paths[index][0]).parts[2][-1]) - 1
        labels = torch.tensor([[-39.263191, -31.227459, 30.902531],  #1
                               [-39.059505, -24.339855, 43.244362],  #2
                               [-39.807362, -43.25309, 60.427628],  #3
                               [-32.064659, -41.070557, 39.225555],  #4
                               [-2.316769, -21.537405, 76.362061],  #5
                               [-52.490776, -9.491112, 66.128883],  # 10
                               [-19.039095, -30.652889, 53.925327],  #7
                               [-15.428528, -42.542801, 24.401356],  #9

                               ])
        labels = labels.numpy()
        label = labels[label_index]
        mesh = Mesh(file=path, opt=self.opt, hold_history=False, export_folder=self.opt.export_folder)
        meta = {'mesh': mesh, 'label': label}
        # get edge features
        edge_features = mesh.extract_features()
        edge_features = pad(edge_features, self.opt.ninput_edges)
        meta['edge_features'] = (edge_features - self.mean) / self.std
        return meta

    def __len__(self):
        return self.size

    # this is when the folders are organized by class...
    @staticmethod
    def find_classes(dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def make_dataset_by_class(dir, class_to_idx, phase):
        meshes = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_mesh_file(fname) and (root.count(phase)==1):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        meshes.append(item)
        return meshes
