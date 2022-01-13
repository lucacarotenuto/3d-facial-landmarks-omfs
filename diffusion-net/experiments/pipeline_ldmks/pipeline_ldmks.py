# Inference pipeline that first predicts landmarks in low-resolution, then refines prediction in high-resolution and
# computes error to ground truth

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from refine_ldmks_dataset import HeadspaceLdmksDataset


def weighted_mse_loss(input, target, weight):
    """
    Weighted mean squared error loss

    """
    return (weight * (input - target) ** 2).mean()


def point_weights(labels):
    """
    Creates per-point weights

    Args:
        labels: per-point labels

    Returns: weights
    """
    weights = torch.clone(labels)
    weights[weights == 1] = 20
    weights[weights == 0.75] = 15
    weights[weights == 0.5] = 10
    weights[weights == 0.25] = 5
    weights[weights == 0] = 1
    return weights


def train_epoch(epoch):
    # Implement lr decay
    if epoch > 0 and epoch % decay_every == 0:
        global lr
        lr *= decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

            # Set model to 'train' mode
    model.train()
    optimizer.zero_grad()

    loss_sum = 0
    for data in tqdm(train_loader):

        # Get data
        verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, folder_num, folder_num_ldmk = data

        # Move to device
        verts = verts.to(device)
        faces = faces.to(device)
        frames = frames.to(device)
        mass = mass.to(device)
        L = L.to(device)
        evals = evals.to(device)
        evecs = evecs.to(device)
        gradX = gradX.to(device)
        gradY = gradY.to(device)
        # labels = labels.to(device)
        labels = [x.to(device) for x in labels]

        # Randomly rotate positions
        # if augment_random_rotate:
        #    verts = diffusion_net.utils.random_rotate_points(verts)

        # Construct features
        if input_features == 'xyz':
            features = verts
        elif input_features == 'hks':
            features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

        # Apply the model
        preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

        # preds = preds.float()
        predstp = torch.transpose(preds, 0, 1)
        predstp = predstp.flatten()
        # labels = torch.cat([x for x in labels])
        labels = torch.FloatTensor(labels)

        weights = point_weights(labels)

        # Evaluate loss
        loss = weighted_mse_loss(predstp.to('cpu'), labels, weights)
        loss.backward()

        # Step the optimizer
        optimizer.step()
        optimizer.zero_grad()
        loss_sum += loss
    return loss_sum / len(train_dataset)


# Do an evaluation pass on the test dataset
def test():
    model.eval()

    with torch.no_grad():
        loss_sum = 0
        for data in tqdm(test_loader):

            # Get data
            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, folder_num, folder_num_ldmk = data

            # Move to device
            verts = verts.to(device)
            faces = faces.to(device)
            frames = frames.to(device)
            mass = mass.to(device)
            L = L.to(device)
            evals = evals.to(device)
            evecs = evecs.to(device)
            gradX = gradX.to(device)
            gradY = gradY.to(device)
            labels = labels.to(device)

            # Construct features
            if input_features == 'xyz':
                features = verts
            elif input_features == 'hks':
                features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

            # Apply the model
            preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
            predstp = torch.transpose(preds, 0, 1)
            predstp = predstp.flatten()

            diffusion_net.utils.ensure_dir_exists(os.path.join(dataset_path, 'preds' + ('_validation' if train
                                                                                                   else '')))
            f = open(os.path.join(dataset_path, 'preds' + ('_validation' if train else ''),
                     'pred{}_{}.pkl'.format(folder_num, folder_num_ldmk)), 'wb+')
            pickle.dump(np.asarray(preds.cpu()), f)
            f.close()

            if not test_without_score:
                labels = torch.FloatTensor(labels)
                weights = point_weights(labels)
                loss_sum += weighted_mse_loss(predstp, labels, weights)

    return loss_sum / len(test_dataset)

# === Options

# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument('--evaluate', action="store_true", help="evaluate using the pretrained model")
parser.add_argument('--input_features', type=str, help="what features to use as input ('xyz' or 'hks') default: hks", default='hks')
parser.add_argument('--data_format', type=str, help="what format does the data have ('pcl' or 'mesh') default: pcl", default='pcl')
parser.add_argument('--data_dir', type=str, help="directory name of dataset", default='pcl')
parser.add_argument('--test_without_score', action='store_true', help="do not evaluate with score"
                                                                      "(if no ground truth available)")
parser.add_argument('--validate', action='store_true', help='if validating the training of refinement model')


args = parser.parse_args()


# system things
#device = torch.device('cuda:0') if
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
dtype = torch.float32

# problem/dataset things
n_class = 1
data_format = args.data_format

# model
input_features = args.input_features # one of ['xyz', 'hks']
k_eig = 128

# training settings
train = not args.evaluate
test_without_score = args.test_without_score
validate = args.validate
n_epoch = 2
lr = 1e-3
decay_every = 50
decay_rate = 0.5
n_block = 4
c_width = 256
#augment_random_rotate = (input_features == 'xyz')


for ldmk_iter in [0, 4, 5, 6, 7, 8, 9]:

    # Important paths
    base_path = os.path.dirname(__file__)
    dataset_path = os.path.join(base_path, args.data_dir)
    op_cache_dir = os.path.join(dataset_path, "op_cache")
    last_model_path = os.path.join(dataset_path, "saved_models/headspace_ldmks_last_{}_{}x{}_ldmk{}.pth".format(input_features,
                                                                                n_block, c_width, ldmk_iter))
    best_model_path = os.path.join(dataset_path, "saved_models/headspace_ldmks_best_{}_{}x{}_ldmk{}.pth".format(input_features,
                                                                                n_block, c_width, ldmk_iter))
    pretrain_path = best_model_path

    # === Load datasets

    # Load the test dataset
    test_dataset = HeadspaceLdmksDataset(dataset_path, data_format=data_format, train=False, num_landmarks=n_class,
                                         test_without_score=test_without_score, validate=validate, ldmk_idx=ldmk_iter, k_eig=k_eig,
                                         use_cache=True, op_cache_dir=op_cache_dir)
    test_loader = DataLoader(test_dataset, batch_size=None)


    # Load the train dataset
    if train:
        train_dataset = HeadspaceLdmksDataset(dataset_path, data_format=data_format, train=True, num_landmarks=n_class,
                                              test_without_score=test_without_score, validate=validate, ldmk_idx=ldmk_iter, k_eig=k_eig,
                                              use_cache=True, op_cache_dir=op_cache_dir)
        train_loader = DataLoader(train_dataset, batch_size=None)

    # === Create the model

    C_in={'xyz':3, 'hks':16}[input_features] # dimension of input features

    model = diffusion_net.layers.DiffusionNet(C_in=C_in,
                                              C_out=n_class,
                                              C_width=c_width,
                                              N_block=n_block,
                                              #last_activation=lambda x : torch.mean(x,dim=1),
                                              outputs_at='vertices',
                                              dropout=True)


    model = model.to(device)

    if not train:
        # load the pretrained model
        print("Loading pretrained model from: " + str(pretrain_path))
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(pretrain_path))
        else:
            model.load_state_dict(torch.load(pretrain_path, map_location=torch.device('cpu')))


    # === Optimize
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)



    if train:
        print("Training...")

        best_acc = 99999
        for epoch in range(n_epoch):
            train_acc = train_epoch(epoch)
            test_acc = test()
            print("Epoch {} - Train overall: {}  Test overall: {}".format(epoch, train_acc, test_acc))
            #if epoch % 10 == 0:
            if test_acc < best_acc:
                best_acc = test_acc
                print(" ==> saving model to " + best_model_path)
                diffusion_net.utils.ensure_dir_exists(os.path.join(dataset_path, 'saved_models'))
                torch.save(model.state_dict(), best_model_path)

        # diffusion_net.utils.ensure_dir_exists(os.path.join(dataset_path, 'saved_models'))
        # print(" ==> saving last model to " + last_model_path)
        # torch.save(model.state_dict(), last_model_path)


    # Test
    test_acc = test()
    print("Overall test accuracy: {}".format(test_acc))

    # remove cache and op_cache
    shutil.rmtree(os.path.join(dataset_path, 'cache'))
    shutil.rmtree(os.path.join(dataset_path, 'op_cache'))




