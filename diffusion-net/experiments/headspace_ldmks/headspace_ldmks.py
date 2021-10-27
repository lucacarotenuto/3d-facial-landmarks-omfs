import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from headspace_ldmks_dataset import HeadspaceLdmksDataset


# === Options

# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", action="store_true", help="evaluate using the pretrained model")
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks", default = 'hks')
args = parser.parse_args()


# system things
#device = torch.device('cuda:0') if
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
dtype = torch.float32

# problem/dataset things
n_class = 68

# model 
input_features = args.input_features # one of ['xyz', 'hks']
k_eig = 128

# training settings
train = not args.evaluate
n_epoch = 1
lr = 1e-3
decay_every = 50
decay_rate = 0.5
#augment_random_rotate = (input_features == 'xyz')



# Important paths
base_path = os.path.dirname(__file__)
op_cache_dir = os.path.join(base_path, "headspace_pcl_hmap2_3k", "op_cache")
pretrain_path = os.path.join(base_path, "pretrained_models/headspace_ldmks_{}_4x128.pth".format(input_features))
model_save_path = os.path.join(base_path, "saved_models/headspace_ldmks_{}_4x128.pth".format(input_features))
dataset_path = os.path.join(base_path, "headspace_pcl_hmap2_3k")


# === Load datasets

# Load the test dataset
test_dataset = HeadspaceLdmksDataset(dataset_path, train=False, k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
test_loader = DataLoader(test_dataset, batch_size=None)

# Load the train dataset
if train:
    train_dataset = HeadspaceLdmksDataset(dataset_path, train=True, k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)



# === Create the model

C_in={'xyz':3, 'hks':16}[input_features] # dimension of input features

model = diffusion_net.layers.DiffusionNet(C_in=C_in,
                                          C_out=n_class,
                                          C_width=128, 
                                          N_block=4, 
                                          #last_activation=lambda x : torch.mean(x,dim=1),
                                          #last_activation=lambda x : torch.nn.Linear(x),
                                          outputs_at='vertices',
                                          dropout=True)


model = model.to(device)

if not train:
    # load the pretrained model
    print("Loading pretrained model from: " + str(pretrain_path))
    model.load_state_dict(torch.load(pretrain_path))


# === Optimize
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
    
    correct = 0
    total_num = 0
    loss_sum = 0
    for data in tqdm(train_loader):

        # Get data
        verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, folder_num = data

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
        #labels = labels.to(device)
        labels = [x.to(device) for x in labels]
        
        # Randomly rotate positions
        #if augment_random_rotate:
        #    verts = diffusion_net.utils.random_rotate_points(verts)

        # Construct features
        if input_features == 'xyz':
            features = verts
        elif input_features == 'hks':
            features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

        # Apply the model
        preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

        #preds = preds.float()
        predstp = torch.transpose(preds,0,1)
        predstp = predstp.flatten()
        labels = torch.cat([x for x in labels])
        weights = torch.clone(labels)
        weights[weights == 1] = 100
        weights[weights == 0.75] = 75
        weights[weights == 0.5] = 50
        weights[weights == 0.25] = 25
        weights[weights == 0] = 1

        #labels = labels.float()
        # Evaluate loss
        #loss = torch.nn.functional.nll_loss(preds, labels)
        #lossf = torch.nn.MSELoss()
        #lossf = weighted_mse_loss(preds)
        loss = weighted_mse_loss(predstp, labels, weights)
#        loss =
        loss.backward()
        
        # track accuracy
        #pred_labels = torch.max(preds, dim=1).indices
        #this_correct = pred_labels.eq(labels).sum().item()
        #this_num = labels.shape[0]
        #correct += this_correct
        #total_num += this_num

        # Step the optimizer
        optimizer.step()
        optimizer.zero_grad()
        loss_sum += loss
    #train_acc = correct / total_num
    return loss_sum/len(train_dataset)


# Do an evaluation pass on the test dataset 
def test():
    
    model.eval()
    
    correct = 0
    total_num = 0
    with torch.no_grad():
        loss_sum = 0
        for data in tqdm(test_loader):

            # Get data
            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, folder_num = data

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

            diffusion_net.utils.ensure_dir_exists(os.path.join(dataset_path, 'preds'))
            f = open(dataset_path + '/preds/hmap_per_class' + str(folder_num) + ".pkl", "wb+")
            #f = open('/Users/carotenuto/clones/diffusion-net/experiments/headspace_ldmks/headspace_pcl_hmap100_fullres/preds/hmap_per_class' + str(folder_num) + ".pkl", "wb+")
            pickle.dump(np.asarray(preds.cpu()), f)
            #for e in preds:
            #    f.write(str(float(e)) + '\n')
            f.close()

            labels = torch.cat([x for x in labels])
            weights = torch.clone(labels)
            weights[weights == 1] = 100
            weights[weights == 0.75] = 75
            weights[weights == 0.5] = 50
            weights[weights == 0.25] = 25
            weights[weights == 0] = 1
            loss_sum += weighted_mse_loss(predstp, labels, weights)
            # track accuracy
            #pred_labels = torch.max(preds, dim=1).indices
            #this_correct = pred_labels.eq(labels).sum().item()
            #this_num = labels.shape[0]
            #correct += this_correct
            #total_num += this_num

    #test_acc = correct / total_num
    return loss_sum/len(test_dataset)
    #return torch.nn.functional.l1_loss(predstp, labels)

def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).mean()

if train:
    print("Training...")

    for epoch in range(n_epoch):
        train_acc = train_epoch(epoch)
        test_acc = test()
        print("Epoch {} - Train overall: {}  Test overall: {}".format(epoch, train_acc, test_acc))

    print(" ==> saving last model to " + model_save_path)
    torch.save(model.state_dict(), model_save_path)



# Test
test_acc = test()
print("Overall test accuracy: {}".format(test_acc))
