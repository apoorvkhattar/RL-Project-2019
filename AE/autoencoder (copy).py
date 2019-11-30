#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import tqdm
import torch
# import faiss
import datetime

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from pyntcloud import PyntCloud
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[3]:


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        conv1 = [nn.Conv1d(3, 64, kernel_size=1), 
                nn.BatchNorm1d(64),
                nn.ReLU()]
        conv2 = [nn.Conv1d(64, 128, kernel_size=1), 
                nn.BatchNorm1d(128),
                nn.ReLU()]
        conv3 = [nn.Conv1d(128, 256, kernel_size=1), 
                nn.BatchNorm1d(256),
                nn.ReLU()]
        conv4 = [nn.Conv1d(256, 128, kernel_size=1), 
                nn.BatchNorm1d(128),
                nn.AdaptiveMaxPool1d(1)]
        self.conv1 = nn.Sequential(*conv1)
        self.conv2 = nn.Sequential(*conv2)        
        self.conv3 = nn.Sequential(*conv3)
        self.conv4 = nn.Sequential(*conv4)
        
    def forward(self, x):
        out_1 = self.conv1(x)
        out_2 = self.conv2(out_1)
        out_3 = self.conv3(out_2)
        out_4 = self.conv4(out_3)
        print(out_4.shape)
        out_4 = out_4.view(-1, out_4.shape[1])
        print(out_4.shape)
        return out_4


# In[4]:


class Decoder(nn.Module):
    def __init__(self, num_points):
        super(Decoder, self).__init__()
        linear1 = [nn.Linear(128, 256), 
                nn.BatchNorm1d(256),
                nn.ReLU()]
        linear2 = [nn.Linear(256, 256), 
                nn.BatchNorm1d(256),
                nn.ReLU()]
        linear3 = [nn.Linear(256, 6144), 
                nn.ReLU()]
        self.linear1 = nn.Sequential(*linear1)
        self.linear2 = nn.Sequential(*linear2)
        self.linear3 = nn.Sequential(*linear3)
        self.num_points = num_points
        
    def forward(self, x):
        out_1 = self.linear1(x)
        out_2 = self.linear2(out_1)
        out_3 = self.linear3(out_2)
        
        return out_3.view(-1, 3, self.num_points)


# In[5]:


class AutoEncoder(nn.Module):
    def __init__(self, num_points):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_points)
        
    def forward(self, x):
        gfv = self.encoder(x)
        out = self.decoder(gfv)
        
        return out, gfv


# In[6]:


class ChamferLoss(nn.Module):
    def __init__(self, num_points):
        super(ChamferLoss, self).__init__()
        self.num_points = num_points
        self.loss = torch.FloatTensor([0]).to(device)

        
    def forward(self, predict_pc, gt_pc):
        z, _ = torch.min(torch.norm(gt_pc.unsqueeze(-2) - predict_pc.unsqueeze(-1),dim=1), dim=-2)
            # self.loss += z.sum()
        self.loss = z.sum() / (len(gt_pc)*self.num_points)

        z_2, _ = torch.min(torch.norm(predict_pc.unsqueeze(-2) - gt_pc.unsqueeze(-1),dim=1), dim=-2)
        self.loss += z_2.sum() / (len(gt_pc)*self.num_points)
        return self.loss


# In[7]:

def robust_norm(var):
    '''
    :param var: Variable of BxCxHxW
    :return: p-norm of BxCxW
    '''
    result = ((var**2).sum(dim=2) + 1e-8).sqrt() # TODO try infinity norm
    # result = (var ** 2).sum(dim=2)

    # try to make the points less dense, caused by the backward loss
    # result = result.clamp(min=7e-3, max=None)
    return result


# In[8]:


def prepare_dataset(root):
    maxx = -1.0e6
    minn = 1.0e6
    list_point_clouds = []
    print('Preparing dataset')
    sub_dirs = os.listdir(root)
    for i in tqdm.trange(len(sub_dirs)):
        list_files = os.listdir(root+sub_dirs[i])
        for j in range(len(list_files)):
            list_point_clouds.append(os.path.join(root+sub_dirs[i], list_files[j]))
            points = PyntCloud.from_file(os.path.join(root+sub_dirs[i], list_files[j]))
            minn = min(minn, np.array(points.points).min())
            maxx = max(maxx, np.array(points.points).max())
    return list_point_clouds, minn, maxx


# In[9]:


DATA_DIR = '../latent_3d_points/data/shape_net_core_uniform_samples_2048/'
# list_point_clouds, minn, maxx = prepare_dataset(DATA_DIR)
# np.save('list_point.npy', list_point_clouds)
# np.save('minn.npy', minn)
# np.save('maxx.npy', maxx)
# exit()

list_point_clouds = np.load('./list_point.npy')
minn = np.load('./minn.npy')
maxx = np.load('./maxx.npy')
list_point_clouds = list_point_clouds[:5000]
np.save('list_point_subset_2.npy', list_point_clouds)

X_train, X_test, _, _ = train_test_split(list_point_clouds, list_point_clouds, test_size=0.1, random_state=42)
print(len(X_train))

# In[11]:


maxx


# In[15]:


class PointcloudDatasetAE(Dataset):
    def __init__(self, root, list_point_clouds):
        self.root = root
        self.list_files = list_point_clouds
        
    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        points = PyntCloud.from_file(self.list_files[index])
        points = np.array(points.points)
        points_normalized = (points - (-0.5)) / (0.5 - (-0.5))
        points = points_normalized.astype(np.float)
        points = torch.from_numpy(points)
        
        return points


# In[16]:


train_dataset = PointcloudDatasetAE(DATA_DIR, X_train)
train_dataloader = DataLoader(train_dataset, num_workers=2, shuffle=False, batch_size=48)

test_dataset = PointcloudDatasetAE(DATA_DIR, X_test)
test_dataloader = DataLoader(test_dataset, num_workers=2, shuffle=False, batch_size=1)

for i, data in enumerate(train_dataloader):
    data = data.permute([0,2,1])
    print(data.shape)
    break


# In[17]:


autoencoder = AutoEncoder(2048).to(device)
# chamfer_loss = ChamferLossOrig(0).to(device)
chamfer_loss = ChamferLoss(2048).to(device)
# In[18]:


ROOT_DIR = './ae_out_copy/'
now =   str(datetime.datetime.now())

if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)

if not os.path.exists(ROOT_DIR + now):
    os.makedirs(ROOT_DIR + now)

LOG_DIR = ROOT_DIR + now + '/logs/'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

OUTPUTS_DIR = ROOT_DIR  + now + '/outputs/'
if not os.path.exists(OUTPUTS_DIR):
    os.makedirs(OUTPUTS_DIR)

MODEL_DIR = ROOT_DIR + now + '/models/'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

summary_writer = SummaryWriter(LOG_DIR)


# In[19]:


lr = 1.0e-4
momentum = 0.95


# In[20]:


optimizer_AE = torch.optim.Adam(autoencoder.parameters(), lr=lr, betas=(momentum, 0.999))


# In[ ]:

print('Training')
for epoch in range(1000):
    autoencoder.train()
    for i, data in enumerate(train_dataloader):
        data = data.permute([0,2,1]).float().to(device)

        optimizer_AE.zero_grad()
        out_data, gfv = autoencoder(data)

        loss = chamfer_loss(out_data, data)
        loss.backward()
        optimizer_AE.step()
        
        print('Epoch: {}, Iteration: {}, Content Loss: {}'.format(epoch, i, loss.item()))
        summary_writer.add_scalar('Content Loss', loss.item())
    
    torch.save(autoencoder.state_dict(), MODEL_DIR+'{}_ae_.pt'.format(epoch))
