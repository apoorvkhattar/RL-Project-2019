import os
import sys
import tqdm
import torch
import datetime

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from pyntcloud import PyntCloud
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

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
                nn.MaxPool1d(2048)]
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

autoencoder = AutoEncoder(2048).to(device)
autoencoder.load_state_dict(torch.load('./789_ae_.pt'))

chamfer_loss = ChamferLoss(2048).to(device)

list_point_clouds = np.load('./list_point_subset_2.npy')
X_train, X_test, _, _ = train_test_split(list_point_clouds, list_point_clouds, test_size=0.1, random_state=42)

for i in range(X_test.shape[0]):
        points = PyntCloud.from_file(X_test[i])
        points = np.array(points.points)
        points_normalized = (points - (-0.5)) / (0.5 - (-0.5))
        points = points_normalized.astype(np.float)
        points = torch.from_numpy(points).unsqueeze(0)
        points = points.permute([0,2,1]).float().to(device)
        print(points.shape)

        autoencoder.eval()

        with torch.no_grad():
                out_data, gfv = autoencoder(points)
                loss = chamfer_loss(out_data, points)
        print(loss.item())
                
        output = out_data[0,:,:]
        output = output.permute([1,0]).detach().cpu().numpy()

        inputt = points[0,:,:]
        inputt = inputt.permute([1,0]).detach().cpu().numpy()

        fig = plt.figure()
        ax_x = fig.add_subplot(111, projection='3d')
        x_ = output
        ax_x.scatter(x_[:, 0], x_[:, 1], x_[:,2])
        ax_x.set_xlim([0,1])
        ax_x.set_ylim([0,1])
        ax_x.set_zlim([0,1])
        fig.savefig('eval_output/{}_{}.png'.format(i, 'out'))

        fig = plt.figure()
        ax_x = fig.add_subplot(111, projection='3d')
        x_ = inputt
        ax_x.scatter(x_[:, 0], x_[:, 1], x_[:,2])
        ax_x.set_xlim([0,1])
        ax_x.set_ylim([0,1])
        ax_x.set_zlim([0,1])
        fig.savefig('eval_output/{}_{}.png'.format(i, 'in'))

        plt.close('all')
