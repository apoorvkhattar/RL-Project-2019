#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from torch import autograd
import torch.nn.functional as  F
# In[2]:

# torch.cuda.set_device(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[3]:
class GenSAGAN(nn.Module):
    def __init__(self, image_size=32, z_dim=32, conv_dim=64):
        super(GenSAGAN, self).__init__()
        repeat_num = int(np.log2(image_size)) - 3
        mult = 2 ** repeat_num

        self.layer1 = nn.ConvTranspose2d(z_dim, conv_dim*mult, 4)
        self.bn1 = nn.BatchNorm2d(conv_dim*mult)

        self.layer2 = nn.ConvTranspose2d(conv_dim*mult, (conv_dim*mult)//2, 3, 2, 2)
        self.bn2 = nn.BatchNorm2d((conv_dim*mult)//2)

        self.layer3 = nn.ConvTranspose2d((conv_dim*mult)//2, (conv_dim*mult)//4, 3, 2, 2)
        self.bn3 = nn.BatchNorm2d((conv_dim*mult)//4)


        self.layer4 = nn.ConvTranspose2d(64, 1, 2, 2, 1)

        self.attn1 = SAttn(64)
        self.attn2 = SAttn(64)

        self.conv1d = nn.ConvTranspose1d(144, 128, 1)


    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        out = F.relu(self.layer1(x))
        out = self.bn1(out)

        out = F.relu(self.layer2(out))
        out = self.bn2(out)

        out = F.relu(self.layer3(out))
        out = self.bn3(out)

        
        out ,  p1 = self.attn1(out)

        out = self.layer4(out)

        out = out.view(-1, 1, 144)
        out = out.transpose(1, 2)

        out = self.conv1d(out)
        out = out.transpose(2, 1)

        out = out.view(-1, 128)

        return out , p1


class DiscSAGAN(nn.Module):

    def __init__(self, image_size=32, conv_dim=64):
        super(DiscSAGAN, self).__init__()
        self.layer1 = nn.Conv2d(1, conv_dim, 3, 2, 2)
        self.layer2 = nn.Conv2d(conv_dim, conv_dim*2, 3, 2, 2)
        self.layer3 = nn.Conv2d(conv_dim*2, conv_dim*4, 3 ,2, 2)

        self.layer4 = nn.Conv2d(conv_dim*4, 1, 4)

        self.attn1 = SAttn(256)
        self.attn2 = SAttn(512)

        self.conv1d = nn.ConvTranspose1d(128, 144, 1)



    def forward(self, x):
        # x = x.squeeze(1)
        x = x.unsqueeze(-1)
        x = self.conv1d(x)
        x = x.transpose(2, 1)
        x = x.view(-1, 1, 12, 12)

        out = F.leaky_relu(self.layer1(x))
        out = F.leaky_relu(self.layer2(out))
        out = F.leaky_relu(self.layer3(out))

        out, p1 = self.attn1(out)

        out = self.layer4(out)
        out = out.reshape(x.shape[0], -1)
        return out, p1


class SAttn(nn.Module):
    def __init__(self, dim):
        super(SAttn, self).__init__()

        self.query = nn.Conv2d(dim, dim // 8, 1)
        self.key = nn.Conv2d(dim, dim//8, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, c, w, h = x.size()
        query = self.query(x)
        query = query.view(batch_size, -1, w*h).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, w*h)
        
        matmul = torch.bmm(query, key)
        attn = self.softmax(matmul)

        value = self.value(x).view(batch_size, -1, w*h)

        out = torch.bmm(value, attn.permute(0,2,1))
        out = out.view(batch_size, c, w, h)
        out = self.gamma*out + x

        return out, attn


class Generator(nn.Module):
    def __init__(self, dim=64, embedding_size=128):
        super(Generator, self).__init__()
        self.linear0 = nn.Linear(32, 64)
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 128)
        
        self.linear3 = nn.Linear(128, 128)
        
        self.bn0 = nn.BatchNorm1d(64)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
    def forward(self, x):
        out = self.bn0(F.leaky_relu(self.linear0(x)))
        out = self.bn1(F.leaky_relu(self.linear1(out)))
        out = self.bn2(F.leaky_relu(self.linear2(out)))
        out = F.leaky_relu(self.linear3(out))
        
        return out


class Discriminator(nn.Module):
    def __init__(self, dim=64, embedding_size=128):
        super(Discriminator, self).__init__()
        self.linear0 = nn.Linear(128, 64)
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 1)
            
    def forward(self, x):
        out = (F.relu(self.linear0(x)))
        out = (F.relu(self.linear1(out)))
        out = (F.leaky_relu(self.linear2(out)))

        return out


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
        # print(out_4.shape)
        out_4 = out_4.view(-1, out_4.shape[1])
        # print(out_4.shape)
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
        
    def encode(self, x):
        gfv = self.encoder(x)
        # out = self.decoder(gfv)
        return gfv

    def decode(self, x):
        return self.decoder(x)

        


# In[6]:


class ChamferLoss(nn.Module):
    def __init__(self, num_points):
        super(ChamferLoss, self).__init__()
        self.num_points = num_points
        self.loss = torch.FloatTensor([0]).to(device)

        
    def forward(self, predict_pc, gt_pc):
        z, _ = torch.min(torch.norm(gt_pc.unsqueeze(-2) - predict_pc.unsqueeze(-1),dim=1), dim=-2)
        self.loss = z.sum() / (len(gt_pc)*self.num_points)

        z_2, _ = torch.min(torch.norm(predict_pc.unsqueeze(-2) - gt_pc.unsqueeze(-1),dim=1), dim=-2)
        self.loss += z_2.sum() / (len(gt_pc)*self.num_points)
        return self.loss


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

BATCH_SIZE = 20
LAMBDA = 1e1
use_cuda = torch.cuda.is_available()
def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


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


train_dataset = PointcloudDatasetAE(DATA_DIR, X_train)
train_dataloader = DataLoader(train_dataset, num_workers=2, shuffle=False, batch_size=BATCH_SIZE)

test_dataset = PointcloudDatasetAE(DATA_DIR, X_test)
test_dataloader = DataLoader(test_dataset, num_workers=2, shuffle=False, batch_size=1)

for i, data in enumerate(train_dataloader):
    data = data.permute([0,2,1])
    print(data.shape)
    break


# In[17]:

z_dim = 5

generator = GenSAGAN(z_dim=z_dim).to(device)
discriminator = DiscSAGAN().to(device) 
autoencoder = AutoEncoder(2048).to(device)
autoencoder.load_state_dict(torch.load('./ae_out_copy/2019-11-27 01:31:11.870769/models/999_ae_.pt'))
# chamfer_loss = ChamferLossOrig(0).to(device)
chamfer_loss = ChamferLoss(2048).to(device)
# In[18]:


ROOT_DIR = './gan_out/'
now =   str(datetime.datetime.now())+'z'+str(z_dim)

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

#%%






def test_model(generator, autoencoder,epoch):
    for i in tqdm.trange(5):
        # points = PyntCloud.from_file(X_test[i])
        # points = np.array(points.points)
        # points_normalized = (points - (-0.5)) / (0.5 - (-0.5))
        # points = points_normalized.astype(np.float)
        # points = torch.from_numpy(points).unsqueeze(0)
        # points = points.permute([0,2,1]).float().to(device)
        # print(points.shape)

        autoencoder.eval()
        generator.eval()
        z = torch.randn(1, z_dim).to(device)
        with torch.no_grad():
                gen_out, _ = generator(z)
                out_data = autoencoder.decode(gen_out)
                # loss = chamfer_loss(out_data, points)
        # print(loss.item())
                
        output = out_data[0,:,:]
        output = output.permute([1,0]).detach().cpu().numpy()

        # inputt = points[0,:,:]
        # inputt = inputt.permute([1,0]).detach().cpu().numpy()

        fig = plt.figure()
        ax_x = fig.add_subplot(111, projection='3d')
        x_ = output
        ax_x.scatter(x_[:, 0], x_[:, 1], x_[:,2])
        ax_x.set_xlim([0,1])
        ax_x.set_ylim([0,1])
        ax_x.set_zlim([0,1])
        fig.savefig(OUTPUTS_DIR+'/{}_{}_{}.png'.format(epoch, i, 'out'))

#%%
# In[19]:

g_lr = 1.0e-4
d_lr = 1.0e-4
lr = 1.0e-4
d_gp_weight = 1e1   
momentum = 0.95


# In[20]:


optimizer_AE = torch.optim.Adam(autoencoder.parameters(), lr=lr, betas=(momentum, 0.999))
g_optim = torch.optim.Adam(generator.parameters(), lr=g_lr)
d_optim = torch.optim.Adam(discriminator.parameters(), lr=d_lr)


# In[ ]:

print('Training')
for epoch in range(1000):
    autoencoder.train()
    for i, data in enumerate(train_dataloader):
        data = data.permute([0,2,1]).float().to(device)

        # optimizer_AE.zero_grad()
        autoencoder.eval()
        generator.train()
        discriminator.train()

        

        with torch.no_grad():
            gfv = autoencoder.encode(data)

        z = torch.randn(data.shape[0], z_dim).to(device)



        g_optim.zero_grad()
        d_optim.zero_grad()

        fake_out, _ = generator(z)
        # print(gfv.device)
        d_fake, _ = discriminator(fake_out)
        d_real, _ = discriminator(gfv)
        d_loss = -(torch.mean(d_real) - torch.mean(d_fake))
        d_grad_penalty = calc_gradient_penalty(discriminator, gfv, fake_out)
        total_d_loss = d_loss + d_grad_penalty
        total_d_loss.backward()
        d_optim.step()

        #####################################

        g_optim.zero_grad()
        d_optim.zero_grad()
        

        g_out, _ = generator(z)
        
        d_fake, _ = discriminator(g_out)
        gen_loss = -torch.mean(d_fake)
        
        out_data = autoencoder.decode(g_out)
        #l2_loss = F.mse_loss(g_out, gfv)
        #ch_loss = chamfer_loss(out_data, data)

        loss = gen_loss #+ 1.*ch_loss + 1.*l2_loss
        loss.backward()
        g_optim.step()
        print('Epoch: {}, Iteration: {},  G Loss: {:.4f} D Loss: {:.4f} '.format(epoch, i, loss.item(), total_d_loss.item()))
        # print('Epoch: {}, Iteration: {}, G Total Loss: {:.4f} G Loss: {:.4f} D Loss: {:.4f} Content Loss: {:.4f}'.format(epoch, i, loss.item(), total_d_loss.item(), gen_loss.item(), ch_loss.item()))
        summary_writer.add_scalar('G Loss', loss.item())
        # summary_writer.add_scalar('G Loss', gen_loss.item())
        # summary_writer.add_scalar('GFV loss', l2_loss.item())
        # summary_writer.add_scalar('Chamfer Loss', ch_loss.item())
        summary_writer.add_scalar('GP  Loss', d_grad_penalty.item())
        summary_writer.add_scalar('D Loss', d_loss.item())
        summary_writer.add_scalar('Total D Loss', total_d_loss.item())
    if epoch % 20 == 0:
        torch.save(generator.state_dict(), MODEL_DIR+'{}_gen_.pt'.format(epoch))
        torch.save(discriminator.state_dict(), MODEL_DIR+'{}_disc_.pt'.format(epoch))
    if epoch % 5 == 0:
        test_model(generator, autoencoder, epoch)