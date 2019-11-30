import os
import sys
import tqdm
import torch
import random
import datetime

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from pyntcloud import PyntCloud
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)


# # All Models

# ### Replay Buffer

# In[3]:


class ReplayBuffer():
    def __init__(self, size):
        self.episodes = []
        self.buffer_size = size

    def add_to_buffer(self, state, action, reward, next_state):
        if len(self.episodes) == self.buffer_size:
            self.episodes = self.episodes[1:]
        self.episodes.append((state.detach().cpu().numpy(), action.detach().cpu().numpy(), reward.detach().cpu().numpy(), next_state.detach().cpu().numpy()))

    def get_batch(self, batch_size=10):
        states = []
        actions = []
        rewards = []
        next_state = []
        done = []

        for i in range(batch_size):
            epi = random.choice(self.episodes)
            states.append(epi[0])
            actions.append(epi[1])
            rewards.append(epi[2])
            next_state.append(epi[3])
        
        rewards = np.array(rewards)
        rewards = rewards.reshape((rewards.shape[0],1))
        return torch.Tensor(states), torch.Tensor(actions), torch.Tensor(rewards), torch.Tensor(next_state)


# ### Critic Network

# In[4]:


class CriticNet(nn.Module):
    def __init__(self, state_dim, z_shape):
        super(CriticNet, self).__init__()
        self.state_dim = state_dim
        self.num_actions = z_shape
        

        self.linear1 = nn.Linear(self.state_dim, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.linear2 = nn.Linear(400 + z_shape, 300)
        self.bn2 = nn.BatchNorm1d(300)
        self.linear3 = nn.Linear(300, 300)
        self.linear4 = nn.Linear(300, 1)

        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, state, z):
        out = (F.relu(self.linear1(state)))
        out = (F.relu(self.linear2(torch.cat([out, z], dim=1))))
        out = self.linear3(out)
        out = self.linear4(out)

        return out


# ### Actor Network

# In[5]:


class ActorNet(nn.Module):
    def __init__(self, state_dim,  z_shape, max_action=10):
        super(ActorNet, self).__init__()
        self.state_dim = state_dim
        self.num_actions = z_shape

        self.linear1 = nn.Linear(self.state_dim, 400)
        self.bn1 = nn.BatchNorm1d(100)

        self.linear2 = nn.Linear(400, 400)
        self.bn2 = nn.BatchNorm1d(300)

        self.linear3 = nn.Linear(400, 300)
        self.linear4 = nn.Linear(300, self.num_actions)

        self.max_action = max_action

        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        out = F.leaky_relu((self.linear1(x)))
        out = F.leaky_relu((self.linear2(out)))
        out = F.tanh(self.linear3(out))
        out = self.max_action * F.tanh(self.linear4(out))
        return out


# ### Autoencoder

# In[6]:


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
        out_4 = out_4.view(-1, out_4.shape[1])
        return out_4

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

class AutoEncoder(nn.Module):
    def __init__(self, num_points):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_points)

    def encode(self, x):
        gfv = self.encoder(x)
        return gfv

    def decode(self, x):
        return self.decoder(x)
    
class ChamferLoss(nn.Module):
    def __init__(self, num_points):
        super(ChamferLoss, self).__init__()
        self.num_points = num_points
        self.loss = torch.FloatTensor([0]).to(device)

    def forward(self, predict_pc, gt_pc):
        z, _ = torch.min(torch.norm(gt_pc.unsqueeze(-2) -
                                    predict_pc.unsqueeze(-1), dim=1), dim=-2)
        self.loss = z.sum() / (len(gt_pc)*(gt_pc.shape[2]+predict_pc.shape[2]))

        z_2, _ = torch.min(torch.norm(
            predict_pc.unsqueeze(-2) - gt_pc.unsqueeze(-1), dim=1), dim=-2)
        self.loss += z_2.sum() / (len(gt_pc)*(gt_pc.shape[2]+predict_pc.shape[2]))
        return self.loss


# ### Generator

# In[7]:


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


# ### Self-Attention

# In[8]:


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


# ### Discriminator

# In[9]:


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

class DDPG(nn.Module):
    def __init__(self, max_action):
        super(DDPG, self).__init__()
        self.actor = ActorNet(128, z_dim, max_action)
        self.critic = CriticNet(128, z_dim)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.replay_buffer = ReplayBuffer(int(1e6))

    def get_optimal_action(self, state):
        return self.actor(state)

    def forward(self):
        state, action, reward, next_state = self.replay_buffer.get_batch(batch_size_actor)
        
        state = state[:,0,:].float()
        next_state = next_state[:,0,:].float()
        action = action[:,0,:].float()
        state = state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        next_state = next_state.to(device)
        
        target_q = reward

        q_batch = self.critic(state, action)

        self.critic_optimizer.zero_grad()

        value_loss = F.mse_loss(q_batch, target_q)
        value_loss.backward()
        
        self.critic_optimizer.step() 

        self.actor_optimizer.zero_grad()

        policy_loss = - self.critic(state, self.actor(state)).mean()
        policy_loss.backward()
        
        self.actor_optimizer.step()

        return value_loss, policy_loss


# In[10]:


weights_ae = './ae_out_copy/2019-11-27 01:31:11.870769/models/990_ae_.pt'
# weights_gen = './gan_out/2019-11-27 16:35:38.423591/models/720_gen_.pt'
# weight_disc = './gan_out/2019-11-27 16:35:38.423591/models/720_disc_.pt'
weights_gen = './gan_out/2019-11-29 17:33:15.146770z5/models/980_gen_.pt'
weight_disc = './gan_out/2019-11-29 17:33:15.146770z5/models/980_disc_.pt'
weight_ddpg = './335000_ddpg_.pt'

max_action = 2
z_dim = 5


# In[11]:


autoencoder = AutoEncoder(2048).to(device)
autoencoder.load_state_dict(torch.load(weights_ae))

generator = GenSAGAN(z_dim=z_dim).to(device)
generator.load_state_dict(torch.load(weights_gen))

discriminator = DiscSAGAN().to(device)
discriminator.load_state_dict(torch.load(weight_disc))

ddpg = DDPG(max_action).to(device)
ddpg.load_state_dict(torch.load(weight_ddpg))

autoencoder.eval()
generator.eval()
discriminator.eval()

DATA_DIR = '../latent_3d_points/data/shape_net_core_uniform_samples_2048/'
list_point_clouds = np.load('./list_point_noisy.npy')
list_point_clouds = list_point_clouds[:5000]


# In[14]:


X_train, X_test, _, _ = train_test_split(list_point_clouds, list_point_clouds, test_size=0.1, random_state=42)


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

class PointcloudDatasetNoisy(Dataset):
    def __init__(self, root, list_point_clouds):
        self.root = root
        self.list_files = list_point_clouds
        
    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        points = self.list_files[index]
        # points = np.array(points.points)
        points_normalized = (points - (-0.5)) / (0.5 - (-0.5))
        points = points_normalized.astype(np.float)
        points = torch.from_numpy(points)
        
        return points

ROOT_DIR = './rl_results/'
now =   str(datetime.datetime.now())

if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)

if not os.path.exists(ROOT_DIR + now):
    os.makedirs(ROOT_DIR + now)

OUTPUTS_DIR = ROOT_DIR  + now + '/outputs/'
if not os.path.exists(OUTPUTS_DIR):
    os.makedirs(OUTPUTS_DIR)

chamferloss = ChamferLoss(2048).to(device)

train_dataset = PointcloudDatasetNoisy(DATA_DIR, X_train)
train_dataloader = DataLoader(train_dataset, num_workers=0, shuffle=True, batch_size=1)

test_dataset = PointcloudDatasetNoisy(DATA_DIR, X_test)
test_dataloader = DataLoader(test_dataset, num_workers=0, shuffle=True, batch_size=1)

for i, data in enumerate(train_dataloader):
    data = data.permute([0,2,1])
    print(data.shape)
    break


for i, data in enumerate(train_dataloader):
    data = data.permute([0,2,1]).float().to(device)
            
    state_t = autoencoder.encode(data)

    optimal_action = ddpg.get_optimal_action(state_t).detach()
    new_state, _ = generator(optimal_action)
    
    out_data = autoencoder.decode(new_state)

    output = out_data[0,:,:]
    output = output.permute([1,0]).detach().cpu().numpy()

    inputt = data[0,:,:]
    inputt = inputt.permute([1,0]).detach().cpu().numpy()


    fig = plt.figure()
    ax_x = fig.add_subplot(111, projection='3d')
    x_ = inputt
    ax_x.scatter(x_[:, 0], x_[:, 1], x_[:,2])
    ax_x.set_xlim([0,1])
    ax_x.set_ylim([0,1])
    ax_x.set_zlim([0,1])
    fig.savefig(OUTPUTS_DIR+'/{}_{}.png'.format(i, 'in'))

    fig = plt.figure()
    ax_x = fig.add_subplot(111, projection='3d')
    x_ = output
    ax_x.scatter(x_[:, 0], x_[:, 1], x_[:,2])
    ax_x.set_xlim([0,1])
    ax_x.set_ylim([0,1])
    ax_x.set_zlim([0,1])
    fig.savefig(OUTPUTS_DIR+'/{}_{}_{}.png'.format(tsteps, i, 'rl_out'))

    output = autoencoder.decode(state_t)
    output = output[0,:,:]
    output = output.permute([1,0]).detach().cpu().numpy()

    fig = plt.figure()
    ax_x = fig.add_subplot(111, projection='3d')
    x_ = output
    ax_x.scatter(x_[:, 0], x_[:, 1], x_[:,2])
    ax_x.set_xlim([0,1])
    ax_x.set_ylim([0,1])
    ax_x.set_zlim([0,1])
    fig.savefig(OUTPUTS_DIR+'/{}_{}_{}.png'.format(tsteps, i, 'ae_out'))

    plt.close('all')

    if i > 2:
        break
