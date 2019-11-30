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

np.random.seed(42)

DATA_DIR = '../latent_3d_points/data/shape_net_core_uniform_samples_2048/'
list_point_clouds = np.load('./list_point_subset.npy')

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

list_new_data = []
for i in tqdm.trange(len(list_point_clouds)):
    points = PyntCloud.from_file(list_point_clouds[i])
    points = np.array(points.points)
    seed_idx = int(np.random.rand() * 2048 * 0.8)
    points_removed = np.concatenate((points[:seed_idx, :], points[seed_idx + int(0.2*2048):, :]), axis=0)
    list_new_data.append(points_removed)

np.save('list_point_noisy.npy', list_new_data)
