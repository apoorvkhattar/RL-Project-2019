# RL-Project-2019
Reimplementation of RL-GAN-Net: A Reinforcement Learning Agent Controlled GAN Network for Real-Time Point Cloud Shape Completion (https://arxiv.org/pdf/1904.12304v1.pdf)

## Dataset
The point cloud dataset is publicly available and can be downloaded from the following link: https://github.com/optas/latent_3d_points
In our experiments, we work on a subset of 5000 point clouds for which we have provided the numpy array containing the path for each point cloud (available here: https://drive.google.com/drive/folders/12tcCWpOsPM06u79WIyFt2uJfBkXh1ZcP?usp=sharing). After downloading the dataset, only the point clouds available in this numpy array have been used for training and testing.
'generate_noisy_point_clouds.py' creates a noisy point cloud for each. This is done by initializing a random seed index and removing 20 percent of points following the seed index.

## Training of Autoencoder
The complete code for dataloader, encoder-decoder models, chamfer loss and training is present in 'autoencoder.ipynb'. In the results folder, we have added the loss plots and some results of the trained autoencoder. 
The pre-trained weights of the autoencoder are available here: https://drive.google.com/drive/folders/1Rj8wpFTxW-OWuWVFdctQfCqA5pAfotuv?usp=sharing

## Training of GAN
The complete code for dataloader, generator, discriminator models and training is present in 'gan.ipynb'. In the original paper, the noise vector input to the generator is a 32-dimensional normal random variable however in our experiments we were able to reproduce the results with 5-dimensional noise vector. Some results and loss plots of both generator and discriminator are present in the results folder for reference.
The pre-trained weights of the generator and discriminator are available here: https://drive.google.com/drive/folders/1BnpPHKXBdtJrH2_Xxl0oA8m-sM8MlCVm?usp=sharing

## Training of RL-Agent
The complete code for actor, critic networks, DDPG and training code is present in the 'rl_agent_classes_only'. Some results and reward plots for the agent are present in the results folder for reference.
The pre-trained weights of the actor and critic are available here: https://drive.google.com/drive/folders/1lYKPY0sfpBpLYvMowh3MvBqYBO1mpi6N?usp=sharing
