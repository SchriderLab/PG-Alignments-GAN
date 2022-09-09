import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import torch
import copy

import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd

from prettytable import PrettyTable

def batched_bincount(x, dim, max_value):
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x, requires_grad = True)
    x = x.long()
    target.scatter_add_(dim, x, values)
    return target

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

# calculates the gradient_penalty for the loss function
# from https://arxiv.org/abs/1704.00028
def gp_gradient_penalty(discriminator, real_sites, real_pos, fake_sites, fake_pos, _lambda, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    #t = torch.FloatTensor(real_batch.shape[0], 1).uniform_(0,1).to(device)
    t = torch.FloatTensor(np.random.random((real_pos.size(0), 1, 1, 1))).to(device)
    interpolated_sites = t * real_sites + ((1-t) * fake_sites)
    
    t = t[:,:,:,0]
    interpolated_pos = t * real_pos + ((1 - t) * fake_pos)
    
    # define as variable to calculate gradient
    interpolated_sites = Variable(interpolated_sites, requires_grad=True)
    interpolated_pos = Variable(interpolated_pos, requires_grad=True)

    # calculate probabilities of interpolated examples
    prob_interpolated = discriminator(interpolated_sites, interpolated_pos)

    # calculate gradients
    gradients = autograd.grad(outputs=prob_interpolated, inputs=[interpolated_sites, interpolated_pos],
                              grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                              create_graph=True, retain_graph=True)[0]
    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * _lambda
    return grad_penalty

# from the 2018 paper: https://openaccess.thecvf.com/content_ECCV_2018/papers/Jiqing_Wu_Wasserstein_Divergence_For_ECCV_2018_paper.pdf
# copied and pasted from Pytorch-GAN github:  https://github.com/eriklindernoren/PyTorch-GAN
# (may want to check the validity of this one) (can't trust em!)
def div_gradient_penalty(d_loss_real, d_loss_fake, real_sites, real_pos, fake_sites, fake_pos, device):
    k = 2
    p = 6
    
    # Compute W-div gradient penalty
    real_grad_out = Variable(torch.Tensor(real_sites.size(0), 1).fill_(1.0), requires_grad=False).to(device)
    real_grad = autograd.grad(
        d_loss_real, [real_sites, real_pos], real_grad_out, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

    fake_grad_out = Variable(torch.Tensor(real_sites.size(0), 1).fill_(1.0), requires_grad=False).to(device)
    fake_grad = autograd.grad(
        d_loss_fake, [fake_sites, fake_pos], fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

    return torch.mean(real_grad_norm + fake_grad_norm) * k / 2

# a modified version of Will / Nick's original class for DC generator with two outputs
# where the position curve is output separately
# really two models here
class DC_Generator_Leaky_Sig(nn.Module):
    def __init__(self, data_size, latent_size, negative_slope, num_channels, k_size):
        super(DC_Generator_Leaky_Sig, self).__init__()
        if k_size == 2:
            pad = 0
        else:
            pad = 1
        self.negative_slope = negative_slope
        self.data_size = data_size
        self.twod_conv_layers = nn.Sequential(
            nn.ConvTranspose2d(latent_size, data_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(data_size * 8),
            #nn.ReLU(True),
            nn.LeakyReLU(negative_slope, inplace=True),
            # state size. (data_size*8) x 4 x 4
            nn.ConvTranspose2d(data_size * 8, data_size * 4, k_size, 2, pad, bias=False),
            nn.BatchNorm2d(data_size * 4),
            #nn.ReLU(True),
            nn.LeakyReLU(negative_slope, inplace=True),
            # state size. (data_size*4) x 8 x 8
            nn.ConvTranspose2d(data_size * 4, data_size * 2, k_size, 2, pad, bias=False),
            nn.BatchNorm2d(data_size * 2),
            #nn.ReLU(True),
            nn.LeakyReLU(negative_slope, inplace=True),
            # state size. (data_size*2) x 16 x 16
            nn.ConvTranspose2d(data_size * 2, data_size, k_size, 2, pad, bias=False),
            nn.BatchNorm2d(data_size),
            #nn.ReLU(True),
            nn.LeakyReLU(negative_slope, inplace=True),
            # state size. (data_size) x 32 x 32
            nn.ConvTranspose2d(data_size, num_channels, k_size, 2, pad, bias=False),
            # state size. (nc) x 64 x 64 (nc = 1 here, defined after data size)
            nn.Sigmoid()
        )
        
        self.oned_conv_layers = nn.Sequential(nn.ConvTranspose1d(latent_size, latent_size // 2, kernel_size = k_size, padding = 0, bias = False),
                                              # 4
                                              nn.BatchNorm1d(latent_size // 2), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              
                                              nn.ConvTranspose1d(latent_size // 2, latent_size // 4, kernel_size = k_size, stride = 2, padding = 1, bias = False),  
                                              # 8
                                              nn.BatchNorm1d(latent_size // 4), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              
                                              nn.ConvTranspose1d(latent_size // 4, latent_size // 8, kernel_size = k_size, stride = 2, padding = 1, bias = False),
                                              # 16
                                              nn.BatchNorm1d(latent_size // 8), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
        
                                              nn.ConvTranspose1d(latent_size // 8, latent_size // 16, kernel_size = k_size, stride = 2, padding = 1, bias = False),
                                              # 32
                                              nn.BatchNorm1d(latent_size // 16), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              
                                              nn.ConvTranspose1d(latent_size // 16, 1, kernel_size = k_size, stride = 2, padding = 1, bias = False),
                                              # 64
                                              nn.Sigmoid()
        )


    def forward(self, x):
        xs = self.twod_conv_layers(x)
        xp = self.oned_conv_layers(x[:,:,:,0])
        
        return xs, xp

class DC_Generator_Conditional_Leaky_Sig(nn.Module):
    def __init__(self, data_size, latent_size, negative_slope, num_channels, k_size, num_classes, embedding_size):
        super(DC_Generator_Conditional_Leaky_Sig, self).__init__()
        if k_size == 2:
            pad = 0
        else:
            pad = 1
        self.negative_slope = negative_slope
        self.data_size = data_size

        self.embedding = nn.Embedding(num_classes, embedding_size)

        self.twod_conv_layers = nn.Sequential(
            nn.ConvTranspose2d(latent_size+embedding_size, data_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(data_size * 8),
            #nn.ReLU(True),
            nn.LeakyReLU(negative_slope, inplace=True),
            # state size. (data_size*8) x 4 x 4
            nn.ConvTranspose2d(data_size * 8, data_size * 4, k_size, 2, pad, bias=False),
            nn.BatchNorm2d(data_size * 4),
            #nn.ReLU(True),
            nn.LeakyReLU(negative_slope, inplace=True),
            # state size. (data_size*4) x 8 x 8
            nn.ConvTranspose2d(data_size * 4, data_size * 2, k_size, 2, pad, bias=False),
            nn.BatchNorm2d(data_size * 2),
            #nn.ReLU(True),
            nn.LeakyReLU(negative_slope, inplace=True),
            # state size. (data_size*2) x 16 x 16
            nn.ConvTranspose2d(data_size * 2, data_size, k_size, 2, pad, bias=False),
            nn.BatchNorm2d(data_size),
            #nn.ReLU(True),
            nn.LeakyReLU(negative_slope, inplace=True),
            # state size. (data_size) x 32 x 32
            nn.ConvTranspose2d(data_size, num_channels, k_size, 2, pad, bias=False),
            # state size. (nc) x 64 x 64 (nc = 1 here, defined after data size)
            nn.Sigmoid()
        )
        
        self.oned_conv_layers = nn.Sequential(                                      
                                              nn.ConvTranspose1d((latent_size + embedding_size), (latent_size + embedding_size) // 2, kernel_size = k_size, padding = 0, bias = False),
                                              # 4
                                              nn.BatchNorm1d((latent_size + embedding_size) // 2), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              
                                              nn.ConvTranspose1d((latent_size + embedding_size) // 2, (latent_size + embedding_size) // 4, kernel_size = k_size, stride = 2, padding = 1, bias = False),  
                                              # 8
                                              nn.BatchNorm1d((latent_size + embedding_size) // 4), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              
                                              nn.ConvTranspose1d((latent_size + embedding_size) // 4, (latent_size + embedding_size) // 8, kernel_size = k_size, stride = 2, padding = 1, bias = False),
                                              # 16
                                              nn.BatchNorm1d((latent_size + embedding_size) // 8), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
        
                                              nn.ConvTranspose1d((latent_size + embedding_size) // 8, (latent_size + embedding_size) // 16, kernel_size = k_size, stride = 2, padding = 1, bias = False),
                                              # 32
                                              nn.BatchNorm1d((latent_size + embedding_size) // 16), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              
                                              nn.ConvTranspose1d((latent_size + embedding_size) // 16, 1, kernel_size = k_size, stride = 2, padding = 1, bias = False),
                                              # 64
                                              nn.Sigmoid()
        )


    def forward(self, x, label):
        label = self.embedding(label).reshape(x.shape[0],x.shape[1], 1,1)
        x = torch.cat([x, label], 1)

        xs = self.twod_conv_layers(x)
        xp = self.oned_conv_layers(x[:,:,:,0])
        
        return xs, xp

class DC_Generator_Leaky_Tan(nn.Module):
    def __init__(self, data_size, latent_size, negative_slope, num_channels, k_size):
        super(DC_Generator_Leaky_Tan, self).__init__()
        if k_size == 2:
            pad = 0
        else:
            pad = 1
        self.negative_slope = negative_slope
        self.data_size = data_size
        self.twod_conv_layers = nn.Sequential(
            nn.ConvTranspose2d(latent_size, data_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(data_size * 8),
            #nn.ReLU(True),
            nn.LeakyReLU(negative_slope, inplace=True),
            # state size. (data_size*8) x 4 x 4
            nn.ConvTranspose2d(data_size * 8, data_size * 4, k_size, 2, pad, bias=False),
            nn.BatchNorm2d(data_size * 4),
            #nn.ReLU(True),
            nn.LeakyReLU(negative_slope, inplace=True),
            # state size. (data_size*4) x 8 x 8
            nn.ConvTranspose2d(data_size * 4, data_size * 2, k_size, 2, pad, bias=False),
            nn.BatchNorm2d(data_size * 2),
            #nn.ReLU(True),
            nn.LeakyReLU(negative_slope, inplace=True),
            # state size. (data_size*2) x 16 x 16
            nn.ConvTranspose2d(data_size * 2, data_size, k_size, 2, pad, bias=False),
            nn.BatchNorm2d(data_size),
            #nn.ReLU(True),
            nn.LeakyReLU(negative_slope, inplace=True),
            # state size. (data_size) x 32 x 32
            nn.ConvTranspose2d(data_size, num_channels, k_size, 2, pad, bias=False),
            # state size. (nc) x 64 x 64 (nc = 1 here, defined after data size)
            nn.Tanh()
        )
        
        self.oned_conv_layers = nn.Sequential(nn.ConvTranspose1d(latent_size, latent_size // 2, kernel_size = k_size, padding = 0, bias = False),
                                              # 4
                                              nn.BatchNorm1d(latent_size // 2), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              
                                              nn.ConvTranspose1d(latent_size // 2, latent_size // 4, kernel_size = k_size, stride = 2, padding = 1, bias = False),  
                                              # 8
                                              nn.BatchNorm1d(latent_size // 4), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              
                                              nn.ConvTranspose1d(latent_size // 4, latent_size // 8, kernel_size = k_size, stride = 2, padding = 1, bias = False),
                                              # 16
                                              nn.BatchNorm1d(latent_size // 8), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
        
                                              nn.ConvTranspose1d(latent_size // 8, latent_size // 16, kernel_size = k_size, stride = 2, padding = 1, bias = False),
                                              # 32
                                              nn.BatchNorm1d(latent_size // 16), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              
                                              nn.ConvTranspose1d(latent_size // 16, 1, kernel_size = k_size, stride = 2, padding = 1, bias = False),
                                              # 64
                                              nn.Tanh()
        )


    def forward(self, x):
        xs = self.twod_conv_layers(x)
        xp = self.oned_conv_layers(x[:,:,:,0])
        
        return xs, xp

class DC_Generator_Gumbel(nn.Module):
    def __init__(self, data_size, latent_size, negative_slope, num_channels, k_size):
        super(DC_Generator_Gumbel, self).__init__()
        if k_size == 2:
            pad = 0
        else:
            pad = 1
        self.negative_slope = negative_slope
        self.data_size = data_size
        self.twod_conv_layers = nn.Sequential(
            nn.ConvTranspose2d(latent_size, data_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(data_size * 8),
            #nn.ReLU(True),
            nn.LeakyReLU(negative_slope, inplace=True),
            # state size. (data_size*8) x 4 x 4
            nn.ConvTranspose2d(data_size * 8, data_size * 4, k_size, 2, pad, bias=False),
            nn.BatchNorm2d(data_size * 4),
            #nn.ReLU(True),
            nn.LeakyReLU(negative_slope, inplace=True),
            # state size. (data_size*4) x 8 x 8
            nn.ConvTranspose2d(data_size * 4, data_size * 2, k_size, 2, pad, bias=False),
            nn.BatchNorm2d(data_size * 2),
            #nn.ReLU(True),
            nn.LeakyReLU(negative_slope, inplace=True),
            # state size. (data_size*2) x 16 x 16
            nn.ConvTranspose2d(data_size * 2, data_size, k_size, 2, pad, bias=False),
            nn.BatchNorm2d(data_size),
            #nn.ReLU(True),
            nn.LeakyReLU(negative_slope, inplace=True),
            # state size. (data_size) x 32 x 32
            nn.ConvTranspose2d(data_size, num_channels, k_size, 2, pad, bias=False),
            # state size. (nc) x 64 x 64 (nc = 1 here, defined after data size)
            #nn.Sigmoid()
        )
        
        self.oned_conv_layers = nn.Sequential(nn.ConvTranspose1d(latent_size, latent_size // 2, kernel_size = k_size, padding = 0, bias = False),
                                              # 4
                                              nn.BatchNorm1d(latent_size // 2), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              
                                              nn.ConvTranspose1d(latent_size // 2, latent_size // 4, kernel_size = k_size, stride = 2, padding = 1, bias = False),  
                                              # 8
                                              nn.BatchNorm1d(latent_size // 4), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              
                                              nn.ConvTranspose1d(latent_size // 4, latent_size // 8, kernel_size = k_size, stride = 2, padding = 1, bias = False),
                                              # 16
                                              nn.BatchNorm1d(latent_size // 8), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
        
                                              nn.ConvTranspose1d(latent_size // 8, latent_size // 16, kernel_size = k_size, stride = 2, padding = 1, bias = False),
                                              # 32
                                              nn.BatchNorm1d(latent_size // 16), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              
                                              nn.ConvTranspose1d(latent_size // 16, 1, kernel_size = k_size, stride = 2, padding = 1, bias = False),
                                              # 64
                                              nn.Sigmoid()
        )


    def forward(self, x):
        xs = torch.unsqueeze(F.gumbel_softmax(self.twod_conv_layers(x), dim = 1, hard = True)[:,1,:,:], dim=1)
        xp = self.oned_conv_layers(x[:,:,:,0])
        
        return xs, xp

class DC_Generator_Gumbel_Split(nn.Module):
    def __init__(self, data_size, latent_size, negative_slope, num_channels, k_size):
        super(DC_Generator_Gumbel_Split, self).__init__()
        if k_size == 2:
            pad = 0
        else:
            pad = 1
        self.negative_slope = negative_slope
        self.data_size = data_size
        self.twod_conv_layers = nn.Sequential(
            nn.ConvTranspose2d(latent_size, data_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(data_size * 8),
            #nn.ReLU(True),
            nn.LeakyReLU(negative_slope, inplace=True),
            # state size. (data_size*8) x 4 x 4
            nn.ConvTranspose2d(data_size * 8, data_size * 4, k_size, 2, pad, bias=False),
            nn.BatchNorm2d(data_size * 4),
            #nn.ReLU(True),
            nn.LeakyReLU(negative_slope, inplace=True),
            # state size. (data_size*4) x 8 x 8
            nn.ConvTranspose2d(data_size * 4, data_size * 2, k_size, 2, pad, bias=False),
            nn.BatchNorm2d(data_size * 2),
            #nn.ReLU(True),
            nn.LeakyReLU(negative_slope, inplace=True),
            # state size. (data_size*2) x 16 x 16
            nn.ConvTranspose2d(data_size * 2, data_size, k_size, 2, pad, bias=False),
            nn.BatchNorm2d(data_size),
            #nn.ReLU(True),
            nn.LeakyReLU(negative_slope, inplace=True),
            # state size. (data_size) x 32 x 32
            nn.ConvTranspose2d(data_size, num_channels, k_size, 2, pad, bias=False),
            # state size. (nc) x 64 x 64 (nc = 1 here, defined after data size)
            #nn.Sigmoid()
        )
        
        self.oned_conv_layers = nn.Sequential(nn.ConvTranspose1d(latent_size, latent_size // 2, kernel_size = k_size, padding = 0, bias = False),
                                              # 4
                                              nn.BatchNorm1d(latent_size // 2), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              
                                              nn.ConvTranspose1d(latent_size // 2, latent_size // 4, kernel_size = k_size, stride = 2, padding = 1, bias = False),  
                                              # 8
                                              nn.BatchNorm1d(latent_size // 4), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              
                                              nn.ConvTranspose1d(latent_size // 4, latent_size // 8, kernel_size = k_size, stride = 2, padding = 1, bias = False),
                                              # 16
                                              nn.BatchNorm1d(latent_size // 8), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
        
                                              nn.ConvTranspose1d(latent_size // 8, latent_size // 16, kernel_size = k_size, stride = 2, padding = 1, bias = False),
                                              # 32
                                              nn.BatchNorm1d(latent_size // 16), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              
                                              nn.ConvTranspose1d(latent_size // 16, 1, kernel_size = k_size, stride = 2, padding = 1, bias = False),
                                              # 64
                                              nn.Sigmoid()
        )


    def forward(self, x):
        xs = torch.unsqueeze(F.gumbel_softmax(self.twod_conv_layers(x), dim = 1, hard = True)[:,1,:,:], dim=1)
        xs_prob = torch.unsqueeze(F.gumbel_softmax(self.twod_conv_layers(x), dim = 1)[:,1,:,:], dim=1)
        xp = self.oned_conv_layers(x[:,:,:,0])
        
        return xs, xp, xs_prob


# 2D GAN discriminator for alignments
# modified version of Will and Nick's discriminator class where there are two inputs (sites and positions)
# their features are concatenated and thrown through a Linear layer for a final single Wasserstein output
class DC_Discriminator(nn.Module):
    def __init__(self, data_size, negative_slope, num_channels, k_size):
        super(DC_Discriminator, self).__init__()
        if k_size == 2:
            pad = 0
        else:
            pad = 1
        # input is (nc) x 64 x 64 (nc defined as 1 here at start, for a single 0/1 channel)
        self.negative_slope = negative_slope
        self.data_size = data_size
        self.twod_conv_layers = nn.Sequential(
            #GaussianNoise(0.1, 0),
            nn.Conv2d(num_channels, data_size, k_size, 2, pad, bias=False),
            nn.GroupNorm(1,data_size),
            nn.LeakyReLU(negative_slope, inplace=True),
            
            #nn.Dropout(p=0.2),
            # state size. (data_size) x 32 x 32
            #GaussianNoise(0.1, 0),
            nn.Conv2d(data_size, data_size * 2, k_size, 2, pad, bias=False),
            #nn.BatchNorm2d(data_size * 2),
            nn.GroupNorm(1,data_size*2),
            nn.LeakyReLU(negative_slope, inplace=True),
            #nn.Dropout(p=0.2),
            # state size. (data_size*2) x 16 x 16
            #GaussianNoise(0.1, 0),
            nn.Conv2d(data_size * 2, data_size * 4, k_size, 2, pad, bias=False),
            #nn.BatchNorm2d(data_size * 4),
            nn.GroupNorm(1,data_size*4),
            nn.LeakyReLU(negative_slope, inplace=True),
            
            #nn.Dropout(p=0.2),
            # state size. (data_size*4) x 8 x 8
            #GaussianNoise(0.1, 0),
            nn.Conv2d(data_size * 4, data_size * 8, k_size, 2, pad, bias=False),
            #nn.BatchNorm2d(data_size * 8),
            nn.GroupNorm(1, data_size * 8),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Flatten()
        )
        
        self.oned_conv_layers = nn.Sequential(nn.Conv1d(1, 16, kernel_size = k_size), 
                                              nn.GroupNorm(1, 16), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              nn.Conv1d(16, 32, kernel_size = k_size),  
                                              nn.GroupNorm(1, 32), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              nn.Conv1d(32, 64, kernel_size = k_size), 
                                              nn.GroupNorm(1, 64), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              nn.Flatten()
        )
        
        self.out = nn.Linear(11712, 1)
        
        #nn.Sequential(nn.Linear(11712, 4096), nn.LayerNorm((4096, )), nn.ReLU(), 
                                 #nn.Linear(4096, 1))

    def forward(self, xs, xp):
        xs = self.twod_conv_layers(xs)
        xp = self.oned_conv_layers(xp)
        
        xs = torch.cat([xs, xp], dim = -1)
        
        return self.out(xs)

class DC_Discriminator_Conditional(nn.Module):
    def __init__(self, data_size, negative_slope, num_channels, k_size, num_classes, embedding_size):
        super(DC_Discriminator_Conditional, self).__init__()
        if k_size == 2:
            pad = 0
        else:
            pad = 1
        # input is (nc) x 64 x 64 (nc defined as 1 here at start, for a single 0/1 channel)
        self.negative_slope = negative_slope
        self.data_size = data_size

        self.embedding = nn.Sequential(
            nn.Embedding(num_classes, embedding_size),
            nn.Flatten()
        )

        self.twod_conv_layers = nn.Sequential(
            #GaussianNoise(0.1, 0),
            nn.Conv2d(num_channels, data_size, k_size, 2, pad, bias=False),
            nn.GroupNorm(1,data_size),
            nn.LeakyReLU(negative_slope, inplace=True),
            
            #nn.Dropout(p=0.2),
            # state size. (data_size) x 32 x 32
            #GaussianNoise(0.1, 0),
            nn.Conv2d(data_size, data_size * 2, k_size, 2, pad, bias=False),
            #nn.BatchNorm2d(data_size * 2),
            nn.GroupNorm(1,data_size*2),
            nn.LeakyReLU(negative_slope, inplace=True),
            #nn.Dropout(p=0.2),
            # state size. (data_size*2) x 16 x 16
            #GaussianNoise(0.1, 0),
            nn.Conv2d(data_size * 2, data_size * 4, k_size, 2, pad, bias=False),
            #nn.BatchNorm2d(data_size * 4),
            nn.GroupNorm(1,data_size*4),
            nn.LeakyReLU(negative_slope, inplace=True),
            
            #nn.Dropout(p=0.2),
            # state size. (data_size*4) x 8 x 8
            #GaussianNoise(0.1, 0),
            nn.Conv2d(data_size * 4, data_size * 8, k_size, 2, pad, bias=False),
            #nn.BatchNorm2d(data_size * 8),
            nn.GroupNorm(1, data_size * 8),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Flatten()
        )
        
        self.oned_conv_layers = nn.Sequential(nn.Conv1d(1, 16, kernel_size = k_size), 
                                              nn.GroupNorm(1, 16), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              nn.Conv1d(16, 32, kernel_size = k_size),  
                                              nn.GroupNorm(1, 32), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              nn.Conv1d(32, 64, kernel_size = k_size), 
                                              nn.GroupNorm(1, 64), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              nn.Flatten()
        )
        
        self.out = nn.Linear(11712 + embedding_size, 1)
        
        #nn.Sequential(nn.Linear(11712, 4096), nn.LayerNorm((4096, )), nn.ReLU(), 
                                 #nn.Linear(4096, 1))

    def forward(self, xs, xp, label):

        label = self.embedding(label)#.reshape(xs.shape[0],128, 1,1)
        #label = torch.flatten(label)

        xs = self.twod_conv_layers(xs)
        xp = self.oned_conv_layers(xp)
        
        xs = torch.cat([xs, xp], dim = -1)
        xsl = torch.cat([xs, label], dim = -1)
        
        return self.out(xsl)

class DC_Discriminator_Conditional_2(nn.Module):
    def __init__(self, data_size, negative_slope, num_channels, k_size, num_classes, embedding_size):
        super(DC_Discriminator_Conditional_2, self).__init__()
        if k_size == 2:
            pad = 0
        else:
            pad = 1
        # input is (nc) x 64 x 64 (nc defined as 1 here at start, for a single 0/1 channel)
        self.negative_slope = negative_slope
        self.data_size = data_size

        self.embedding = nn.Sequential(
            nn.Embedding(num_classes, embedding_size),
            #nn.Flatten()
        )

        self.twod_conv_layers = nn.Sequential(
            #GaussianNoise(0.1, 0),
            nn.Conv2d(num_channels+1, data_size, k_size, 2, pad, bias=False),
            nn.GroupNorm(1,data_size),
            nn.LeakyReLU(negative_slope, inplace=True),
            
            #nn.Dropout(p=0.2),
            # state size. (data_size) x 32 x 32
            #GaussianNoise(0.1, 0),
            nn.Conv2d(data_size, data_size * 2, k_size, 2, pad, bias=False),
            #nn.BatchNorm2d(data_size * 2),
            nn.GroupNorm(1,data_size*2),
            nn.LeakyReLU(negative_slope, inplace=True),
            #nn.Dropout(p=0.2),
            # state size. (data_size*2) x 16 x 16
            #GaussianNoise(0.1, 0),
            nn.Conv2d(data_size * 2, data_size * 4, k_size, 2, pad, bias=False),
            #nn.BatchNorm2d(data_size * 4),
            nn.GroupNorm(1,data_size*4),
            nn.LeakyReLU(negative_slope, inplace=True),
            
            #nn.Dropout(p=0.2),
            # state size. (data_size*4) x 8 x 8
            #GaussianNoise(0.1, 0),
            nn.Conv2d(data_size * 4, data_size * 8, k_size, 2, pad, bias=False),
            #nn.BatchNorm2d(data_size * 8),
            nn.GroupNorm(1, data_size * 8),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Flatten()
        )
        
        self.oned_conv_layers = nn.Sequential(nn.Conv1d(1, 16, kernel_size = k_size), 
                                              nn.GroupNorm(1, 16), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              nn.Conv1d(16, 32, kernel_size = k_size),  
                                              nn.GroupNorm(1, 32), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              nn.Conv1d(32, 64, kernel_size = k_size), 
                                              nn.GroupNorm(1, 64), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              nn.Flatten()
        )
        
        self.out = nn.Linear(11712, 1)
        
        #nn.Sequential(nn.Linear(11712, 4096), nn.LayerNorm((4096, )), nn.ReLU(), 
                                 #nn.Linear(4096, 1))

    def forward(self, xs, xp, label):
        label = self.embedding(label).reshape(xs.shape[0],1,64,64)
        #print(label.shape)#.reshape(xs.shape[0],1,64,64)
        #label = torch.flatten(label)
        xs = torch.cat([xs,label], dim = 1)
        xs = self.twod_conv_layers(xs)
        xp = self.oned_conv_layers(xp)
        
        xs = torch.cat([xs, xp], dim = -1)
        #xsl = torch.cat([xs, label], dim = -1)
        
        return self.out(xs)


class DC_Discriminator_SN(nn.Module):
    def __init__(self, data_size, negative_slope, num_channels, k_size):
        super(DC_Discriminator_SN, self).__init__()
        if k_size == 2:
            pad = 0
        else:
            pad = 1
        # input is (nc) x 64 x 64 (nc defined as 1 here at start, for a single 0/1 channel)
        self.negative_slope = negative_slope
        self.data_size = data_size
        self.twod_conv_layers = nn.Sequential(
            #GaussianNoise(0.1, 0),
            nn.Conv2d(num_channels, data_size, k_size, 2, pad, bias=False),
            nn.GroupNorm(1,data_size),
            nn.LeakyReLU(negative_slope, inplace=True),
            
            #nn.Dropout(p=0.2),
            # state size. (data_size) x 32 x 32
            #GaussianNoise(0.1, 0),
            nn.Conv2d(data_size, data_size * 2, k_size, 2, pad, bias=False),
            #nn.BatchNorm2d(data_size * 2),
            nn.GroupNorm(1,data_size*2),
            nn.LeakyReLU(negative_slope, inplace=True),
            #nn.Dropout(p=0.2),
            # state size. (data_size*2) x 16 x 16
            #GaussianNoise(0.1, 0),
            nn.Conv2d(data_size * 2, data_size * 4, k_size, 2, pad, bias=False),
            #nn.BatchNorm2d(data_size * 4),
            nn.GroupNorm(1,data_size*4),
            nn.LeakyReLU(negative_slope, inplace=True),
            
            #nn.Dropout(p=0.2),
            # state size. (data_size*4) x 8 x 8
            #GaussianNoise(0.1, 0),
            nn.Conv2d(data_size * 4, data_size * 8, k_size, 2, pad, bias=False),
            #nn.BatchNorm2d(data_size * 8),
            nn.GroupNorm(1, data_size * 8),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Flatten()
        )
        
        self.oned_conv_layers = nn.Sequential(nn.Conv1d(1, 16, kernel_size = k_size), 
                                              nn.GroupNorm(1, 16), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              nn.Conv1d(16, 32, kernel_size = k_size),  
                                              nn.GroupNorm(1, 32), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              nn.Conv1d(32, 64, kernel_size = k_size), 
                                              nn.GroupNorm(1, 64), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              nn.Flatten()
        )
        
        self.out = spectral_norm(nn.Linear(11712, 1))
        
        #nn.Sequential(nn.Linear(11712, 4096), nn.LayerNorm((4096, )), nn.ReLU(), 
                                 #nn.Linear(4096, 1))

    def forward(self, xs, xp):
        xs = self.twod_conv_layers(xs)
        xp = self.oned_conv_layers(xp)
        
        xs = torch.cat([xs, xp], dim = -1)
        
        return self.out(xs)



class DC_Discriminator_Stats(nn.Module):
    def __init__(self, data_size, negative_slope, num_channels, k_size):
        super(DC_Discriminator_Stats, self).__init__()
        if k_size == 2:
            pad = 0
        else:
            pad = 1
        # input is (nc) x 64 x 64 (nc defined as 1 here at start, for a single 0/1 channel)
        self.negative_slope = negative_slope
        self.data_size = data_size
        self.twod_conv_layers = nn.Sequential(
            #GaussianNoise(0.1, 0),
            nn.Conv2d(num_channels, data_size, k_size, 2, pad, bias=False),
            nn.GroupNorm(1,data_size),
            nn.LeakyReLU(negative_slope, inplace=True),
            
            #nn.Dropout(p=0.2),
            # state size. (data_size) x 32 x 32
            #GaussianNoise(0.1, 0),
            nn.Conv2d(data_size, data_size * 2, k_size, 2, pad, bias=False),
            #nn.BatchNorm2d(data_size * 2),
            nn.GroupNorm(1,data_size*2),
            nn.LeakyReLU(negative_slope, inplace=True),
            #nn.Dropout(p=0.2),
            # state size. (data_size*2) x 16 x 16
            #GaussianNoise(0.1, 0),
            nn.Conv2d(data_size * 2, data_size * 4, k_size, 2, pad, bias=False),
            #nn.BatchNorm2d(data_size * 4),
            nn.GroupNorm(1,data_size*4),
            nn.LeakyReLU(negative_slope, inplace=True),
            
            #nn.Dropout(p=0.2),
            # state size. (data_size*4) x 8 x 8
            #GaussianNoise(0.1, 0),
            nn.Conv2d(data_size * 4, data_size * 8, k_size, 2, pad, bias=False),
            #nn.BatchNorm2d(data_size * 8),
            nn.GroupNorm(1, data_size * 8),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Flatten()
        )
        
        self.oned_conv_layers = nn.Sequential(nn.Conv1d(1, 16, kernel_size = k_size), 
                                              nn.GroupNorm(1, 16), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              nn.Conv1d(16, 32, kernel_size = k_size),  
                                              nn.GroupNorm(1, 32), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              nn.Conv1d(32, 64, kernel_size = k_size), 
                                              nn.GroupNorm(1, 64), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              nn.Flatten()
        )
        
        self.out = nn.Linear(11777, 1)
        
        #nn.Sequential(nn.Linear(11712, 4096), nn.LayerNorm((4096, )), nn.ReLU(), 
                                 #nn.Linear(4096, 1))

    def forward(self, xs, xp):
        xs_conv = self.twod_conv_layers(xs)
        xp = self.oned_conv_layers(xp)

        ## stats on the image columns to approximate the SFS, each adds data_size (probably 64) nodes to the linear output##
        #xs_mean = torch.mean(xs, dim = 2)

        xs_sum = torch.sum(xs, dim = 2)
        #xs_sum.requires_grad = True
        xs_sfs = torch.cat([batched_bincount(x,1,65) for x in xs_sum],0).reshape(xs.shape[0],1,65)
        
        #xs_max = torch.max(xs, dim = 2)[0]
        #xs_min = torch.min(xs, dim = 2)[0]
        #xs_var = torch.var(xs, dim = 2, unbiased = False)
        
        #xs_stats = torch.cat([xs_mean, xs_max, xs_min, xs_var], dim = 1)
        xs_stats = xs_sfs
        xs_stats = xs_stats.flatten(1)
        
        xs = torch.cat([xs_conv, xp, xs_stats], dim = -1)
        
        return self.out(xs)

class DC_Discriminator_Soft_SFS(nn.Module):
    def __init__(self, data_size, negative_slope, num_channels, k_size):
        super(DC_Discriminator_Soft_SFS, self).__init__()
        if k_size == 2:
            pad = 0
        else:
            pad = 1
        # input is (nc) x 64 x 64 (nc defined as 1 here at start, for a single 0/1 channel)
        self.negative_slope = negative_slope
        self.data_size = data_size
        self.twod_conv_layers = nn.Sequential(
            #GaussianNoise(0.1, 0),
            nn.Conv2d(num_channels, data_size, k_size, 2, pad, bias=False),
            nn.GroupNorm(1,data_size),
            nn.LeakyReLU(negative_slope, inplace=True),
            
            #nn.Dropout(p=0.2),
            # state size. (data_size) x 32 x 32
            #GaussianNoise(0.1, 0),
            nn.Conv2d(data_size, data_size * 2, k_size, 2, pad, bias=False),
            #nn.BatchNorm2d(data_size * 2),
            nn.GroupNorm(1,data_size*2),
            nn.LeakyReLU(negative_slope, inplace=True),
            #nn.Dropout(p=0.2),
            # state size. (data_size*2) x 16 x 16
            #GaussianNoise(0.1, 0),
            nn.Conv2d(data_size * 2, data_size * 4, k_size, 2, pad, bias=False),
            #nn.BatchNorm2d(data_size * 4),
            nn.GroupNorm(1,data_size*4),
            nn.LeakyReLU(negative_slope, inplace=True),
            
            #nn.Dropout(p=0.2),
            # state size. (data_size*4) x 8 x 8
            #GaussianNoise(0.1, 0),
            nn.Conv2d(data_size * 4, data_size * 8, k_size, 2, pad, bias=False),
            #nn.BatchNorm2d(data_size * 8),
            nn.GroupNorm(1, data_size * 8),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Flatten()
        )
        
        self.oned_conv_layers = nn.Sequential(nn.Conv1d(1, 16, kernel_size = k_size), 
                                              nn.GroupNorm(1, 16), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              nn.Conv1d(16, 32, kernel_size = k_size),  
                                              nn.GroupNorm(1, 32), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              nn.Conv1d(32, 64, kernel_size = k_size), 
                                              nn.GroupNorm(1, 64), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              nn.Flatten()
        )
        
        self.out = nn.Linear(11777, 1)
        
        self.alpha = 0.1
        
        self.I = nn.Parameter(torch.FloatTensor(np.ones((64 + 1, 1))), requires_grad = False)
        self.beta = nn.Parameter(torch.FloatTensor(np.array(range(64 + 1), dtype = np.float32).reshape(64 + 1, 1)), requires_grad = False)
        
    def compute_sfs_soft(self, im):
        eps = 1e-5
        
        x = torch.sum(im, dim = 2)
        batch_size = x.shape[0]
        n_sites = x.shape[-1]
        
        # (batch, 1, n_sites) -> (batch, n_sites, 1) -> (batch * n_sites, 1)
        x = torch.flatten(x.transpose(1, 2), 0, 1)
        
        # (batch * n_sites, 1) -> (batch * n_sites, n_ind + 1)
        # softmaxed version of each site sum
        x = torch.exp(-1 * self.alpha * torch.pow((torch.matmul(self.I, x.transpose(0, 1)) - self.beta + eps).transpose(0, 1), 2))
        x = x / torch.unsqueeze(x.sum(dim = 1), 1)
        
        x = x.view(batch_size, n_sites, x.shape[-1])
        
        # (batch, n_sites, n_ind + 1) -> ~ SFS (batch, n_ind + 1)
        x = torch.sum(x, dim = 1)
        
        return x

    def forward(self, xs, xp):
        xs_conv = self.twod_conv_layers(xs)
        xp = self.oned_conv_layers(xp)

        ## stats on the image columns to approximate the SFS, each adds data_size (probably 64) nodes to the linear output##
        #xs_mean = torch.mean(xs, dim = 2)

        #xs_sum = torch.sum(xs, dim = 2)
        #xs_sum.requires_grad = True
        xs_sfs = self.compute_sfs_soft(xs)
        
        #xs_max = torch.max(xs, dim = 2)[0]
        #xs_min = torch.min(xs, dim = 2)[0]
        #xs_var = torch.var(xs, dim = 2, unbiased = False)
        
        #xs_stats = torch.cat([xs_mean, xs_max, xs_min, xs_var], dim = 1)
        xs_stats = xs_sfs
        xs_stats = xs_stats.flatten(1)
        
        xs = torch.cat([xs_conv, xp, xs_stats], dim = -1)
        
        return self.out(xs)

class DC_Discriminator_Stats_Up(nn.Module):
    def __init__(self, data_size, negative_slope, num_channels, k_size):
        super(DC_Discriminator_Stats_Up, self).__init__()
        if k_size == 2:
            pad = 0
        else:
            pad = 1
        # input is (nc) x 64 x 64 (nc defined as 1 here at start, for a single 0/1 channel)
        self.negative_slope = negative_slope
        self.data_size = data_size
        self.twod_conv_layers = nn.Sequential(
            #GaussianNoise(0.1, 0),
            nn.Conv2d(num_channels, data_size, k_size, 2, pad, bias=False),
            nn.GroupNorm(1,data_size),
            nn.LeakyReLU(negative_slope, inplace=True),
            
            # state size. (data_size) x 32 x 32
            nn.Conv2d(data_size, data_size * 2, k_size, 2, pad, bias=False),
            nn.GroupNorm(1,data_size*2),
            nn.LeakyReLU(negative_slope, inplace=True),

            # state size. (data_size*2) x 16 x 16
            nn.Conv2d(data_size * 2, data_size * 4, k_size, 2, pad, bias=False),
            nn.GroupNorm(1,data_size*4),
            nn.LeakyReLU(negative_slope, inplace=True),
            
            # state size. (data_size*4) x 8 x 8
            nn.Conv2d(data_size * 4, data_size * 8, k_size, 2, pad, bias=False),
            nn.GroupNorm(1, data_size * 8),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Flatten()
        )
        
        self.oned_conv_layers = nn.Sequential(nn.Conv1d(1, 16, kernel_size = k_size), 
                                              nn.GroupNorm(1, 16), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              nn.Conv1d(16, 32, kernel_size = k_size),  
                                              nn.GroupNorm(1, 32), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              nn.Conv1d(32, 64, kernel_size = k_size), 
                                              nn.GroupNorm(1, 64), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              nn.Flatten()
        )
        
        self.up_stats_mlp = nn.ModuleList()
        for k in range(4):
            self.up_stats_mlp.append(MLP_Basic(data_size, data_size * 4, data_size * 2))

        self.out = nn.Linear(12736, 1)
        
        #nn.Sequential(nn.Linear(11712, 4096), nn.LayerNorm((4096, )), nn.ReLU(), 
                                 #nn.Linear(4096, 1))

    def forward(self, xs, xp):
        xs_conv = self.twod_conv_layers(xs)
        xp = self.oned_conv_layers(xp)

        ## stats on the image columns to approximate the SFS, each adds data_size (probably 64) nodes to the linear output##
        xs_mean = self.up_stats_mlp[0](torch.mean(xs, dim = 2))
        xs_max = self.up_stats_mlp[1](torch.max(xs, dim = 2)[0])
        xs_min = self.up_stats_mlp[2](torch.min(xs, dim = 2)[0])
        xs_var = self.up_stats_mlp[3](torch.var(xs, dim = 2, unbiased = False))


        xs_stats = torch.cat([xs_mean, xs_max, xs_min, xs_var], dim = 1)
        xs_stats = xs_stats.flatten(1)
        
        xs = torch.cat([xs_conv, xp, xs_stats], dim = -1)
        
        return self.out(xs)

class DC_Discriminator_Stats_Up_Feat_Down(nn.Module):
    def __init__(self, data_size, negative_slope, num_channels, k_size):
        super(DC_Discriminator_Stats_Up_Feat_Down, self).__init__()
        if k_size == 2:
            pad = 0
        else:
            pad = 1
        # input is (nc) x 64 x 64 (nc defined as 1 here at start, for a single 0/1 channel)
        self.negative_slope = negative_slope
        self.data_size = data_size
        self.twod_conv_layers = nn.Sequential(
            #GaussianNoise(0.1, 0),
            nn.Conv2d(num_channels, data_size, k_size, 2, pad, bias=False),
            nn.GroupNorm(1,data_size),
            nn.LeakyReLU(negative_slope, inplace=True),
            
            # state size. (data_size) x 32 x 32
            nn.Conv2d(data_size, data_size * 2, k_size, 2, pad, bias=False),
            nn.GroupNorm(1,data_size*2),
            nn.LeakyReLU(negative_slope, inplace=True),

            # state size. (data_size*2) x 16 x 16
            nn.Conv2d(data_size * 2, data_size * 4, k_size, 2, pad, bias=False),
            nn.GroupNorm(1,data_size*4),
            nn.LeakyReLU(negative_slope, inplace=True),
            
            # state size. (data_size*4) x 8 x 8
            nn.Conv2d(data_size * 4, data_size * 8, k_size, 2, pad, bias=False),
            nn.GroupNorm(1, data_size * 8),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Flatten()
        )
        
        self.oned_conv_layers = nn.Sequential(nn.Conv1d(1, 16, kernel_size = k_size), 
                                              nn.GroupNorm(1, 16), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              nn.Conv1d(16, 32, kernel_size = k_size),  
                                              nn.GroupNorm(1, 32), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              nn.Conv1d(32, 64, kernel_size = k_size), 
                                              nn.GroupNorm(1, 64), 
                                              nn.LeakyReLU(negative_slope, inplace=True),
                                              nn.Flatten()
        )
        
        self.downsample_site_features = nn.Sequential(nn.Linear(8192, 2048))#, 
                                                      #nn.GroupNorm(256, 4096), 
                                                      #nn.LeakyReLU(negative_slope, inplace=True))

        self.downsample_position_features = nn.Sequential(nn.Linear(3520, 880))#, 
                                                          #nn.GroupNorm(110,1760), 
                                                          #nn.LeakyReLU(negative_slope, inplace=True))

        self.up_stats_mlp = nn.ModuleList()
        for k in range(4):
            self.up_stats_mlp.append(MLP_Basic(data_size, data_size * 4, data_size * 2)) 

        self.out = nn.Linear(3184, 1)
        
        #nn.Sequential(nn.Linear(11712, 4096), nn.LayerNorm((4096, )), nn.ReLU(), 
                                 #nn.Linear(4096, 1))

    def forward(self, xs, xp):
        xs_conv = self.twod_conv_layers(xs)
        xs_conv_down = self.downsample_site_features(xs_conv)

        xp = self.oned_conv_layers(xp)
        xp_down = self.downsample_position_features(xp)

        xs_mean = torch.mean(xs, dim = 2)       
        xs_max = torch.max(xs, dim = 2)[0]
        xs_min = torch.min(xs, dim = 2)[0]
        xs_var = torch.var(xs, dim = 2, unbiased = False)

        ## stats on the image columns to approximate the SFS, each adds data_size (probably 64) nodes to the linear output##
        #xs_mean = self.up_stats_mlp[0](torch.mean(xs, dim = 2))
        #xs_max = self.up_stats_mlp[1](torch.max(xs, dim = 2)[0])
        #xs_min = self.up_stats_mlp[2](torch.min(xs, dim = 2)[0])
        #xs_var = self.up_stats_mlp[3](torch.var(xs, dim = 2, unbiased = False))


        xs_stats = torch.cat([xs_mean, xs_max, xs_min, xs_var], dim = 1)
        xs_stats = xs_stats.flatten(1)
        
        xs = torch.cat([xs_conv_down, xp_down, xs_stats], dim = -1)
        
        return self.out(xs)        



class LeakyMLP_GroupNorm(nn.Module):
    def __init__(self, input_dim, output_dim, dim=256, n_blk=3):
        super(LeakyMLP_GroupNorm, self).__init__()
        layers = [nn.Linear(input_dim, dim), nn.GroupNorm(1,dim), nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(n_blk - 2):
            layers += [nn.Linear(dim, dim), nn.GroupNorm(1,dim), nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Linear(dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

class MLP_Basic(nn.Module):
    def __init__(self, input_dim, output_dim, dim=256, n_blk=3):
        super(MLP_Basic, self).__init__()
        layers = [nn.Linear(input_dim, dim)]
        for _ in range(n_blk - 2):
            layers += [nn.Linear(dim, dim)]
        layers += [nn.Linear(dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

class ResidualBlock_B(nn.Module):
    def __init__(self, in_features, out_features, norm="in"):
        super(ResidualBlock_B, self).__init__()

        norm_layer = nn.BatchNorm2d

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, out_features, 3),
            norm_layer(out_features),
            nn.LeakyReLU(negative_slope = 0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_features, out_features, 3),
            norm_layer(out_features),
            
        )

    def forward(self, x):
        return x + self.block(x)

# 2d Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, norm="in"):
        super(ResidualBlock, self).__init__()

        norm_layer = nn.InstanceNorm2d

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, out_features, 3),
            norm_layer(out_features),
            nn.LeakyReLU(negative_slope = 0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_features, out_features, 3),
            norm_layer(out_features),
            
        )

    def forward(self, x):
        return x + self.block(x)
    
# 2d Residual Block
class ResidualBlock_(nn.Module):
    def __init__(self, in_features, out_features, norm="in"):
        super(ResidualBlock_B_, self).__init__()

        norm_layer = nn.InstanceNorm2d

        self.block = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3, padding = 1),
            norm_layer(out_features),
            nn.LeakyReLU(negative_slope = 0.2, inplace=True),
            nn.Conv2d(out_features, out_features, 3, padding = 1),
            norm_layer(out_features),
            
        )

    def forward(self, x):
        return x + self.block(x)

class ResidualBlock_B_(nn.Module):
    def __init__(self, in_features, out_features, norm="in"):
        super(ResidualBlock_B_, self).__init__()

        norm_layer = nn.GroupNorm

        self.block = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3, padding = 1),
            norm_layer(1,out_features),
            nn.LeakyReLU(negative_slope = 0.2, inplace=True),
            nn.Conv2d(out_features, out_features, 3, padding = 1),
            norm_layer(1,out_features),
            
        )

    def forward(self, x):
        return x + self.block(x)
    
# 2d Residual Block
class ResidualBlock1d_B_(nn.Module):
    def __init__(self, in_features, out_features, norm="in"):
        super(ResidualBlock1d_, self).__init__()

        norm_layer = nn.GroupNorm

        self.block = nn.Sequential(
            nn.Conv2d(in_features, out_features, (1, 3), stride = (1, 1), padding = (0, 1)),
            norm_layer(1,out_features),
            nn.LeakyReLU(negative_slope = 0.2, inplace=True),
            nn.Conv2d(out_features, out_features, (1, 3), stride = (1, 1), padding = (0, 1)),
            norm_layer(1,out_features),
            
        )

    def forward(self, x):
        return x + self.block(x)

class ResidualBlock1d_(nn.Module):
    def __init__(self, in_features, out_features, norm="in"):
        super(ResidualBlock1d_, self).__init__()

        norm_layer = nn.InstanceNorm1d

        self.block = nn.Sequential(
            nn.Conv2d(in_features, out_features, (1, 3), stride = (1, 1), padding = (0, 1)),
            norm_layer(out_features),
            nn.LeakyReLU(negative_slope = 0.2, inplace=True),
            nn.Conv2d(out_features, out_features, (1, 3), stride = (1, 1), padding = (0, 1)),
            norm_layer(out_features),
            
        )

    def forward(self, x):
        return x + self.block(x)

# 1d residual block
class ResidualBlock1d(nn.Module):
    def __init__(self, in_features, out_features, norm="in"):
        super(ResidualBlock1d, self).__init__()

        norm_layer = nn.InstanceNorm1d

        self.block = nn.Sequential(
            nn.ReflectionPad1d(1),
            nn.Conv1d(in_features, out_features, 3),
            norm_layer(out_features),
            nn.LeakyReLU(negative_slope = 0.2, inplace=True),
            nn.ReflectionPad1d(1),
            nn.Conv1d(out_features, out_features, 3),
            norm_layer(out_features),
        )
        
    def forward(self, x):
        return x + self.block(x)

class ResidualBlock1d_B(nn.Module):
    def __init__(self, in_features, out_features, norm="in"):
        super(ResidualBlock1d_B, self).__init__()

        norm_layer = nn.BatchNorm1d

        self.block = nn.Sequential(
            nn.ReflectionPad1d(1),
            nn.Conv1d(in_features, out_features, 3),
            norm_layer(out_features),
            nn.LeakyReLU(negative_slope = 0.2, inplace=True),
            nn.ReflectionPad1d(1),
            nn.Conv1d(out_features, out_features, 3),
            norm_layer(out_features),
        )
        
    def forward(self, x):
        return x + self.block(x)
    
class ResidualBlock1d_pad_regular(nn.Module):
    def __init__(self, in_features, out_features, norm="in"):
        super(ResidualBlock1d_pad_regular, self).__init__()

        norm_layer = nn.InstanceNorm1d

        self.block = nn.Sequential(
            nn.Conv1d(in_features, out_features, 3, padding = 1),
            norm_layer(out_features),
            nn.LeakyReLU(negative_slope = 0.2, inplace=True),
            nn.Conv1d(out_features, out_features, 3, padding = 1),
            norm_layer(out_features),
        )
        
    def forward(self, x):
        return x + self.block(x)

class ResidualBlock1d_pad_regular_B(nn.Module):
    def __init__(self, in_features, out_features, norm="in"):
        super(ResidualBlock1d_pad_regular_B, self).__init__()

        norm_layer = nn.GroupNorm

        self.block = nn.Sequential(
            nn.Conv1d(in_features, out_features, 3, padding = 1),
            norm_layer(1,out_features),
            nn.LeakyReLU(negative_slope = 0.2, inplace=True),
            nn.Conv1d(out_features, out_features, 3, padding = 1),
            norm_layer(1,out_features),
        )
        
    def forward(self, x):
        return x + self.block(x)
    
class AdaptiveInstanceNorm2d(nn.Module):
    """Reference: https://github.com/NVlabs/MUNIT/blob/master/networks.py"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c, h, w = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, h, w)

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps
        )

        return out.view(b, c, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"

# a basic MLP module
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim=256, n_blk=3, activ="relu"):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, dim), nn.LayerNorm((dim,)), nn.ReLU(inplace=True)]
        for _ in range(n_blk - 2):
            layers += [nn.Linear(dim, dim), nn.LayerNorm((dim,)), nn.ReLU(inplace=True)]
        layers += [nn.Linear(dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

# a basic MLP module (no norm or activations)
class MLP_B(nn.Module):
    def __init__(self, input_dim, output_dim, dim=256, n_blk=3, activ="relu"):
        super(MLP_B, self).__init__()
        layers = [nn.Linear(input_dim, dim)]
        for _ in range(n_blk - 2):
            layers += [nn.Linear(dim, dim)]
        layers += [nn.Linear(dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

class init_lin(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(init_lin, self).__init__()
        self.model = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
    
class ResidualGenerator1dV01(nn.Module):
    def __init__(self, latent_dim = 128):
        super(ResidualGenerator1dV01, self).__init__()
        
        # upsampling linear layer
        self.linear = MLP(latent_dim, 256 * 4, 512)
        
        self.start_size = [256, 4]
        
        channels = [256, 128, 64, 32, 16]
        
        s = 8
        
        _ = []
        for k in range(len(channels) - 1):
            _.append(ResidualBlock1d(channels[k], channels[k]))
            #_.append(nn.ConvTranspose1d(channels[k], channels[k + 1], 2, stride = 2))
            #_.append(nn.Conv2d(channels[k], channels[k+1], 5, stride=1, padding=2))
            _.append(nn.Upsample(scale_factor = 2))
            _.append(nn.Conv1d(channels[k], channels[k+1], 5, stride=1, padding=2))
            _.append(nn.LayerNorm((channels[k+1], s)))
            #_.append(nn.ReLU(inplace = True))
            
            s *= 2

        _.append(nn.Conv1d(channels[-1], 1, 1))
        #_.append(nn.Sigmoid())
            
        self.decoder = nn.Sequential(*_)
        
    def forward(self, z):
        im = self.linear(z).view(*([z.shape[0]] + self.start_size))
        im = self.decoder(im)
        
        return im
    
class ResGRUHead(nn.Module):
    def __init__(self, n = 64, s = 64, in_channels = 8, hidden_size = 128, bidirectional = False, n_layers = 1):
        super(ResGRUHead, self).__init__()
        
        self.bidirectional = bidirectional
        
        # (N, L, C)
        self.att = nn.Parameter(torch.ones((1, 1, s)) * 0.5, requires_grad = True)
        
        layers = []        
        layers += [
            ResidualBlock1d_(in_channels, in_channels),
            nn.Conv2d(in_channels, 1, (1, 1), stride = (1, 1)),
            nn.LayerNorm((1, n, s)),
        ]
        
        self.res_block = nn.Sequential(*layers)
        self.gru = nn.GRU(s, hidden_size, batch_first = True, bidirectional = bidirectional, num_layers = n_layers)
        self.gru_norm = nn.LayerNorm((128, ))
        
    def forward(self, x):
        x = torch.squeeze(self.res_block(x))
        x = x * torch.nn.functional.softmax(self.att, dim = -1)

        _, x = self.gru(x)
        
        if not self.bidirectional:
            x = torch.squeeze(x)
        else:
            x = torch.mean(x, dim = 0)
        
        x = self.gru_norm(x)
        
        return x
        
class MixedDiscriminatorV01(nn.Module):
    def __init__(self, in_channels = 1, dim = 8, n_blocks = 4):
        return
    
class MultiHeadGRUDiscriminatorV01(nn.Module):
    def __init__(self, in_channels = 1, 
                 n_heads = 8, 
                 hidden_size = 128,
                 s = 64,
                 n = 64, 
                 s_dim = 1,
                 im_dim = 4,
                 n_blocks = 4, 
                 bidirectional = False):
        super(MultiHeadGRUDiscriminatorV01, self).__init__()
        
        # Initial convolution block
        layers = [
            nn.Conv2d(in_channels, im_dim, (1, 7), stride = (1, 1), padding = (0, 3)),
            nn.InstanceNorm2d(im_dim),
            nn.ReLU()
        ]
        
        for _ in range(n_blocks):
            layers += [
                ResidualBlock1d_(im_dim, im_dim),
                nn.Conv2d(im_dim, im_dim * 2, (1, 5), stride = (1, 1), padding = (0, 2)),
                nn.InstanceNorm2d(im_dim * 2),
            ]
            im_dim *= 2
            
        self.head = nn.Sequential(*layers)
        
        self.gru_heads_im = nn.ModuleList()
        
        for k in range(n_heads):
            self.gru_heads_im.append(ResGRUHead(n, s, im_dim, hidden_size, bidirectional = bidirectional))
            
        dim = s_dim
            
        # Initial convolution block
        layers = [
            nn.Conv1d(in_channels, dim, 7, padding = 3),
            nn.InstanceNorm1d(dim),
        ]

        for _ in range(n_blocks):
            layers += [
                ResidualBlock1d(dim, dim),
                nn.Conv1d(dim, dim * 2, 4, stride = 2, padding = 1),
                nn.InstanceNorm1d(dim * 2),
            ]
            dim *= 2
        
        self.seq_encoder = nn.Sequential(*layers)
        
        self.out = nn.Sequential(nn.Linear(1088, 1024), nn.LayerNorm((1024, )), nn.ReLU(),
                                 nn.Linear(1024, 512), nn.LayerNorm((512, )), nn.ReLU(),
                                 nn.Linear(512, 1))
        
    def forward(self, im, s):
        im = self.head(im)
        
        x = []
        for k in range(len(self.gru_heads_im)):
            x.append(self.gru_heads_im[k](im))
            
        s = self.seq_encoder(s)
        s = s.flatten(1)
        
        x = torch.cat(x, dim = -1)
        x = torch.cat([x, s], dim = -1)
        
        return self.out(x)
    
class PermInvariantDiscriminatorV02(nn.Module):
    def __init__(self, in_channels = 1, dim = 4, n_blocks = 4):
        super(PermInvariantDiscriminatorV02, self).__init__()
        
        dim_ = copy.copy(dim)
        
        # Initial convolution block
        self.init_conv = nn.Sequential(
            nn.ReflectionPad2d((3, 3, 0, 0)),
            nn.Conv2d(in_channels, dim_, (1, 7), stride = (1, 1)),
            nn.InstanceNorm2d(dim_),
        )
        
        layers = []
        for _ in range(n_blocks):
            layers += [
                ResidualBlock1d_(dim_, dim_),
                nn.Conv2d(dim_, dim_ * 2, (1, 5), stride = (1, 1), padding = (0, 2)),
                nn.InstanceNorm2d(dim_ * 2),
            ]
            dim_ *= 2

        self.im_encoder = nn.Sequential(*layers)
        
        dim = dim
        
        # Initial convolution block
        layers = [
            nn.ReflectionPad1d(3),
            nn.Conv1d(in_channels, dim, 7),
            nn.InstanceNorm1d(dim),
        ]

        for _ in range(n_blocks):
            layers += [
                ResidualBlock1d(dim, dim),
                nn.Conv1d(dim, dim * 2, 5, stride = 1, padding = 2),
                nn.InstanceNorm1d(dim * 2),
            ]
            dim *= 2
        
        self.seq_encoder = nn.Sequential(*layers)
        
        self.out = nn.Sequential(nn.Linear(30592, 8096), nn.LayerNorm((8096, )), nn.ReLU(),
                                 nn.Linear(8096, 4096), nn.LayerNorm((4096, )), nn.ReLU(),
                                 nn.Linear(4096, 1))
        
    def forward(self, im, s):
        im1 = self.init_conv(im)
        im2 = self.im_encoder(im1)
        
        s = self.seq_encoder(s)
        
        im = torch.cat([im, im1, im2], dim = 1)  
        
        im_mean_ind = torch.mean(im, dim = 2)
        im_var_ind = torch.var(im, dim = 2, unbiased = False)
        im_max_ind = torch.max(im, dim = 2)[0]
        im_min_ind = torch.min(im, dim = 2)[0]
        
        im_mean_sites = torch.mean(im, dim = -1)
        im_var_sites = torch.var(im, dim = -1, unbiased = False)
        im_max_sites = torch.max(im, dim = -1)[0]
        im_min_sites = torch.min(im, dim = -1)[0]

        im = torch.cat([im_mean_ind, im_var_ind, im_mean_sites, im_var_sites, im_max_sites, im_min_sites], dim = 1)
        im = im.flatten(1)
        s = s.flatten(1)
        
        im = torch.cat([im, s], dim = -1)
        
        return self.out(im)
    
class PermInvariantDiscriminatorV01(nn.Module):
    def __init__(self, in_channels = 1, dim = 4, n_blocks = 4):
        super(PermInvariantDiscriminatorV01, self).__init__()
        
        dim_ = copy.copy(dim)
        
        # Initial convolution block
        layers = [
            nn.ReflectionPad2d((3, 3, 0, 0)),
            nn.Conv2d(in_channels, dim_, (1, 7), stride = (1, 1)),
            nn.InstanceNorm2d(dim_),
        ]
        
        # [4, 8, 16, 32, 64]
        
        for _ in range(n_blocks):
            layers += [
                ResidualBlock1d_(dim_, dim_),
                nn.Conv2d(dim_, dim_ * 2, (1, 5), stride = 1, padding = (0, 2)),
                nn.InstanceNorm2d(dim_ * 2),
            ]
            dim_ *= 2

        self.im_encoder = nn.Sequential(*layers)
        
        dim = dim * 4
        
        # Initial convolution block
        layers = [
            nn.ReflectionPad1d(3),
            nn.Conv1d(in_channels, dim, 7),
            nn.InstanceNorm1d(dim),
        ]

        for _ in range(n_blocks):
            layers += [
                ResidualBlock1d(dim, dim),
                nn.Conv1d(dim, dim * 2, 4, stride=2, padding=1),
                nn.InstanceNorm1d(dim * 2),
            ]
            dim *= 2
        
        self.seq_encoder = nn.Sequential(*layers)
        
        self.out = nn.Sequential(nn.Linear(17408, 4096), nn.LayerNorm((4096, )), nn.ReLU(),
                                 nn.Linear(4096, 2048), nn.LayerNorm((2048, )), nn.ReLU(),
                                 nn.Linear(2048, 1))
        
    def forward(self, im, s):
        im = self.im_encoder(im)
        
        print(im.shape)
        s = self.seq_encoder(s)
        
        im_mean = torch.mean(im, dim = 2)
        im_max = torch.max(im, dim = 2)[0]
        im_min = torch.min(im, dim = 2)[0]
        im_var = torch.var(im, dim = 2, unbiased = False)

        im = torch.cat([im_mean, im_max, im_min, im_var], dim = 1)
        im = im.flatten(1)
        s = s.flatten(1)
        
        im = torch.cat([im, s], dim = -1)
        
        return self.out(im)
    
class ResidualDiscriminatorV01(nn.Module):
    def __init__(self, in_channels = 1, dim = 32, n_residual = 0, n_downsample = 4):
        super(ResidualDiscriminatorV01, self).__init__()
        
        dim_ = copy.copy(dim)
        
        # Initial convolution block
        layers = [
            nn.Conv2d(in_channels, dim_, 7, padding = 3),
            nn.InstanceNorm2d(dim_),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        for _ in range(n_downsample):
            layers += [
                ResidualBlock_(dim_, dim_),
                nn.Conv2d(dim_, dim_ * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim_ * 2),
                nn.ReLU(inplace=True),
            ]
            dim_ *= 2

        # Residual blocks
        for _ in range(n_residual):
            layers += [ResidualBlock_(dim_, dim_, norm="in")]

        #layers += [nn.AvgPool2d(2, 2)]
        layers += [nn.Flatten()]

        self.im_encoder = nn.Sequential(*layers)
        
        dim = dim // 2
        
        # Initial convolution block
        layers = [
            nn.Conv1d(in_channels, dim, 7, padding = 3),
            nn.InstanceNorm1d(dim),
            nn.ReLU(inplace=True),
        ]
        
        # Downsampling
        for _ in range(n_downsample):
            layers += [
                ResidualBlock1d_pad_regular(dim, dim),
                nn.Conv1d(dim, dim * 2, 4, stride=2, padding=1),
                nn.InstanceNorm1d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2

        # Residual blocks
        for _ in range(n_residual):
            layers += [ResidualBlock1d_pad_regular(dim, dim, norm="in")]

        #layers += [nn.AvgPool1d(2)]
        layers += [nn.Flatten()]
        
        self.seq_encoder = nn.Sequential(*layers)
        
        #self.out = nn.Sequential(nn.Linear(9216, 2048), nn.LayerNorm((2048, )), nn.ReLU(),
                                # nn.Linear(2048, 1))
        self.out = nn.Linear(9216, 1)
        
    def forward(self, im, s):
        im = self.im_encoder(im)
        s = self.seq_encoder(s)
        
        im = torch.cat([im, s], dim = -1)
        
        return self.out(im)
    
class ResidualGeneratorV02(nn.Module):
    def __init__(self, latent_dim = 128, twod_oned_op = 'max', final_layer = 'sigmoid'):
        super(ResidualGeneratorV02, self).__init__()
        
        # upsampling linear layer
        self.linear = MLP(latent_dim, 512 * 4 * 4, 2048)
        self.linear1d = MLP(latent_dim, 512 * 4, 1024)
    
        self.twod_oned_op = twod_oned_op
        
        self.start_size = [512, 4, 4]
                
        self.blocks_2d = nn.ModuleList()
        self.blocks_1d = nn.ModuleList()
        
        channels_2d = [512, 256, 128, 64, 32]
        channels_1d = [512, 256, 128, 64, 32]
        
        s = 8
        
        for k in range(len(channels_2d) - 1):
            _ = []
            
            _.append(ResidualBlock(channels_2d[k], channels_2d[k]))
            _.append(nn.ConvTranspose2d(channels_2d[k], channels_2d[k + 1], 2, stride = 2))
            _.append(nn.Conv2d(channels_2d[k + 1], channels_2d[k + 1], 5, stride = 1, padding = 2))
            _.append(nn.LayerNorm((channels_2d[k+1], s, s)))
            
            s *= 2
            
            block = nn.Sequential(*_)
            
            self.blocks_2d.append(block)
            
        s = 8
        for k in range(len(channels_1d) - 1):
            _ = []
            
            _.append(ResidualBlock1d(channels_1d[k], channels_1d[k]))
            _.append(nn.ConvTranspose1d(channels_1d[k], channels_1d[k + 1], 2, stride = 2))
            _.append(nn.Conv1d(channels_1d[k + 1], channels_1d[k + 1], 5, stride = 1, padding = 2))
            _.append(nn.LayerNorm((channels_1d[k+1], s)))
            #_.append(nn.ReLU(inplace = True))
            
            s *= 2
            
            block = nn.Sequential(*_)
            
            self.blocks_1d.append(block)
            

        if final_layer == 'sigmoid':
            self.out_2d = nn.Sequential(nn.Conv2d(channels_2d[-1], 1, 1), nn.Sigmoid())
            self.out_1d = nn.Sequential(
                                        nn.Conv1d(channels_1d[-1], 1, 1), 
                                        nn.Softmax(dim = -1))
        elif final_layer == 'linear':
            self.out_2d = nn.Sequential(nn.Conv2d(channels_2d[-1], 1, 1))
            self.out_1d = nn.Sequential(
                                        nn.Conv1d(channels_1d[-1], 1, 1))
            
        
    def forward(self, z):
        x2 = self.linear(z).view(*([z.shape[0]] + self.start_size))
        x1 = self.linear1d(z).view(*([z.shape[0]] + self.start_size[:-1]))
        
        for k in range(len(self.blocks_2d)):
            x2 = self.blocks_2d[k](x2)
            x1 = self.blocks_1d[k](x1)
                
        x2 = self.out_2d(x2)
        x1 = torch.cumsum(self.out_1d(x1), dim = -1)
        
        return x2, x1

class ResidualGenerator_B(nn.Module):
    def __init__(self, latent_dim = 128, twod_oned_op = 'max', final_layer = 'sigmoid'):
        super(ResidualGenerator_B, self).__init__()
        
        # upsampling linear layer
        self.linear = MLP_B(latent_dim, 512 * 4 * 4, 2048)
        self.linear1d = MLP_B(latent_dim, 512 * 4, 1024)

        #self.linear = init_lin(latent_dim, 512 * 4 * 4)
        #self.linear1d = init_lin(latent_dim, 512 * 4)

        self.twod_oned_op = twod_oned_op
        
        self.start_size = [512, 4, 4]
                
        self.blocks_2d = nn.ModuleList()
        self.blocks_1d = nn.ModuleList()
        
        channels_2d = [512, 256, 128, 64, 32]
        channels_1d = [512, 256, 128, 64, 32]
        
        s = 8
        
        for k in range(len(channels_2d) - 1):
            _ = []
            
            _.append(ResidualBlock_B(channels_2d[k], channels_2d[k]))
            _.append(nn.ConvTranspose2d(channels_2d[k], channels_2d[k + 1], 2, stride = 2))
            _.append(nn.Conv2d(channels_2d[k + 1], channels_2d[k + 1], 5, stride = 1, padding = 2))
            _.append(nn.BatchNorm2d(channels_2d[k+1]))
            
            s *= 2
            
            block = nn.Sequential(*_)
            
            self.blocks_2d.append(block)
            
        s = 8
        for k in range(len(channels_1d) - 1):
            _ = []
            
            _.append(ResidualBlock1d_B(channels_1d[k], channels_1d[k]))
            _.append(nn.ConvTranspose1d(channels_1d[k], channels_1d[k + 1], 2, stride = 2))
            _.append(nn.Conv1d(channels_1d[k + 1], channels_1d[k + 1], 5, stride = 1, padding = 2))
            _.append(nn.BatchNorm1d(channels_1d[k+1]))
            #_.append(nn.ReLU(inplace = True))
            
            s *= 2
            
            block = nn.Sequential(*_)
            
            self.blocks_1d.append(block)
            

        if final_layer == 'sigmoid':
            self.out_2d = nn.Sequential(nn.Conv2d(channels_2d[-1], 1, 1), nn.Sigmoid())
            self.out_1d = nn.Sequential(
                                        nn.Conv1d(channels_1d[-1], 1, 1), 
                                        nn.Softmax(dim = -1))
        elif final_layer == 'linear':
            self.out_2d = nn.Sequential(nn.Conv2d(channels_2d[-1], 1, 1))
            self.out_1d = nn.Sequential(
                                        nn.Conv1d(channels_1d[-1], 1, 1))
            
        
    def forward(self, z):
        x2 = self.linear(z).view(*([z.shape[0]] + self.start_size))
        x1 = self.linear1d(z).view(*([z.shape[0]] + self.start_size[:-1]))
        
        for k in range(len(self.blocks_2d)):
            x2 = self.blocks_2d[k](x2)
            x1 = self.blocks_1d[k](x1)
                
        x2 = self.out_2d(x2)
        x1 = torch.cumsum(self.out_1d(x1), dim = -1)
        
        return x2, x1

class ResidualDiscriminator_B(nn.Module):
    def __init__(self, in_channels = 1, dim = 32, n_residual = 0, n_downsample = 4):
        super(ResidualDiscriminator_B, self).__init__()
        
        dim_ = copy.copy(dim)
        
        # Initial convolution block
        layers = [
            nn.Conv2d(in_channels, dim_, 7, padding = 3),
            nn.GroupNorm(1,dim_),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        for _ in range(n_downsample):
            layers += [
                ResidualBlock_B_(dim_, dim_),
                nn.Conv2d(dim_, dim_ * 2, 4, stride=2, padding=1),
                nn.GroupNorm(1,dim_ * 2),
                nn.ReLU(inplace=True),
            ]
            dim_ *= 2

        # Residual blocks
        for _ in range(n_residual):
            layers += [ResidualBlock_B_(dim_, dim_, norm="in")]

        #layers += [nn.AvgPool2d(2, 2)]
        layers += [nn.Flatten()]

        self.im_encoder = nn.Sequential(*layers)
        
        dim = dim // 2
        
        # Initial convolution block
        layers = [
            nn.Conv1d(in_channels, dim, 7, padding = 3),
            nn.GroupNorm(1,dim),
            nn.ReLU(inplace=True),
        ]
        
        # Downsampling
        for _ in range(n_downsample):
            layers += [
                ResidualBlock1d_pad_regular_B(dim, dim),
                nn.Conv1d(dim, dim * 2, 4, stride=2, padding=1),
                nn.GroupNorm(1,dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2

        # Residual blocks
        for _ in range(n_residual):
            layers += [ResidualBlock1d_pad_regular_B(dim, dim, norm="in")]

        #layers += [nn.AvgPool1d(2)]
        layers += [nn.Flatten()]
        
        self.seq_encoder = nn.Sequential(*layers)
        
        #self.out = nn.Sequential(nn.Linear(9216, 2048), nn.LayerNorm((2048, )), nn.ReLU(),
                                # nn.Linear(2048, 1))
        self.out = nn.Linear(9216, 1)
        
    def forward(self, im, s):
        im = self.im_encoder(im)
        s = self.seq_encoder(s)
        
        im = torch.cat([im, s], dim = -1)
        
        return self.out(im)
    
class Generator(nn.Module):
    def __init__(self, latent_dim = 128):
        super(Generator, self).__init__()
        
        self.im_gen = ResidualGenerator2dV01(latent_dim = latent_dim)
        self.s_gen = ResidualGenerator1dV01(latent_dim = latent_dim)
        
    def forward(self, z):
        im = self.im_gen(z)
        s = self.s_gen(z)
        
        return im, s
    
if __name__ == '__main__':
    res = ResidualGeneratorV02()
    z = torch.randn((64, 128))
    
    im, pos = res(z)
    
    print(im.shape, pos.shape)
    
    disc = MultiHeadGRUDiscriminatorV01()
    
    y = disc(im, pos)

    print(y.shape)
    
            
        
        
        
