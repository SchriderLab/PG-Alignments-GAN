import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import glob
import logging
from data_processing_v2 import * 
from torchvision.utils import make_grid, save_image
import torchvision


plt.switch_backend('agg')

import torch
import argparse
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn

from layers import *

import sys
from collections import deque

from prettytable import PrettyTable

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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def flip_labels(x_real, x_fake, p_flip):
    # determine the number of labels to flip
    n_select = int(p_flip * x_real.shape[0])
    # choose labels to flip
    flip_ix = np.random.choice([i for i in range(x_real.shape[0])], size=n_select)
    ### flip labels
    x_real_temp = x_fake[flip_ix]
    x_fake_temp = x_real[flip_ix]
    x_real[flip_ix] = x_real_temp
    x_fake[flip_ix] = x_fake_temp

    return x_real, x_fake

class weightConstraint(object):
    def __init__(self):
        pass
    
    def __call__(self,module):
        if hasattr(module,'weight'):
            if module.weight is not None:
                w = module.weight.data
                w = w.clamp(-0.01, 0.01)
                module.weight.data = w
                
                
        
class FakeBuffer(object):
    def __init__(self, max_size = 2560):
        self.sites = deque(maxlen = max_size)
        self.pos = deque(maxlen = max_size)
        
        self.length = 0
        
    def sample(self, batch_size = 64):
        ix = np.random.choice(range(self.length), batch_size, replace = False)
    
        pos = torch.FloatTensor(np.array([self.pos[u] for u in ix]))
        sites = torch.FloatTensor(np.array([self.sites[u] for u in ix]))
        
        return sites, pos
    
    def add(self, sites, pos):
        self.sites.extend(list(sites))
        self.pos.extend(list(pos))

        self.length = len(self.sites)
        
def switch_labels(x):
    for k in range(x.shape[1]):
        if np.sum(x[:,k]) > x.shape[0] / 2.:
            x[:,k] = 1 - x[:,k]
        elif np.sum(x[:,k]) == x.shape[0] / 2.:
            if np.random.choice([0, 1]) == 0:
                x[:,k] = 1 - x[:,k]
                
    return x

def switch_monomorphic(sites, positions, lower_bound, upper_bound):
    """
    input: sites tensor, positions tensor, lower bound of # of monomorphic sites, upper bound of # of monomorphic sites

    return: sites tensor, positions tensor

    function: changes some sites to monomorphic and places their positions inbetween variable sites
    """

    for aln in range(sites.shape[0]):
        n_monomorphic = random.randint(lower_bound, upper_bound)
        mono_sites = np.random.choice(range(sites.shape[3]), n_monomorphic, replace = False)
        anc_sites = mono_sites[:(len(mono_sites)//2)]
        der_sites = mono_sites[(len(mono_sites)//2):]

        #print(sites[aln,:,:,anc_sites[0]])
        #print(sites[aln,:,:,anc_sites[0]].shape)


        sites[aln,:,:,anc_sites] = torch.zeros(size = sites[aln,:,:,anc_sites].shape)
        sites[aln,:,:,der_sites] = torch.ones(size = sites[aln,:,:,der_sites].shape)


        for i in mono_sites:
            if i == 0:
                positions[aln,:,i] = np.random.uniform(0, positions[aln,:,i+1].detach().numpy())[0]
            elif i == positions.shape[2]-1:
                positions[aln,:,i] = np.random.uniform(positions[aln,:,i-1].detach().numpy(),1)[0]
            else:
                positions[aln,:,i] = np.random.uniform(positions[aln,:,i-1].detach().numpy(),positions[aln,:,i+1].detach().numpy())[0]

    return sites, positions

def sample_genotypes_bernoulli(sites_tensor, require_variation = True):
    """
    input: tensor of sites data from the generator, values are floats that range 0.0 - 1.0

    return: tensor of sites data after genotyping, values are 0.0 or 1.0 to reflect ancestral or derived sites

    function: uses the sigmoid activation output from the generator as probabilites for sampling genotypes from a binomial distribution. 
              Can be conditioned on requiring variation at all positions (columns) reflective of simulation data
    """
    sites_tensor_sampled = torch.bernoulli(sites_tensor)

    if require_variation:
        for aln in range(sites_tensor_sampled.shape[0]):
            for pos in range(sites_tensor_sampled.shape[3]):
                if torch.min(sites_tensor_sampled[aln,:,:,pos]) == torch.max(sites_tensor_sampled[aln,:,:,pos]):                    
                    
                    switch_value = 1 - torch.min(sites_tensor_sampled[aln,:,:,pos])

                    probs = sites_tensor[aln,0,:,pos].detach().cpu().numpy()
                    if torch.min(sites_tensor_sampled[aln,:,:,pos]) == 1:
                        probs = 1 - probs
                    probs = probs + 0.0000001
                    sum_of_probs = sum(probs)
                    scalar = 1/sum_of_probs
                    scaled_probs = [x*scalar for x in probs]
                    sampled_idx = np.random.choice(range(sites_tensor_sampled.shape[2]), 1, p=scaled_probs)[0]

                    sites_tensor_sampled[aln,:,:,pos] = torch.bernoulli(sites_tensor[aln,:,:,pos])
                    sites_tensor_sampled[aln,:,sampled_idx,pos] = switch_value
    
    return sites_tensor_sampled

def sample_genotypes_bernoulli_gumbel(sites_tensor_discrete, sites_tensor_probs, require_variation = True):
    """
    input: tensor of sites data from the generator, values are floats that range 0.0 - 1.0

    return: tensor of sites data after genotyping, values are 0.0 or 1.0 to reflect ancestral or derived sites

    function: uses the sigmoid activation output from the generator as probabilites for sampling genotypes from a binomial distribution. 
              Can be conditioned on requiring variation at all positions (columns) reflective of simulation data
    """
    sites_tensor_sampled = sites_tensor_discrete

    if require_variation:
        for aln in range(sites_tensor_sampled.shape[0]):
            for pos in range(sites_tensor_sampled.shape[3]):
                if torch.min(sites_tensor_sampled[aln,:,:,pos]) == torch.max(sites_tensor_sampled[aln,:,:,pos]):                    
                    
                    switch_value = 1 - torch.min(sites_tensor_sampled[aln,:,:,pos])

                    probs = sites_tensor_probs[aln,0,:,pos].detach().cpu().numpy()
                    if torch.min(sites_tensor_sampled[aln,:,:,pos]) == 1:
                        probs = 1 - probs
                    probs = probs + 0.0000001
                    sum_of_probs = sum(probs)
                    scalar = 1/sum_of_probs
                    scaled_probs = [x*scalar for x in probs]
                    sampled_idx = np.random.choice(range(sites_tensor_sampled.shape[2]), 1, p=scaled_probs)[0]

                    #sites_tensor_sampled[aln,:,:,pos] = torch.bernoulli(sites_tensor[aln,:,:,pos])
                    sites_tensor_sampled[aln,:,sampled_idx,pos] = switch_value
    
    return sites_tensor_sampled


class DataGeneratorDisk(object): 
    def __init__(self, idir, num_in=10000, batch_size = 8):
        self.positions = list(glob.glob(os.path.join(idir, '*_pos.csv')))
        self.sites = list(glob.glob(os.path.join(idir, '*_sites.csv')))
        
        self.indices = random.sample(list(range(len(self.positions))), k=num_in)

        self.length = len(self.indices) // batch_size
        self.batch_size = batch_size
        
        self.on_epoch_end()

    def on_epoch_end(self):
        random.shuffle(self.indices)
        
        self.ix = 0

    def get_batch(self, label_smooth = False, shuffle = False):
        pos_files = [self.positions[u] for u in self.indices[self.ix:self.ix + self.batch_size]]
        site_files = [self.sites[u] for u in self.indices[self.ix:self.ix + self.batch_size]]
        
        self.ix += self.batch_size
        
        pos = []
        sites = []
        
        for k in range(len(pos_files)):
            p = np.loadtxt(pos_files[k])
            s = np.loadtxt(site_files[k], delimiter = ',')

            if shuffle:
                np.random.shuffle(s)
            
            pos.append(p)
            sites.append(s)
            
        sites = np.array(sites)
            
        if label_smooth:
            ey = np.random.uniform(0., 0.05, sites.shape)
            
            sites = sites * (1 - ey) + 0.05 * ey
            
        return torch.FloatTensor(np.expand_dims(sites, 1)), torch.FloatTensor(np.expand_dims(np.array(pos), 1))

class DataGenerator(object):
    def __init__(self, sites_data, pos_data, batch_size = 64):
        self.sites_data = sites_data
        self.pos_data = pos_data
        
        self.length = self.sites_data.shape[0] // batch_size
        self.batch_size = batch_size
        
        self.on_epoch_end()
        
    def __len__(self):
        return self.length
        
    def on_epoch_end(self):
        self.indices = list(range(self.sites_data.shape[0]))
        random.shuffle(self.indices)
        
    def get_batch(self, sample = False):
        xs = self.sites_data[self.indices[:self.batch_size]]
        ey = torch.FloatTensor(np.random.uniform(0., 0.05, xs.shape))
        
        xs = xs * (1 - ey) + 0.5 * ey
        
        xp = self.pos_data[self.indices[:self.batch_size]]
        
        if not sample:
            del self.indices[:self.batch_size]
        else:
            random.shuffle(self.indices)
            
        return xs, xp
    
import matplotlib.pyplot as plt

MS_PATH = os.path.join(os.getcwd(), 'msdir/ms')

def split(word):
    return [char for char in word]

def read_ms_data(ms_lines):
    idx_list = [idx for idx, value in enumerate(ms_lines) if '//' in value] + [len(ms_lines)]

    ms_chunks = [ms_lines[idx_list[k]:idx_list[k+1]] for k in range(len(idx_list) - 1)]
    ms_chunks[-1] += ['\n']

        
    X = []
    positions = []
    for chunk in ms_chunks[:-1]:
        pos = np.array([u for u in chunk[2].split(' ')[1:-1] if u != ''], dtype = np.float32)

        x = np.array([list(map(int, split(u.replace('\n', '')))) for u in chunk[3:-1]], dtype = np.float32)
        
        #x, _ = seriate_x(x)
        
        X.append(x)
        positions.append(pos)
        
    return np.expand_dims(np.array(X), 1), np.expand_dims(np.array(positions), 2)

class SimulatorGenerator(object):
    def __init__(self, batch_size = 64, sim_class = 1):
        if sim_class == 0:
            cmd = '{0} 64 {1} -s 64'
        elif sim_class == 1:
            cmd = '{0} 64 {1} -s 64 -I 2 32 32 0.01'
            
        self.cmd = cmd.format(MS_PATH, batch_size + 1)
        
    def get_batch(self, label_smooth = True):
        lines = os.popen(self.cmd).read().split('\n')
        
        sites, pos = read_ms_data(lines)
        
        if label_smooth:
            ey = np.random.uniform(0., 0.1, sites.shape)
            
            sites = sites * (1 - ey) + 0.5 * sites
        
        return torch.FloatTensor(sites), torch.FloatTensor(pos).transpose(1, 2)

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

def parse_args():
    parser = argparse.ArgumentParser()

    # gen params
    parser.add_argument("--latent_size", default="128", help="size of latent/noise vector")
    
    # IO
    parser.add_argument("--idir", default="ms_sims/64ind_6sites_5.16e-04theta_10krep_pos/", help="input directory")
    parser.add_argument("--odir", default="output/", help="output directory")
    parser.add_argument("--plot", action="store_true", help="plot summaries in output")
    
    # architecture
    parser.add_argument("--gen", default="sigGen", help="set what type of generator to be used. Options: sigGen tanGen tanNorm")
    
    # training paramaters
    parser.add_argument("--loss", default = "gp", help="whether to use gp or div to make the loss 1-Lipschitz compatible")
    parser.add_argument("--gen_lr", default="0.00005", help="generator learning rate")
    parser.add_argument("--disc_lr", default="0.00005", help="discriminator learning rate")
    parser.add_argument("--num_in", default="10000", help="number of input alignments")
    parser.add_argument("--use_cuda", action="store_true", help="use cuda?")
    parser.add_argument("--save_freq", default="0", help="save model every save_freq epochs") # zero means don't save
    parser.add_argument("--batch_size", default="64", help="set batch size")
    parser.add_argument("--epochs", default="10000", help="total number of epochs")
    
    # GAN params
    parser.add_argument("--critic_iter", default="5", help="number of generator iterations per critic iteration") ### number of generator iterations per critic iteration
    parser.add_argument("--gp_lambda", default="10", help="lambda for gradient penalty") ### lambda for gradient penalty
    
    # reinforcement learning additions
    parser.add_argument("--use_buffer", action = "store_true", help = "use a buffer for fake data sampling")
    parser.add_argument("--buffer_n", default = "20", help = "the buffer size will be this many batches large (integer)")
    
    # data augmentation + regularization
    parser.add_argument("--permute", action = "store_true", help = "permute real data along the individual axis")
    parser.add_argument("--label_smooth", action = "store_true", help = "label smooth both real and fake data")
    parser.add_argument("--label_noise", default = "0.05", help = "upper bound of the uniform distribution used to label smooth")
    parser.add_argument("--mono_switch", action = "store_true", help = "switch some input sites to monomorphic for training")
    parser.add_argument("--normalize", action = "store_true", help = "normalize inputs for tanh activation")
    parser.add_argument("--shuffle_inds", action = "store_true", help = "shuffle individuals in each input alignment")
    
    parser.add_argument("--verbose", action="store_true", help="verbose output to log")
    args = parser.parse_args()



    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.mkdir(args.odir)
            logging.debug('root: made output directory {0}'.format(args.odir))
        else:
            os.system('rm -rf {0}'.format(os.path.join(args.odir, '*')))

    return args

def main():

    args = parse_args()

    idir = args.idir
    odir = args.odir
    gen_type = args.gen
    latent_size = int(args.latent_size)
    g_learn = float(args.gen_lr)
    d_learn = float(args.disc_lr) 
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    use_cuda = args.use_cuda
    save_freq = int(args.save_freq)
    critic_iter = int(args.critic_iter)
    gp_lambda = int(args.gp_lambda)
    num_in = int(args.num_in)
    beta1 = 0.5
    beta2 = 0.999
    
    
    transform_tanh = torchvision.transforms.Normalize(0.5, 0.5)
    w_dists_pca = []
    epochs_vec = []
    
    plot_ix = 0


    device = torch.device('cuda' if use_cuda else 'cpu')
    
    data_size = 64
    num_channels = 1

    data_generator = DataGeneratorDisk(args.idir, num_in = num_in, batch_size = batch_size)

    max_gen_size = 1000
    data_gen_plot = DataGeneratorDisk(args.idir, num_in = num_in, batch_size = max_gen_size)
    in_plot_data_sites, in_plot_data_pos  = data_gen_plot.get_batch(shuffle = args.shuffle_inds)

    if args.mono_switch:
        in_plot_data_sites, in_plot_data_pos = switch_monomorphic(in_plot_data_sites, in_plot_data_pos, 0, 10)

    in_plot_data_sites = in_plot_data_sites.cpu()
    in_plot_data_pos = in_plot_data_pos.cpu()
    #generator = ResidualGenerator_B().to(device)
    #discriminator = ResidualDiscriminator_B().to(device)
    
    generator = DC_Generator_Leaky_Sig(data_size, latent_size, 0.1, 1, k_size=4).to(device)
    discriminator = DC_Discriminator(data_size, 0.1, 1, k_size=4).to(device)

    #count_parameters(generator)
    #count_parameters(discriminator)
    
    #discriminator.apply(weights_init)
    losses = []

    # set optimizers
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_learn, betas=(beta1, beta2))
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=g_learn, betas=(beta1, beta2))

    one = torch.tensor(1, dtype=torch.float).to(device)
    neg_one = one * -1
    neg_one = neg_one.to(device)

    #epoch_length = 500
    epoch_length = batch_size
    constraint = weightConstraint()
    
    adversarial_loss = nn.BCEWithLogitsLoss()
    fake_buffer = FakeBuffer(max_size = 20 * batch_size)
    
    history = dict()
    history['epoch'] = []
    history['disc_loss'] = []
    history['gen_loss'] = []
    
    # Loop through each epoch
    for i in range(1,epochs+1):
        disc_losses = []
        gen_losses = []
    
        # Loop through each batch in dataloader
        for j in range(data_generator.length):
            # WGAN-GP takes multiple critic steps for each generator step
            for h in range(critic_iter):
                X_real_sites, X_real_pos = data_generator.get_batch(shuffle = args.shuffle_inds)

                if args.normalize:
                    X_real_sites = transform_tanh(X_real_sites)
                    X_real_pos = transform_tanh(X_real_pos)
                
                if args.mono_switch:
                    X_real_sites, X_real_pos = switch_monomorphic(X_real_sites, X_real_pos, 0, 10)

                X_real_sites = X_real_sites.to(device)
                X_real_pos = X_real_pos.to(device)
                
                # I did this as a patch...but because of critic iter the 
                # data gen runs out of batches.  need to fix in the class
                if X_real_sites.shape[0] != batch_size:
                    break
                
                z = torch.randn(batch_size, latent_size, 1, 1, device=device)
                
                X_fake_sites, X_fake_pos = generator(z)
                
                #X_fake_sites = X_fake_sites.detach()
                #X_fake_pos = X_fake_pos.detach()
                
                #X_fake_sites = X_fake_sites.detach().cpu().numpy()
                #X_fake_pos = X_fake_pos.detach().cpu().numpy()
                
                #fake_buffer.add(X_fake_sites, X_fake_pos)
                
                #X_fake_sites, X_fake_pos = fake_buffer.sample(batch_size = batch_size)

                X_fake_sites = X_fake_sites.to(device)
                X_fake_pos = X_fake_pos.to(device)
                
                #interchange samples from real and fake to assist training (flipping)
                #X_real_sites, X_fake_sites = flip_labels(X_real_sites, X_fake_sites, 0.01)
                #X_real_pos, X_fake_pos = flip_labels(X_real_pos, X_fake_pos, 0.01)
                
                # these splits are for whether we want to use divergence loss in which case the inputs require gradients
                if args.loss == 'gp':
                    X_real_sites = X_real_sites.to(device, dtype=torch.float)
                    X_real_pos = X_real_pos.to(device, dtype=torch.float)
                elif args.loss == 'div':
                    X_real_sites = Variable(X_real_sites.to(device, dtype=torch.float), requires_grad = True)
                    X_real_pos = Variable(X_real_pos.to(device, dtype=torch.float), requires_grad = True)
                    

                ### ----------------------------------------------------------------- ###
                #                           train critic                                #
                ### ----------------------------------------------------------------- ###
                disc_optimizer.zero_grad()

                if args.loss == 'div':
                    z = Variable(z, requires_grad = True)
                
                if args.loss == 'div':
                    X_fake_sites = Variable(X_fake_sites.to(device), requires_grad = True)
                    X_fake_pos = Variable(X_fake_pos.to(device), requires_grad = True)
                else:
                    X_fake_sites = X_fake_sites.to(device)
                    X_fake_pos = X_fake_pos.to(device)
                    
                
                im, s = torch.cat([X_real_sites, X_fake_sites]), torch.cat([X_real_pos, X_fake_pos])
                d_loss = discriminator(im, s)
                
                if args.loss == 'gp':
                    d_loss_real = d_loss[:batch_size].mean()
                    d_loss_fake = d_loss[batch_size:].mean()
                    
                    wasserstein_d = d_loss_fake - d_loss_real + gp_gradient_penalty(discriminator, X_real_sites, X_real_pos, X_fake_sites, X_fake_pos, gp_lambda, device)
                elif args.loss == 'div':
                    d_loss_real = d_loss[:batch_size]
                    d_loss_fake = d_loss[batch_size:]
                    
                    div_gp = div_gradient_penalty(d_loss_real, d_loss_fake, X_real_sites, X_real_pos, X_fake_sites, X_fake_pos, device)
                    
                    d_loss_real = d_loss_real.mean()
                    d_loss_fake = d_loss_fake.mean()
                    
                    wasserstein_d = d_loss_fake - d_loss_real + div_gp
                elif args.loss == 'll':
                    d_loss_real = d_loss[:batch_size]
                    d_loss_fake = d_loss[batch_size:]
                    
                    valid = Variable(torch.FloatTensor(d_loss_real.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
                    fake = Variable(torch.FloatTensor(d_loss_fake.shape[0], 1).fill_(0.0), requires_grad=False).to(device)
                    
                    d_loss_real = adversarial_loss(d_loss_real, valid)
                    d_loss_fake = adversarial_loss(d_loss_fake, fake)
                    wasserstein_d = (d_loss_real + d_loss_fake) / 2
                    
                wasserstein_d.backward()

                # take optimization step
                disc_optimizer.step()
                
                #discriminator.apply(constraint)
                
            ### ----------------------------------------------------------------- ###
            #                           train generator                           #
            ### ----------------------------------------------------------------- ###
            gen_optimizer.zero_grad()

            z = torch.randn(batch_size, latent_size, 1, 1, device=device)
            #z = torch.randn(batch_size, latent_size, device=device)
            # create fake batch and test discriminator
            X_fake_sites, X_fake_pos = generator(z)
            gen_loss = discriminator(X_fake_sites, X_fake_pos)

            # calculate loss and take update step
            if args.loss != 'll':
                gen_loss = -1 * gen_loss.mean()
            else:
                gen_loss = adversarial_loss(gen_loss, valid)
            
            gen_loss.backward()
            gen_optimizer.step()
            
            disc_losses.append(wasserstein_d.item())
            gen_losses.append(gen_loss.item())

            if X_real_sites.shape[0] != batch_size:
                break

            if j % epoch_length - 1 == 0 and j != 0:
                losses.append((wasserstein_d.item(), gen_loss.item(), d_loss_real.item(), d_loss_fake.item()))
                logging.info("Epoch:\t%d/%d Discriminator loss: %6.8f Generator loss: %6.8f d_real: %6.8f d_fake: %6.8f" % (i, epochs, wasserstein_d.item(), gen_loss.item(), d_loss_real.item(), d_loss_fake.item()))

            #if (j + 1) % 10 == 0 and save_freq != 0 and (i == 1 or i % save_freq == 0 or i == epochs):        
            #    
            #    X_fake_sites = X_fake_sites.detach().cpu().numpy()
            #    X_fake_sites = (X_fake_sites - np.min(X_fake_sites)) / (np.max(X_fake_sites) - np.min(X_fake_sites))
            #    
            #    X_real_sites = X_real_sites.detach().cpu().numpy()
            #    
            #   X_fake_pos = X_fake_pos.detach().cpu().numpy()
            #   X_real_pos = X_real_pos.detach().cpu().numpy()
            #   
            #   fig, axes = plt.subplots(nrows = 2, ncols = 3)
            #   im = axes[0,0].imshow(X_fake_sites[0,0,:,:])
            #   fig.colorbar(im, ax = axes[0,0])
            #   
            #   axes[0,2].imshow(np.round(X_fake_sites[0,0,:,:]))
            #   
            #   axes[1,0].imshow(X_real_sites[0,0,:,:], vmin = 0., vmax = 1.)
            #   
            #   axes[0, 1].plot(X_fake_pos[0,0,:])
            #   axes[1, 1].plot(X_real_pos[0,0,:])
            #   
            #   plt.savefig(os.path.join(odir, '{0:04d}_{1:05d}_current_perf.png'.format(i, j)), dpi = 100)
            #    plt.close()
        
        history['epoch'].append(i)
        history['gen_loss'].append(np.mean(gen_losses))
        history['disc_loss'].append(np.mean(disc_losses))
        
        data_generator.on_epoch_end()

        if i == 1:
            save_image(in_plot_data_sites[:64,:,:,:], os.path.join(odir, "input_sites.png"), pad_value = 0.5)
            make_center_line_alignments(in_plot_data_sites[:64,0,:,:], i, odir + "input_", input_align=True)
            make_viridis_alignments(in_plot_data_sites[:64,0,:,:], i, odir + "input_", input_align=True)
        #    fig, axes = plt.subplots(nrows = 4, ncols = 4)

        if save_freq != 0 and (i == 1 or i % save_freq == 0 or i == epochs):
            epochs_vec.append(i)
            # Save models

            plotting_z = torch.randn(max_gen_size, latent_size, 1, 1, device=device)
            fake_sites_plotting, fake_pos_plotting = generator(plotting_z)

            if args.normalize:
                fake_sites_plotting = fake_sites_plotting / 2 + 0.5
                fake_pos_plotting = fake_pos_plotting / 2 + 0.5

            fake_sites_plotting = sample_genotypes_bernoulli(fake_sites_plotting)
            fake_sites_plotting = fake_sites_plotting.detach().cpu().numpy()
            fake_pos_plotting = fake_pos_plotting.detach().cpu().numpy()
        
            save_models(generator, discriminator, os.path.join(odir, "generator_model_{0:04d}.pt".format(i)),
                        os.path.join(odir, "discriminator_model_{0:04d}.pt".format(i)))
            
            
            if args.plot:
                if not os.path.exists(os.path.join(odir, str(i) + '/')):
                    os.mkdir(os.path.join(odir, str(i) + '/'))
                    logging.debug('root: made output directory {0}'.format(i))
                else:
                    os.system('rm -rf {0}'.format(os.path.join(odir, str(i) + '/*')))
                
                aa_truth_array = []
                aa_synth_array = []

                rand_1000 = np.random.choice(range(max_gen_size),1000, replace = False) ## keeps sites and pos together for any plotting that requires both
                
                plot_windowed_stats_paneled(fake_sites_plotting[:1000,:,:,:], fake_pos_plotting[:1000], in_plot_data_sites[rand_1000,:,:,:], in_plot_data_pos[rand_1000], i, odir)
                plot_sumstats_dist(fake_sites_plotting[:1000,:,:,:], fake_pos_plotting[:1000], in_plot_data_sites[rand_1000,:,:,:], in_plot_data_pos[rand_1000], i, odir)
                
                generated_alignments_img = plot_sites_grid(fake_sites_plotting[:64,:,:,:], i, odir, device)
                make_center_line_alignments(fake_sites_plotting[:64,0,:,:], i, odir)
                make_viridis_alignments(fake_sites_plotting[:64,0,:,:], i, odir)

                plot_losses(odir, losses, i)
                
                w_dists_pca.append(plot_sfs_pca(fake_sites_plotting[:500,:,:,:], i, in_plot_data_sites[np.random.choice(range(in_plot_data_sites.shape[0]),500, replace = False),:,:,:], data_size, odir, device))
                with open(os.path.join(odir,'w_dist_pca.txt'), 'a') as w_dist_file:
                    w_dist_file.write(str(epochs_vec[-1])+"\t")
                    w_dist_file.write(str(w_dists_pca[-1])+"\n")
                if i > 1:
                    plot_w_dist_pca(w_dists_pca,epochs_vec,i,odir)
                
                sfs_1k_real, sfs_1k_gen, aa_truth, aa_synth = plot_sfs_avg_bar(fake_sites_plotting[:1000,:,:,:], i, in_plot_data_sites[np.random.choice(range(in_plot_data_sites.shape[0]),1000, replace = False),:,:,:], odir, return_sfs=True)
                aa_truth_array.append(aa_truth)
                aa_synth_array.append(aa_synth)
                plot_aa(epochs_vec,aa_truth_array,aa_synth_array,odir)
                make_stairway_blueprint(sfs_1k_gen, sfs_1k_real, i, odir)
                make_stairway_blueprint_OoA2pop(fake_sites_plotting[:1000,:,:,:], in_plot_data_sites[np.random.choice(range(in_plot_data_sites.shape[0]),1000, replace = False),:,:,:], i, odir)
                plot_subpop_joint_sfs(fake_sites_plotting[:1000,:,:,:], i, in_plot_data_sites[np.random.choice(range(in_plot_data_sites.shape[0]),1000, replace = False),:,:,:], data_size, 32, 32, odir, device)
                plot_ld(fake_sites_plotting[:1000,:,:,:], fake_pos_plotting[:1000], i, in_plot_data_sites[rand_1000,:,:,:], in_plot_data_pos[rand_1000], data_size, odir, device)
                plot_position_vector_same(fake_pos_plotting[:100], i, in_plot_data_pos[np.random.choice(range(in_plot_data_pos.shape[0]),100, replace = False)], odir)
                plot_branch_length_distributions(fake_sites_plotting[:5,:,:,:], i, in_plot_data_sites[np.random.choice(range(in_plot_data_sites.shape[0]),5, replace = False),:,:,:], data_size, odir, device)
if __name__ == "__main__":
    main()
