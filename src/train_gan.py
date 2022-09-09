import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging

plt.switch_backend('agg')

import torch
import torch.nn as nn
import argparse
from gan import Generator, Discriminator
from torch.autograd import Variable
from data_loader import GenomesDataset
from torch.utils.data import DataLoader
from data_processing import save_models, plot_losses, plot_pca, create_AGs


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--latent_size", default="600", help="size of noise input")
    parser.add_argument("--negative_slope", default="0.01", help="alpha value for LeakyReLU")
    parser.add_argument("--gen_lr", default="1e-4", help="generator learning rate")
    parser.add_argument("--disc_lr", default="8e-4", help="discriminator learning rate")
    parser.add_argument("--epochs", default="10000")
    parser.add_argument("--ag_size", default="216", help="number of artificial genomes (haplotypes) to be created"
                                                         "if 0, then no genomes created")
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--save_freq", default="0", help="save model every save_freq epochs") # zero means don't save
    parser.add_argument("--batch_size", default="32")
    parser.add_argument("--ifile", default="1000G_real_genomes/805_SNP_1000G_real.hapt")
    parser.add_argument("--odir", default="output/")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--gpu_count", default="0")
    parser.add_argument("--verbose", action="store_true")

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

    ifile = args.ifile
    odir = args.odir
    latent_size = int(args.latent_size)
    negative_slope = float(args.negative_slope)
    g_learn = float(args.gen_lr)
    d_learn = float(args.disc_lr)
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    ag_size = int(args.ag_size)
    use_cuda = args.use_cuda
    gpu_count = int(args.gpu_count)
    save_freq = int(args.save_freq)
    device = torch.device('cuda' if use_cuda else 'cpu')

    # The .hapt files are the data taken from the original paper
    if ".hapt" in ifile:
        df = pd.read_csv(ifile, sep=' ', header=None)
        df = df.sample(frac=1).reset_index(drop=True)
        df = df.drop(df.columns[0:2], axis=1)
        data = df.values
        data_size = data.shape[1]
    else:
        data = pd.read_csv(ifile)
        data_size = data.shape[1]

        # dropping all allele values that are not 0 or 1
        mask = data.isin([2, 3])
        data = data[~mask]
        data = data.dropna()

        # grabbing only a subset of real data otherwise our pca plots are covered with a bunch of data points
        if len(data) > ag_size * 5:
            data = data.sample(n=ag_size * 5)  # need to test what this multiple should be
        data = data.values
    df = pd.DataFrame(data)

    # The original paper did this. Perhaps to add some stochasticity in the input
    data = torch.FloatTensor(data - np.random.uniform(0, 0.1, size=(data.shape[0], data.shape[1])))

    # Load data into pytorch dataloader
    genomes_data = GenomesDataset(data)
    dataloader = DataLoader(dataset=genomes_data, batch_size=batch_size, shuffle=True, drop_last=True)

    # Make generator
    generator = Generator(data_size, latent_size, negative_slope).to(device)

    # Make discriminator
    discriminator = Discriminator(data_size, negative_slope).to(device)

    losses = []

    loss_fn = nn.BCELoss()

    # set optimizers
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=g_learn)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_learn)

    # Loop through each epoch
    for i in range(epochs):

        disc_losses = []
        gen_losses = []

        # Loop through each batch in dataloader
        for j, X_real in enumerate(dataloader):

            # move to gpu
            X_real = X_real.to(device)

            # create labels for real and fake data
            ones = Variable(torch.Tensor(batch_size, 1).fill_(1).type(torch.FloatTensor)).to(device)
            zeros = Variable(torch.Tensor(batch_size, 1).fill_(0).type(torch.FloatTensor)).to(device)

            # create batch from generator using noise as input
            z = torch.normal(0, 1, size=(batch_size, latent_size)).to(device)
            X_fake = generator(z).detach()

            ### ----------------------------------------------------------------- ###
            #                           train discriminator                       #
            ### ----------------------------------------------------------------- ###
            discriminator.train()
            disc_optimizer.zero_grad()
            generator.eval()

            # test the discriminator on real data
            real_preds = discriminator(X_real)
            disc_loss = loss_fn(real_preds, ones -
                                torch.FloatTensor(ones.shape[0], ones.shape[1]).uniform_(0, 0.1).to(device))

            # test the discriminator on fake data
            fake_preds = discriminator(X_fake)

            # calculate loss and take update step
            disc_loss += loss_fn(fake_preds, zeros)
            disc_loss.backward()
            disc_optimizer.step()

            ### ----------------------------------------------------------------- ###
            #                           train generator                           #
            ### ----------------------------------------------------------------- ###
            generator.train()
            gen_optimizer.zero_grad()
            discriminator.eval()
            z = torch.normal(0, 1, size=(batch_size, latent_size)).to(device)

            # create another fake batch
            X_batch_fake = generator(z)

            # test the discriminator on fake data again
            y_pred = discriminator(X_batch_fake).to(device)

            # calculate generator loss and take update step
            gen_loss = loss_fn(y_pred, ones)
            gen_loss.backward()
            gen_optimizer.step()

            # record losses and log performance
            losses.append((disc_loss.item(), gen_loss.item()))
            gen_losses.append(gen_loss.item())
            disc_losses.append(disc_loss.item())
        logging.info("Epoch:\t%d/%d Discriminator loss: %6.4f Generator loss: %6.4f" % (i + 1, epochs, np.mean(disc_losses), np.mean(gen_losses)))

        # every save_freq batches
        if save_freq != 0 and (i % save_freq == 1 or i == epochs):

            # Save models
            save_models(generator, discriminator, os.path.join(odir, "generator_model.pt"),
                        os.path.join(odir, "discriminator_model.pt"))

            if ag_size > 0:
                # Create AGs
                generated_genomes_df = create_AGs(generator, i, ag_size, latent_size, df, odir, use_cuda=use_cuda,
                                                  device=device)

                # plot losses and pca
                if args.plot:
                    plot_losses(odir, losses, i)

                    plot_pca(df, i, generated_genomes_df, odir)


if __name__ == "__main__":
    main()
