import torch
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.manifold import TSNE
import numpy as np
import os
from sklearn.decomposition import PCA
import pandas as pd
from torchvision.utils import make_grid, save_image
import allel
import ot
from sliced import sliced_wasserstein_distance
from scipy.spatial.distance import squareform, cdist
import scipy.special
import seaborn as sns
from Bio.Phylo.TreeConstruction import *
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from data_calcs import *
from matplotlib.colors import LogNorm
import re

# saves models
def save_models(gen, disc, save_gen_path, save_disc_path):
    torch.save(gen.state_dict(), save_gen_path)
    torch.save(disc.state_dict(), save_disc_path)


# plots and records losses for entire training time
def plot_losses(odir, losses, curr_epoch):
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots()
    plt.plot(np.array([losses]).T[0], label='Discriminator', color = '#440154FF')
    #plt.plot(np.array([losses]).T[1], label='Generator', color = '#FDE725FF')
    #plt.plot(np.array([losses]).T[2], label='D_loss_real', color = '#404788FF')
    #plt.plot(np.array([losses]).T[3], label='D_loss_fake', color = '#73D055FF')
    
    plt.title("Training Losses")
    plt.legend()
    fig.set_size_inches(10, 10)
    fig.set_dpi(500)
    fig.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + '_loss.png'), format='png')
    fig.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + '_loss.svg'), format='svg')

    plt.close(fig)


# plots and records pca comparing real and generated sequences
def plot_pca(df, curr_epoch, generated_genomes_df, odir, model_type="normal", labels=None, pop_dict=None, projection="2d"):
    plt.rcParams.update({'font.size': 18})

    # copy original data
    df_temp = df.copy()

    # set up data labels for plotting
    if model_type == "conditional":
        pop_labels = list(pop_dict.keys())
        pops = ["Real_" + x for x in pop_labels]
        pops.extend(["AG_" + x for x in pop_labels])
        df_temp["label"] = labels.map(lambda x: "Real_" + x)
        generated_genomes_df['label'] = generated_genomes_df.loc[:, "label"].map(lambda x: "AG_" + x)
    else:
        df_temp["label"] = "Real"
        pops = ['Real', 'AG']

    # append Real and AG data
    df_all_pca = pd.concat([df_temp, generated_genomes_df])

    # calculate principal components and collect labels
    n_components = int(projection[0])
    pca = PCA(n_components=n_components)
    labels = df_all_pca.pop("label").to_list()
    PCs = pca.fit_transform(df_all_pca)

    fig = plt.figure(figsize=(10, 10))

    if projection == "2d":
        PCs_df = pd.DataFrame(data=PCs, columns=['PC1', 'PC2'])
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
    else:
        PCs_df = pd.DataFrame(data=PCs, columns=["PC1", "PC2", "PC3"])
        ax = plt.axes(projection="3d")

    PCs_df['Pop'] = labels

    # plot data
    for pop in pops:
        ix = PCs_df['Pop'] == pop
        if projection == "2d":
            ax.scatter(PCs_df.loc[ix, 'PC1']
                       , PCs_df.loc[ix, 'PC2']
                       , s=50, alpha=0.2)
        else:
            ax.scatter3D(PCs_df.loc[ix, 'PC1'], PCs_df.loc[ix, 'PC2'], PCs_df.loc[ix, 'PC3'], s=50, alpha=0.2)

    ax.legend(pops)
    fig.set_size_inches(10, 10)
    fig.set_dpi(500)
    fig.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + '_pca.pdf'), format='pdf')
    plt.cla()
    plt.close(fig)

def plot_sfs_pca(fake_sites, curr_epoch, in_dat, data_size, odir, device):
    plt.rcParams.update({'font.size': 18})
   
    fake_sites[fake_sites < 0] = 0
    fake_sites = fake_sites[:,0,:,:]
    fake_sites = np.rint(fake_sites)
    sfs_array_fake = [[]]*fake_sites.shape[0]
    for f in range(fake_sites.shape[0]):
        sfs_sum = ([sum(x) for x in zip(*fake_sites[f])])
        temp_sfs = allel.sfs(np.array(sfs_sum).astype(int))
        if np.array(temp_sfs).shape[0] < (fake_sites.shape[2]+1):
            sfs_array_fake[f] = np.append(temp_sfs,np.zeros((fake_sites.shape[2]+1)-np.array(temp_sfs).shape[0])).astype(int)
        else:
            sfs_array_fake[f] = temp_sfs
    
    in_dat[in_dat < 0] = 0
    in_dat = in_dat[:,0,:,:]
    in_dat = np.rint(in_dat)

    sfs_array_real = [[]]*in_dat.shape[0]
    for f in range(in_dat.shape[0]):
        sfs_sum = ([sum(x) for x in zip(*in_dat[f])])
        temp_sfs = allel.sfs(np.array(sfs_sum).astype(int))
        if np.array(temp_sfs).shape[0] < (in_dat.shape[2]+1):
            sfs_array_real[f] = np.append(temp_sfs,np.zeros((in_dat.shape[2]+1)-np.array(temp_sfs).shape[0])).astype(int)
        else:
            sfs_array_real[f] = temp_sfs

    ########## setting up for PCA ########
    #in_dat["label"] = "Real"
    #sfs_array["label"] = "Fake"
    sfs_array_real = pd.DataFrame(sfs_array_real)
    sfs_array_fake = pd.DataFrame(sfs_array_fake)
    labels = pd.concat([pd.DataFrame(["Input"]*sfs_array_real.shape[0], columns=['label']) , pd.DataFrame(["Generated"]*sfs_array_fake.shape[0], columns=['label'])])
    df_all_pca = pd.concat([sfs_array_real, sfs_array_fake])
    labels = labels.reset_index(drop=True)
    df_all_pca = df_all_pca.reset_index(drop=True)

    pca = PCA(n_components=2)
    PCs = pca.fit_transform(df_all_pca)
    explained_var = pca.explained_variance_ratio_
    principalDf = pd.DataFrame(data = PCs, columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, labels], axis = 1)

    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1 ('+str(round(explained_var[0],2))+"%)")
    ax.set_ylabel('Principal Component 2 ('+str(round(explained_var[1],2))+"%)")
    fig.suptitle("2 component PCA")

    labels = ['Input', 'Generated']
    colors = ['#440154FF','#29AF7FFF']
    #colors = ['r','b']
    for label, color in zip(labels,colors):
        indicesToKeep = finalDf['label'] == label
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
                , c = color
                , s = 50
                , alpha = 0.7)
    ax.legend(labels)
    fig.set_size_inches(10, 10)
    fig.set_dpi(500)
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + '_pca.png'), format='png')
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + '_pca.svg'), format='svg')

    #plt.savefig(os.path.join(odir, str(curr_epoch) + '_pca.png'), format='png')
    plt.close(fig)

    w_dist_pca = sliced_wasserstein_distance(np.asarray(principalDf)[:sfs_array_real.shape[0],:], np.asarray(principalDf)[sfs_array_real.shape[0]:sfs_array_real.shape[0]+sfs_array_fake.shape[0],:])
    
    return w_dist_pca

def plot_w_dist_pca(w_dists_pca,epochs_vec,i,odir):
    plt.rcParams.update({'font.size': 18})

    plt.figure(figsize=(10,10), dpi=500)
    plt.plot(epochs_vec,w_dists_pca, label = "2DSWD", color = 'black', linewidth=3)
    plt.xlabel("Epoch")
    plt.ylabel("2D Sliced Wasserstein Distance")
    plt.title("2D Sliced Wasserstein Distance Generated vs. Real SFS 2-Component PCA")
    plt.savefig(os.path.join(odir,'w_dist_pca.png'), format='png')
    plt.savefig(os.path.join(odir,'w_dist_pca.svg'), format='svg')

    plt.close()

def plot_aa(epochs_vec,aa_truth,aa_synth,odir):

    aa_ts = aa_truth + aa_synth / 2
    plt.rcParams.update({'font.size': 18})

    fig, ax = plt.subplots()

    plt.plot(epochs_vec, aa_truth, label='AA_truth', color = '#440154FF')
    plt.plot(epochs_vec, aa_synth, label='AA_synth', color = '#FDE725FF')
    plt.plot(epochs_vec, aa_ts, label='AA_ts', color = '#404788FF')


    #plt.title("Adversarial Accuracy")
    plt.legend()
    plt.ylim([0,1])
    fig.set_size_inches(10, 10)
    plt.savefig(os.path.join(odir,'adversarial_accuracy.png'), format='png')
    plt.savefig(os.path.join(odir,'adversarial_accuracy.svg'), format='svg')
    plt.close(fig)
    dt = pd.DataFrame(np.array([epochs_vec,aa_truth,aa_synth,aa_ts]).transpose(),columns=['epoch','aa_truth','aa_synth','aa_ts'])
    dt.to_csv(os.path.join(odir,'adversarial_accuracy.csv'), header=True, index=False)
    plt.close()


def plot_sfs_ind(generator, curr_epoch, num_sfs, latent_size, data_size, odir, device, normalize):
    plt.rcParams.update({'font.size': 18})
   
    z = torch.randn(num_sfs, latent_size, 1, 1, device=device)
    #z = torch.randn(num_sfs, latent_size, device=device)
    generator.eval()
    if normalize:
        fake_sites = generator(z).detach().cpu().numpy() / 2 + 0.5
    else:
        fake_sites = generator(z).detach().cpu().numpy()
    fake_sites[fake_sites < 0] = 0
    fake_sites = fake_sites[:,0:1,:,:]
    fake_sites = np.rint(fake_sites)
    fake_sites_3d = np.reshape(fake_sites,(num_sfs,data_size,data_size))
    
    fig, axes = plt.subplots(nrows=int(num_sfs/4), ncols=int(num_sfs/4), figsize = (20, 20))

    for f in range(fake_sites_3d.shape[0]):
        derived_counts = [sum(x) for x in zip(*fake_sites_3d[f])]
        sfs = allel.sfs(np.asarray(derived_counts).astype(int))
        allel.plot_sfs(sfs, ax=axes.flatten()[f]).set_yscale('linear')
    plt.setp(axes, xticks=[0,10,20,30,40,50,60])
    fig.set_size_inches(10, 10)
    fig.set_dpi(500)
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_sfs.png"))
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_sfs.svg"))

    plt.close(fig)

    
def plot_sfs(generator, curr_epoch, num_sfs, latent_size, data_size, odir, device, normalize):
    plt.rcParams.update({'font.size': 18})
   
    sfs_avg = 20 ######### number of sfs to average across
    z = torch.randn(num_sfs*sfs_avg, latent_size, 1, 1, device=device)
    #z = torch.randn(num_sfs*sfs_avg, latent_size, device=device)
    generator.eval()
    if normalize:
        fake_sites = generator(z).detach().cpu().numpy() / 2 + 0.5
    else:
        fake_sites = generator(z).detach().cpu().numpy()
    fake_sites[fake_sites < 0] = 0
    fake_sites = fake_sites[:,0:1,:,:]
    fake_sites = np.rint(fake_sites)
    fake_sites_3d = np.reshape(fake_sites,(num_sfs*sfs_avg,data_size,data_size))
    
    fig, axes = plt.subplots(nrows=int(num_sfs/4), ncols=int(num_sfs/4), figsize = (20, 20))
    for curr_sfs in range(num_sfs):
        sfs_sums = [[]]*sfs_avg
        temp_genomes=fake_sites_3d[(curr_sfs*sfs_avg):(sfs_avg*(curr_sfs+1))]
        for f in range(temp_genomes.shape[0]):
            sfs_sums[f] = ([sum(x) for x in zip(*temp_genomes[f])])
        sfs_array = [[]]*sfs_avg
        for f in range(np.array(sfs_sums).shape[0]):
            temp_sfs = allel.sfs(np.array(sfs_sums[f]).astype(int))
            if np.array(temp_sfs).shape[0] < (temp_genomes.shape[2]+1):
                sfs_array[f] = np.append(temp_sfs,np.zeros((temp_genomes.shape[2]+1)-np.array(temp_sfs).shape[0])).astype(int)
            else:
                sfs_array[f] = temp_sfs
        allel.plot_sfs(np.average(sfs_array,axis=0), ax=axes.flatten()[curr_sfs]).set_yscale('linear')
    plt.setp(axes, xticks=[0,10,20,30,40,50,60])
    fig.set_size_inches(10, 10)
    fig.set_dpi(500)
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_sfsavg20.png"))
    plt.close(fig)

def plot_sfs_avg(fake_sites, curr_epoch, in_dat, data_size, odir, device, return_sfs=True):
    plt.rcParams.update({'font.size': 18})

    fake_sites[fake_sites < 0] = 0
    fake_sites = np.rint(fake_sites)
    #fake_sites = add_singletons(fake_sites)
    fake_sites_3d = fake_sites[:,0,:,:]
    #fake_sites_3d = np.reshape(fake_sites,(num_sfs,data_size,data_size))
    sfs_array_fake = [[]]*fake_sites_3d.shape[0]
    for f in range(fake_sites_3d.shape[0]):
        sfs_sum = ([sum(x) for x in zip(*fake_sites_3d[f])])
        temp_sfs = allel.sfs(np.array(sfs_sum).astype(int))
        if np.array(temp_sfs).shape[0] < (fake_sites_3d.shape[2]+1):
            sfs_array_fake[f] = np.append(temp_sfs,np.zeros((fake_sites_3d.shape[2]+1)-np.array(temp_sfs).shape[0])).astype(int)
        else:
            sfs_array_fake[f] = temp_sfs
    
    in_dat[in_dat < 0] = 0
    in_dat = in_dat[:,0,:,:]
    in_dat = np.rint(in_dat)

    sfs_array_real = [[]]*in_dat.shape[0]
    for f in range(in_dat.shape[0]):
        sfs_sum = ([sum(x) for x in zip(*in_dat[f])])
        temp_sfs = allel.sfs(np.array(sfs_sum).astype(int))
        if np.array(temp_sfs).shape[0] < (in_dat.shape[2]+1):
            sfs_array_real[f] = np.append(temp_sfs,np.zeros((in_dat.shape[2]+1)-np.array(temp_sfs).shape[0])).astype(int)
        else:
            sfs_array_real[f] = temp_sfs
    real_kwargs = dict()
    real_kwargs.setdefault('color', '#440154FF')
    real_kwargs.setdefault('alpha', 0.8)

    gen_kwargs = dict()
    gen_kwargs.setdefault('color', '#29AF7FFF')
    gen_kwargs.setdefault('alpha', 0.8)

    fig, axes = plt.subplots(figsize = (10, 10))
    sfs_real = np.average(sfs_array_real,axis=0)
    sfs_gen = np.average(sfs_array_fake,axis=0)
    axes.plot(np.arange(1, sfs_real.shape[0]-1), sfs_real[1:-1], color="#440154FF", label="Input", linewidth=3)
    axes.plot(np.arange(1, sfs_gen.shape[0]-1), sfs_gen[1:-1], color="#29AF7FFF", label="Generated", linewidth=3)
    axes.set_xlabel('Derived Allele Count')
    axes.set_ylabel('Site Frequency')
    axes.autoscale(axis='x', tight=True)
    axes.set_yscale('linear')
    axes.legend(prop={'size': 16})
#    allel.plot_sfs(np.average(sfs_array_real,axis=0),plot_kwargs=real_kwargs).set_yscale('linear')
#    allel.plot_sfs(np.average(sfs_array_fake,axis=0),plot_kwargs=gen_kwargs).set_yscale('linear')
    fig.suptitle("Input vs. Generated SFS", fontsize=16)
    #axes[1].set_title("Generated SFS")
    plt.setp(axes, xticks=[0,10,20,30,40,50,60])
    fig.set_size_inches(10, 10)
    fig.set_dpi(500)
    #plt.savefig(os.path.join(odir, str(curr_epoch) + "_sfs1kavg.png"))
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_sfs1kavg.png"))
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_sfs1kavg.svg"), format='svg')
    plt.close(fig)

    if return_sfs:
        return sfs_real[1:-1], sfs_gen[1:-1]
def plot_sfs_avg_bar(fake_sites, curr_epoch, in_dat, odir, return_sfs=True):
    plt.rcParams.update({'font.size': 18})

    fake_sites[fake_sites < 0] = 0
    fake_sites = np.rint(fake_sites)
    #fake_sites = add_singletons(fake_sites)
    fake_sites_3d = fake_sites[:,0,:,:]
    #fake_sites_3d = np.reshape(fake_sites,(num_sfs,data_size,data_size))
    sfs_array_fake = [[]]*fake_sites_3d.shape[0]
    for f in range(fake_sites_3d.shape[0]):
        sfs_sum = ([sum(x) for x in zip(*fake_sites_3d[f])])
        temp_sfs = allel.sfs(np.array(sfs_sum).astype(int))
        if np.array(temp_sfs).shape[0] < (fake_sites_3d.shape[2]+1):
            sfs_array_fake[f] = np.append(temp_sfs,np.zeros((fake_sites_3d.shape[2]+1)-np.array(temp_sfs).shape[0])).astype(int)
        else:
            sfs_array_fake[f] = temp_sfs
    
    in_dat[in_dat < 0] = 0
    in_dat = in_dat[:,0,:,:]
    in_dat = np.rint(in_dat)

    sfs_array_real = [[]]*in_dat.shape[0]
    for f in range(in_dat.shape[0]):
        sfs_sum = ([sum(x) for x in zip(*in_dat[f])])
        temp_sfs = allel.sfs(np.array(sfs_sum).astype(int))
        if np.array(temp_sfs).shape[0] < (in_dat.shape[2]+1):
            sfs_array_real[f] = np.append(temp_sfs,np.zeros((in_dat.shape[2]+1)-np.array(temp_sfs).shape[0])).astype(int)
        else:
            sfs_array_real[f] = temp_sfs
    real_kwargs = dict()
    real_kwargs.setdefault('color', '#440154FF')
    real_kwargs.setdefault('alpha', 0.8)

    gen_kwargs = dict()
    gen_kwargs.setdefault('color', '#29AF7FFF')
    gen_kwargs.setdefault('alpha', 0.8)

    fig, axes = plt.subplots(figsize = (10, 10))
    sfs_real = np.average(sfs_array_real,axis=0)
    sfs_gen = np.average(sfs_array_fake,axis=0)
    bar_width = 0.4
    axes.bar(np.arange(1, sfs_real.shape[0]-1) - bar_width/2, sfs_real[1:-1], color="#440154FF", width=bar_width, label="Input")
    axes.bar(np.arange(1, sfs_gen.shape[0]-1) + bar_width/2, sfs_gen[1:-1], color="#29AF7FFF", width=bar_width, label="Generated")
    axes.set_xlabel('Derived Allele Count')
    axes.set_ylabel('Site Frequency')
    axes.autoscale(axis='x', tight=True)
    axes.set_yscale('linear')
    axes.legend(prop={'size': 16})
#    allel.plot_sfs(np.average(sfs_array_real,axis=0),plot_kwargs=real_kwargs).set_yscale('linear')
#    allel.plot_sfs(np.average(sfs_array_fake,axis=0),plot_kwargs=gen_kwargs).set_yscale('linear')
    fig.suptitle("Input vs. Generated SFS", fontsize=16)
    #axes[1].set_title("Generated SFS")
    plt.setp(axes, xticks=[0,10,20,30,40,50,60])
    #plt.savefig(os.path.join(odir, str(curr_epoch) + "_sfs1kavg.png"))
    fig.set_size_inches(10, 10)
    fig.set_dpi(500)
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + '_sfs1k_bar.png'), format='png')
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + '_sfs1k_bar.svg'), format='svg')
    plt.close(fig)
    aa_truth, aa_synth = calc_aa_scores(sfs_array_real, sfs_array_fake)

    if return_sfs:
        return sfs_real[1:-1], sfs_gen[1:-1], aa_truth, aa_synth

def plot_sfs_subpop_avg(generator, curr_epoch, in_dat, latent_size, data_size, sub_1_size, sub_2_size, odir, device, normalize):
    plt.rcParams.update({'font.size': 18})
    in_dat = in_dat.cpu()
    ##### Averages across 1000 snps
    z = torch.randn(1000, latent_size, 1, 1, device=device)
    #z = torch.randn(1000, latent_size, device=device)
    generator.eval()
    
    if normalize:
        fake_sites = generator(z).detach().cpu().numpy() / 2 + 0.5
        in_dat = in_dat / 2 + 0.5
    else:
        fake_sites = generator(z).detach().cpu().numpy()
    fake_sites[fake_sites < 0] = 0
    #fake_sites = add_singletons(fake_sites)
    fake_sites_3d = fake_sites[:,0,:,:]
    fake_sites_3d = np.rint(fake_sites_3d)
    fake_sites_3d_sub_1 = fake_sites_3d[:,:sub_1_size,:]
    fake_sites_3d_sub_2 = fake_sites_3d[:,sub_1_size:(sub_1_size + sub_2_size),:]
    #fake_sites_3d = np.reshape(fake_sites,(num_sfs,data_size,data_size))
    sfs_array_fake_sub_1 = [[]]*fake_sites_3d.shape[0]
    sfs_array_fake_sub_2 = [[]]*fake_sites_3d.shape[0]

    for f in range(fake_sites_3d_sub_1.shape[0]):
        #### Check if derived allele is the main allele in this population, and switch if that's the case
        #for site in range(fake_sites_3d_sub_1.shape[2]):
        #    if sum(fake_sites_3d_sub_1[f,:,site]) > sub_1_size / 2:
        #        fake_sites_3d_sub_1[f,:,site] = np.logical_xor(fake_sites_3d_sub_1[f,:,site],1).astype(int)
        #### Calc sfs ###        
        sfs_sum = ([sum(x) for x in zip(*fake_sites_3d_sub_1[f])])
        temp_sfs = allel.sfs(np.array(sfs_sum).astype(int))
        if np.array(temp_sfs).shape[0] < (sub_1_size+1):
            sfs_array_fake_sub_1[f] = np.append(temp_sfs,np.zeros((sub_1_size+1)-np.array(temp_sfs).shape[0])).astype(int)
        else:
            sfs_array_fake_sub_1[f] = temp_sfs

    for f in range(fake_sites_3d_sub_2.shape[0]):
        #### Check if derived allele is the main allele in this population, and switch if that's the case
        #for site in range(fake_sites_3d_sub_2.shape[2]):
        #    if sum(fake_sites_3d_sub_2[f,:,site]) > sub_2_size / 2:
        #        fake_sites_3d_sub_2[f,:,site] = np.logical_xor(fake_sites_3d_sub_2[f,:,site],1).astype(int)
        #### Calc sfs ###
        sfs_sum = ([sum(x) for x in zip(*fake_sites_3d_sub_2[f])])
        temp_sfs = allel.sfs(np.array(sfs_sum).astype(int))
        if np.array(temp_sfs).shape[0] < (sub_2_size+1):
            sfs_array_fake_sub_2[f] = np.append(temp_sfs,np.zeros((sub_2_size+1)-np.array(temp_sfs).shape[0])).astype(int)
        else:
            sfs_array_fake_sub_2[f] = temp_sfs
    
    in_dat[in_dat < 0] = 0
    in_dat = in_dat[:,0,:,:].detach().numpy()
    in_dat = np.rint(in_dat)
    in_dat_sub_1 = in_dat[:,:sub_1_size,:]
    in_dat_sub_2 = in_dat[:,sub_1_size:(sub_1_size + sub_2_size),:]

    sfs_array_real_sub_1 = [[]]*in_dat.shape[0]
    sfs_array_real_sub_2 = [[]]*in_dat.shape[0]

    for f in range(in_dat_sub_1.shape[0]):
        #### Check if derived allele is the main allele in this population, and switch if that's the case
        #for site in range(in_dat_sub_1.shape[2]):
        #    if sum(in_dat_sub_1[f,:,site]) > sub_1_size / 2:
        #        in_dat_sub_1[f,:,site] = np.logical_xor(in_dat_sub_1[f,:,site],1).astype(int)
        #### Calc sfs ###
        sfs_sum = ([sum(x) for x in zip(*in_dat_sub_1[f])])
        temp_sfs = allel.sfs(np.array(sfs_sum).astype(int))
        if np.array(temp_sfs).shape[0] < (sub_1_size+1):
            sfs_array_real_sub_1[f] = np.append(temp_sfs,np.zeros((sub_1_size+1)-np.array(temp_sfs).shape[0])).astype(int)
        else:
            sfs_array_real_sub_1[f] = temp_sfs

    for f in range(in_dat_sub_2.shape[0]):
        #### Check if derived allele is the main allele in this population, and switch if that's the case
        #for site in range(in_dat_sub_2.shape[2]):
        #    if sum(in_dat_sub_2[f,:,site]) > sub_2_size / 2:
        #        in_dat_sub_2[f,:,site] = np.logical_xor(in_dat_sub_2[f,:,site],1).astype(int)
        #### Calc sfs ###
        sfs_sum = ([sum(x) for x in zip(*in_dat_sub_2[f])])
        temp_sfs = allel.sfs(np.array(sfs_sum).astype(int))
        if np.array(temp_sfs).shape[0] < (sub_2_size+1):
            sfs_array_real_sub_2[f] = np.append(temp_sfs,np.zeros((sub_2_size+1)-np.array(temp_sfs).shape[0])).astype(int)
        else:
            sfs_array_real_sub_2[f] = temp_sfs

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (20, 10))
    allel.plot_sfs(np.average(sfs_array_real_sub_1,axis=0), ax=axes.flatten()[0], label="Input Subpopulation 1").set_yscale('linear')
    allel.plot_sfs(np.average(sfs_array_fake_sub_1,axis=0), ax=axes.flatten()[0], label="Generated Subpopulation 1").set_yscale('linear')
    allel.plot_sfs(np.average(sfs_array_real_sub_2,axis=0), ax=axes.flatten()[1], label="Input Subpopulation 2").set_yscale('linear')
    allel.plot_sfs(np.average(sfs_array_fake_sub_2,axis=0), ax=axes.flatten()[1], label="Generated Subpopulation 2").set_yscale('linear')
    axes[0].set_title("Subpopulation 1")
    axes[1].set_title("Subpopulation 2")
    axes[0].legend()
    axes[1].legend()
    plt.setp(axes, xticks=[0,10,20,30])
    fig.set_size_inches(10, 10)
    fig.set_dpi(500)
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_sfs_subpop_1kavg.png"))
    plt.close(fig)

def plot_subpop_joint_sfs(fake_sites, curr_epoch, in_dat, data_size, sub_1_size, sub_2_size, odir, device):
    plt.rcParams.update({'font.size': 18})

    fake_sites[fake_sites < 0] = 0
    #fake_sites = add_singletons(fake_sites)
    fake_sites_3d = fake_sites[:,0,:,:]
    fake_sites_3d = np.rint(fake_sites_3d)
    fake_sites_3d_sub_1 = fake_sites_3d[:,:sub_1_size,:]
    fake_sites_3d_sub_2 = fake_sites_3d[:,sub_1_size:(sub_1_size + sub_2_size),:]
    #fake_sites_3d = np.reshape(fake_sites,(num_sfs,data_size,data_size))
    #sfs_array_fake_sub_1 = [[]]*fake_sites_3d.shape[0]
    #sfs_array_fake_sub_2 = [[]]*fake_sites_3d.shape[0]

    sfs_array_fake_sub_1 = []
    sfs_array_fake_sub_2 = []

    for f in range(fake_sites_3d_sub_1.shape[0]):
        #### Check if derived allele is the main allele in this population, and switch if that's the case
        #for site in range(fake_sites_3d_sub_1.shape[2]):
        #    if sum(fake_sites_3d_sub_1[f,:,site]) > sub_1_size / 2:
        #        fake_sites_3d_sub_1[f,:,site] = np.logical_xor(fake_sites_3d_sub_1[f,:,site],1).astype(int)
        #### Calc sfs ###        
        sfs_sum_sub_1 = ([sum(x) for x in zip(*fake_sites_3d_sub_1[f])])
        sfs_array_fake_sub_1 = sfs_array_fake_sub_1 + sfs_sum_sub_1

        sfs_sum_sub_2 = ([sum(x) for x in zip(*fake_sites_3d_sub_2[f])])
        sfs_array_fake_sub_2 = sfs_array_fake_sub_2 + sfs_sum_sub_2

    
    in_dat[in_dat < 0] = 0
    in_dat = in_dat[:,0,:,:].detach().numpy()
    in_dat = np.rint(in_dat)
    in_dat_sub_1 = in_dat[:,:sub_1_size,:]
    in_dat_sub_2 = in_dat[:,sub_1_size:(sub_1_size + sub_2_size),:]

    sfs_array_real_sub_1 = []
    sfs_array_real_sub_2 = []

    for f in range(in_dat_sub_1.shape[0]):
        #### Check if derived allele is the main allele in this population, and switch if that's the case
        #for site in range(in_dat_sub_1.shape[2]):
        #    if sum(in_dat_sub_1[f,:,site]) > sub_1_size / 2:
        #        in_dat_sub_1[f,:,site] = np.logical_xor(in_dat_sub_1[f,:,site],1).astype(int)
        #### Calc sfs ###
        sfs_sum_sub_1 = ([sum(x) for x in zip(*in_dat_sub_1[f])])
        sfs_array_real_sub_1 = sfs_array_real_sub_1 + sfs_sum_sub_1

        sfs_sum_sub_2 = ([sum(x) for x in zip(*in_dat_sub_2[f])])
        sfs_array_real_sub_2 = sfs_array_real_sub_2 + sfs_sum_sub_2


    input_joint_sfs = allel.joint_sfs(np.asarray(sfs_array_real_sub_1).astype(int), np.asarray(sfs_array_real_sub_2).astype(int))
    generated_joint_sfs = allel.joint_sfs(np.asarray(sfs_array_fake_sub_1).astype(int), np.asarray(sfs_array_fake_sub_2).astype(int))

    plt.viridis()

    imshow_kwargs = dict()
    imshow_kwargs.setdefault('cmap', 'viridis')
    imshow_kwargs.setdefault('interpolation', 'none')
    imshow_kwargs.setdefault('aspect', 'auto')
    imshow_kwargs.setdefault('norm', LogNorm())

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (20, 10))

    allel.plot_joint_sfs(input_joint_sfs, ax=axes.flatten()[0], imshow_kwargs=imshow_kwargs)
    allel.plot_joint_sfs(generated_joint_sfs, ax=axes.flatten()[1], imshow_kwargs=imshow_kwargs)


    axes[0].set_title("Input")
    axes[1].set_title("Generated")
    fig.set_size_inches(20, 10)
    fig.set_dpi(500)
    #plt.setp(axes, xticks=[0,10,20,30])
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_subpop_joint_sfs_1kavg.png"))
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_subpop_joint_sfs_1kavg.svg"), format='svg')
    plt.close(fig)

def plot_position_vector_same(fake_pos, curr_epoch, in_pos, odir):
    plt.rcParams.update({'font.size': 18})

    fake_pos = fake_pos[:,0,:]
    in_pos = in_pos[:,0,:]
    ######## Plotting #############
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize = (10, 10))
    for i in range(fake_pos.shape[0]):
        axes.plot([x+1 for x in list(range(in_pos.shape[1]))], in_pos[i], color = '#440154FF', alpha = 0.1)
        #axes.plot([x+1 for x in list(range(fake_pos.shape[1]))], fake_pos[i], color = '#29AF7FFF', alpha = 0.1)
    axes.set_xlabel("Site")
    axes.set_ylabel("Position")
    fig.set_size_inches(10, 10)
    fig.set_dpi(500)
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + '_pos_in.png'), format='png')
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + '_pos_in.svg'), format='svg')
    plt.close(fig)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize = (10, 10))
    for i in range(fake_pos.shape[0]):
        #axes.plot([x+1 for x in list(range(in_pos.shape[1]))], in_pos[i], color = '#440154FF', alpha = 0.1)
        axes.plot([x+1 for x in list(range(fake_pos.shape[1]))], fake_pos[i], color = '#29AF7FFF', alpha = 0.1)
    axes.set_xlabel("Site")
    axes.set_ylabel("Position")
    fig.set_size_inches(10, 10)
    fig.set_dpi(500)
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + '_pos_gen.png'), format='png')
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + '_pos_gen.svg'), format='svg')
    plt.close(fig)


    ######## Plotting #############
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize = (10, 10))
    for i in range(100):
        axes.plot([x+1 for x in list(range(in_pos.shape[1]))], in_pos[i], color = '#440154FF', alpha = 0.2)
        axes.plot([x+1 for x in list(range(fake_pos.shape[1]))], fake_pos[i], color = '#29AF7FFF', alpha = 0.2)
    axes.set_xlabel("Site")
    axes.set_ylabel("Position")
    fig.set_size_inches(10, 10)
    fig.set_dpi(500)
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + '_pos_both.png'), format='png')
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + '_pos_both.svg'), format='svg')
    plt.close(fig)

def plot_position_vector(fake_pos, curr_epoch, in_pos, data_size, odir, device):
    plt.rcParams.update({'font.size': 18})

    fake_pos = fake_pos[:,0,:]
    in_pos = in_pos[:,0,:]
    ######## Plotting #############
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (20, 10))
    for i in range(fake_pos.shape[0]):
        axes[0].plot([x+1 for x in list(range(in_pos.shape[1]))], in_pos[i], color = 'black', alpha = 0.1)
        axes[1].plot([x+1 for x in list(range(fake_pos.shape[1]))], fake_pos[i], color = 'black', alpha = 0.1)
    axes[0].set_title("Input")
    axes[1].set_title("Generated")
    axes[0].set_xlabel("Site")
    axes[0].set_ylabel("Position")
    axes[1].set_xlabel("Site")
    axes[1].set_ylabel("Position")
    fig.set_size_inches(20, 10)
    fig.set_dpi(500)
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_positions.png"))
    plt.close(fig)

def plot_ld(fake_sites, fake_pos, curr_epoch, in_sites, in_pos, data_size, odir, device):
    plt.rcParams.update({'font.size': 18})
    

    fake_sites[fake_sites < 0] = 0
    fake_sites = fake_sites[:,0,:,:]
    fake_sites = np.rint(fake_sites).astype(int) 

    fake_pos = fake_pos * 10000
    fake_pos_sorted, fake_pos_indices = torch.sort(torch.tensor(fake_pos))
    fake_pos_sorted = np.rint(fake_pos_sorted.cpu().numpy()).astype(int)

    input_sites = in_sites[:,0,:,:].detach().cpu().numpy()
    input_sites = np.rint(input_sites).astype(int)
    input_positions = in_pos
    input_positions = input_positions * 10000
    input_pos_sorted, input_pos_indices = torch.sort(input_positions)
    input_pos_sorted = np.rint(input_pos_sorted.detach().cpu().numpy()).astype(int)

    generated_ld_array = []
    #random_gen_ld_array = [] ### also plot LD array w/ random sorted positions

    input_ld_array = []
    #random_input_ld_array = [] ### also plot LD array w/ random sorted positions

    ### Calc LD decay
    for i in range(fake_sites.shape[0]):
        generated_ld_array.append(calc_binned_ld_mean(fake_sites[i], fake_pos_sorted[i], 0, 10000, 500))
        input_ld_array.append(calc_binned_ld_mean(input_sites[i], input_pos_sorted[i], 0, 10000, 500))
        #rand_pos = np.random.random(64) * 10000
        #rand_pos.sort()
        #random_input_ld_array.append(calc_binned_ld_mean(input_sites[i], rand_pos, 0, 10000, 500))

    generated_ld_mean = np.mean(generated_ld_array,0)
    generated_ld_stdev = np.std(generated_ld_array,0)

    input_ld_mean = np.mean(input_ld_array,0)
    input_ld_stdev = np.std(input_ld_array,0)


    #### plot real and generated 
    fig, ax1 = plt.subplots()
    ax1.errorbar(np.arange(0,1,0.05), input_ld_mean, yerr=input_ld_stdev, color="#440154FF", label="Input", linewidth=3, elinewidth=1, capsize=3, capthick=1)
    #ax1.errorbar(np.arange(0,1,0.05), random_input_ld_mean, yerr=random_input_ld_stdev, color="gray", label="random positions input")
    ax1.errorbar(np.arange(0,1,0.05)+ 0.01, generated_ld_mean, yerr=generated_ld_stdev, color="#29AF7FFF", label="Generated", linewidth=3, elinewidth=1, capsize=3, capthick=1)
    #ax1.errorbar(np.arange(0,1,0.05)+ 0.02, random_gen_ld_mean, yerr=random_gen_ld_stdev, color="chartreuse", label="random positions generated")
    ax1.set_title('LD Decay')
    ax1.legend()
    ax1.set_xlabel("SNP distance")
    ax1.set_ylabel(r'LD ($r^2$)')
    fig.set_size_inches(10, 10)
    fig.set_dpi(500)

    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_ld_decay.png"))
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_ld_decay.svg"), format='svg')

    plt.close(fig)

def plot_branch_length_distributions(fake_sites, curr_epoch, in_sites, data_size, odir, device):
    plt.rcParams.update({'font.size': 18})
    
    fake_sites[fake_sites < 0] = 0
    fake_sites_sites = fake_sites[:,0,:,:]

    #### Get branch lengths for generated alignments #####
    all_branch_lengths = []
    for aln in range(fake_sites.shape[0]):
        sample_align = np.rint(fake_sites_sites[aln,:,:]).astype('int').astype('str')
        sample_align[sample_align == '0'] = 'A'
        sample_align[sample_align == '1'] = 'T'
        alignment = MultipleSeqAlignment([])
        for i in range(sample_align.shape[1]):
            alignment.append(SeqRecord(''.join(sample_align[i,:]), str(i)))
        calculator = DistanceCalculator('identity')
        dm = calculator.get_distance(alignment)
        constructor = DistanceTreeConstructor(calculator, 'nj')
        starting_tree = constructor.build_tree(alignment)
        scorer = ParsimonyScorer()
        searcher = NNITreeSearcher(scorer)
        constructor = ParsimonyTreeConstructor(searcher, starting_tree)
        pars_tree = constructor.build_tree(alignment)
        branch_lengths = np.asarray(re.findall("branch_length=(.*?),", pars_tree.__str__())).astype('float32')
        all_branch_lengths.append(branch_lengths)

    generated_branch_lengths = np.vstack(all_branch_lengths).flatten()

    #### Get branch lengths for input alignments #####
    all_branch_lengths = []
    for aln in range(in_sites.shape[0]):
        sample_align = np.rint(in_sites[aln,0,:,:].numpy()).astype('int').astype('str')
        sample_align[sample_align == '0'] = 'A'
        sample_align[sample_align == '1'] = 'T'
        alignment = MultipleSeqAlignment([])
        for i in range(sample_align.shape[1]):
            alignment.append(SeqRecord(''.join(sample_align[i,:]), str(i)))
        calculator = DistanceCalculator('identity')
        dm = calculator.get_distance(alignment)
        constructor = DistanceTreeConstructor(calculator, 'nj')
        starting_tree = constructor.build_tree(alignment)
        scorer = ParsimonyScorer()
        searcher = NNITreeSearcher(scorer)
        constructor = ParsimonyTreeConstructor(searcher, starting_tree)
        pars_tree = constructor.build_tree(alignment)
        branch_lengths = np.asarray(re.findall("branch_length=(.*?),", pars_tree.__str__())).astype('float32')
        all_branch_lengths.append(branch_lengths)

    input_branch_lengths = np.vstack(all_branch_lengths).flatten()

    fig, ax1 = plt.subplots()
    ax1.set_title('Branch Length Distribution')
    ax1.set_xlabel("Branch Length")
    ax1.set_ylabel("Density")
    sns.kdeplot(input_branch_lengths, fill = True, color = '#440154FF', label = "Input")
    sns.kdeplot(generated_branch_lengths, fill = True, color = '#29AF7FFF', label = "Generated")
    ax1.legend()
    fig.set_size_inches(10, 10)
    fig.set_dpi(500)
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_branch_lengths.png"), bbox_inches="tight")
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_branch_lengths.svg"), bbox_inches="tight", format='svg')

    plt.close(fig)

def plot_sites_grid(fake_sites, curr_epoch, odir, device):

    fake_sites[fake_sites < 0] = 0
    fake_sites = fake_sites[:,0:1,:,:]
    fake_sites = np.rint(fake_sites)

    fake_sites_imgs = save_image(torch.tensor(fake_sites), os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_output_sites.png"), pad_value = 0.5)

    return fake_sites


def plot_sumstats_dist(fake_sites, fake_pos, in_sites, in_pos, curr_epoch, odir):
    plt.rcParams.update({'font.size': 18})

    fake_sites[fake_sites < 0] = 0
    fake_sites = fake_sites[:,0,:,:]
    fake_sites = np.rint(fake_sites)

    fake_positions = fake_pos * 10000
    fake_pos_sorted, fake_pos_indices = torch.sort(torch.tensor(fake_positions))
    fake_pos_sorted = np.rint(fake_pos_sorted.cpu().numpy())

    input_sites = in_sites[:,0,:,:].detach().cpu().numpy()

    input_positions = in_pos * 10000
    input_pos_sorted, input_pos_indices = torch.sort(input_positions)
    input_pos_sorted = np.rint(input_pos_sorted.detach().cpu().numpy())

    fake_pi = []
    fake_tajD = []
    fake_omega = []
    fake_fst = []

    in_pi = []
    in_tajD = []
    in_omega = []
    in_fst = []

    for aln in range(fake_sites.shape[0]):
            fake_ac = allel.HaplotypeArray(fake_sites[aln,:,:].astype(int)).count_alleles()
            in_ac = allel.HaplotypeArray(input_sites[aln,:,:].astype(int)).count_alleles()

            fake_pi.append(allel.sequence_diversity(fake_pos_sorted[aln,0,:].astype(int), fake_ac, start=1, stop=10001))
            fake_tajD.append(allel.tajima_d(fake_ac, fake_pos_sorted[aln,0,:].astype(int), start=1, stop=10001))
            fake_omega.append(calc_omega(fake_sites[aln,:,:].astype(int)))
            fake_fst.append(compute_fst(fake_sites[aln,:,:].astype(int)))

            in_pi.append(allel.sequence_diversity(input_pos_sorted[aln,0,:].astype(int), in_ac, start=1, stop=10001))
            in_tajD.append(allel.tajima_d(in_ac, input_pos_sorted[aln,0,:].astype(int), start=1, stop=10001))
            in_omega.append(calc_omega(input_sites[aln,:,:].astype(int)))
            in_fst.append(compute_fst(input_sites[aln,:,:].astype(int)))

    fig, ax1 = plt.subplots()
    #ax1.set_title("Sequence Diversity Distribution")
    #ax1.set_xlabel(r"Nucleotide Diversity ($\pi$)")
    ax1.set_title("Omega Distribution")
    ax1.set_xlabel(r"Kim and Nielsen's $\omega$")
    ax1.set_ylabel("Density")
    sns.kdeplot(in_omega, fill = True, color = '#440154FF', label = "Input")
    sns.kdeplot(fake_omega, fill = True, color = '#29AF7FFF', label = "Generated")
    ax1.legend()
    fig.set_size_inches(10, 10)
    fig.set_dpi(500) 
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_omega.png"), bbox_inches="tight")
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_omega.svg"), bbox_inches="tight", format='svg')

    plt.close(fig)

    fig, ax1 = plt.subplots()
    ax1.set_title("Tajima's D Distribution")
    ax1.set_xlabel("Tajima's D")
    ax1.set_ylabel("Density")
    sns.kdeplot(in_tajD, fill = True, color = '#440154FF', label = "Input")
    sns.kdeplot(fake_tajD, fill = True, color = '#29AF7FFF', label = "Generated")
    ax1.legend()
    fig.set_size_inches(10, 10)
    fig.set_dpi(500)
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_tajD.png"), bbox_inches="tight")
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_tajD.svg"), bbox_inches="tight", format='svg')

    plt.close(fig)

    fig, ax1 = plt.subplots()
    ax1.set_title("Pi Distribution")
    ax1.set_xlabel(r"Nucleotide Diversity ($\pi$)")
    ax1.set_ylabel("Density")
    sns.kdeplot(in_pi, fill = True, color = '#440154FF', label = "Input")
    sns.kdeplot(fake_pi, fill = True, color = '#29AF7FFF', label = "Generated")
    ax1.legend()
    fig.set_size_inches(10, 10)
    fig.set_dpi(500)
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_pi.png"), bbox_inches="tight")
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_pi.svg"), bbox_inches="tight", format='svg')
 
    plt.close(fig)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('$F_{ST}$')
    ax1.set_ylabel("Density")
    sns.kdeplot(in_fst, fill = True, color = '#440154FF', label = "Input")
    sns.kdeplot(fake_fst, fill = True, color = '#29AF7FFF', label = "Generated")
    ax1.legend()
    fig.set_size_inches(10, 10)
    fig.set_dpi(500)
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_fst.png"), bbox_inches="tight")
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_fst.svg"), bbox_inches="tight", format='svg')

    plt.close(fig)

def plot_windowed_stats(fake_sites, fake_pos, in_sites, in_pos, curr_epoch, odir):
    plt.rcParams.update({'font.size': 18})

    fake_sites[fake_sites < 0] = 0
    fake_sites_sites = fake_sites[:,0,:,:]
    fake_sites = np.rint(fake_sites)

    fake_positions = fake_pos * 10000
    fake_pos_sorted, fake_pos_indices = torch.sort(torch.tensor(fake_positions))
    fake_pos_sorted = np.rint(fake_pos_sorted.cpu().numpy())

    input_sites = in_sites[:,0,:,:].detach().cpu().numpy()

    input_positions = in_pos * 10000
    input_pos_sorted, input_pos_indices = torch.sort(input_positions)
    input_pos_sorted = np.rint(input_pos_sorted.detach().cpu().numpy())

    fake_pi_array = []
    fake_theta_array = []
    fake_tajD_array = []
    fake_omega_array = []

    in_pi_array = []
    in_theta_array = []
    in_tajD_array = []
    in_omega_array = []


    for aln in range(fake_sites.shape[0]):
        fake_pi, fake_windows_pi, fake_n_bases_pi, fake_counts_pi = allel.windowed_diversity(fake_pos_sorted[aln,0,:], fake_sites[aln,0,:,:].astype(int), size=500, start=1, stop=10001,step=500)
        fake_theta, fake_windows_theta, fake_n_bases_theta, fake_counts_theta = allel.windowed_watterson_theta(fake_pos_sorted[aln,0,:], fake_sites[aln,0,:,:].astype(int), size=500, start=1, stop=10001,step=500)
        fake_tajD, fake_windows_tajD, fake_counts_tajD = allel.windowed_tajima_d(fake_pos_sorted[aln,0,:], fake_sites[aln,0,:,:].astype(int), size=500, start=1, stop=10001,step=500)
        fake_omega, fake_windows_omega, fake_counts_omega = calc_windowed_omega(fake_sites[aln,0,:,:].astype(int), fake_pos_sorted[aln,0,:], 3000, start=1, stop=10001, step=1000)

        in_pi, in_windows_pi, in_n_bases_pi, in_counts_pi = allel.windowed_diversity(input_pos_sorted[aln,0,:], input_sites[aln,:,:].astype(int), size=500, start=1, stop=10001,step=500)
        in_theta, in_windows_theta, in_n_bases_theta, in_counts_theta = allel.windowed_watterson_theta(input_pos_sorted[aln,0,:].astype(int), input_sites[aln,:,:], size=500, start=1, stop=10001,step=500)
        in_tajD, in_windows_tajD, in_counts_tajD = allel.windowed_tajima_d(input_pos_sorted[aln,0,:], input_sites[aln,:,:].astype(int), size=500, start=1, stop=10001,step=500)
        in_omega, in_windows_omega, in_counts_omega = calc_windowed_omega(input_sites[aln,:,:], input_pos_sorted[aln,0,:], 3000, start=1, stop=10001, step=1000)
        
        fake_pi_array.append(fake_pi)
        fake_theta_array.append(fake_theta)
        fake_tajD_array.append(fake_tajD)
        fake_omega_array.append(fake_omega)

        in_pi_array.append(in_pi)
        in_theta_array.append(in_theta)
        in_tajD_array.append(in_tajD)
        in_omega_array.append(in_omega)


    fake_pi_array_mean = np.mean(fake_pi_array,0)
    fake_pi_array_sd = np.std(fake_pi_array,0)
    fake_theta_array_mean = np.mean(fake_theta_array,0)
    fake_theta_array_sd = np.std(fake_theta_array,0)
    fake_tajD_array_mean = np.mean(fake_tajD_array,0)
    fake_tajD_array_sd = np.std(fake_tajD_array,0)
    fake_omega_array_mean = np.nanmean(fake_omega_array,0)
    fake_omega_array_sd = np.nanstd(fake_omega_array,0)

    in_pi_array_mean = np.mean(in_pi_array,0)
    in_pi_array_sd = np.std(in_pi_array,0)
    in_theta_array_mean = np.mean(in_theta_array,0)
    in_theta_array_sd = np.std(in_theta_array,0)
    in_tajD_array_mean = np.mean(in_tajD_array,0)
    in_tajD_array_sd = np.std(in_tajD_array,0)
    in_omega_array_mean = np.mean(in_omega_array,0)
    in_omega_array_sd = np.std(in_omega_array,0)

    pl_idx = np.cumsum(fake_n_bases)
        
    fig, ax1 = plt.subplots()
    ax1.errorbar(pl_idx / pl_idx[-1], in_pi_array_mean, yerr=in_pi_array_sd, color="gray", label="Input")
    ax1.errorbar(pl_idx / pl_idx[-1] + 0.01, fake_pi_array_mean, yerr=fake_pi_array_sd, color="chartreuse", label="Generated")
    ax1.set_title('Windowed Nucleotide Diversity')
    ax1.legend()
    ax1.set_xlabel("Position")
    ax1.set_ylabel(r"Nucleotide Diversity ($\pi$)")
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_windowed_pi.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax1 = plt.subplots()
    ax1.errorbar(pl_idx / pl_idx[-1], in_theta_array_mean, yerr=in_theta_array_sd, color="gray", label="Input")
    ax1.errorbar(pl_idx / pl_idx[-1] + 0.01, fake_theta_array_mean, yerr=fake_theta_array_sd, color="chartreuse", label="Generated")
    ax1.set_title('Windowed Theta')
    ax1.legend()
    ax1.set_xlabel("Position")
    ax1.set_ylabel(r"Watterson's Theta ($\theta$)")
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_windowed_theta.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax1 = plt.subplots()
    ax1.errorbar(pl_idx / pl_idx[-1], in_tajD_array_mean, yerr=in_tajD_array_sd, color="gray", label="Input")
    ax1.errorbar(pl_idx / pl_idx[-1] + 0.01, fake_tajD_array_mean, yerr=fake_tajD_array_sd, color="chartreuse", label="Generated")
    ax1.set_title("Windowed Tajima's D")
    ax1.legend()
    ax1.set_xlabel("Position")
    ax1.set_ylabel("Tajima's D")
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_windowed_tajD.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax1 = plt.subplots()
    ax1.errorbar((fake_windows_omega[:,0]+((fake_windows_omega[:,1] - b[:,0])/2))/fake_windows_omega[-1,1], in_omega_array_mean, yerr=in_omega_array_sd, color="gray", label="Input")
    ax1.errorbar((fake_windows_omega[:,0]+((fake_windows_omega[:,1] - b[:,0])/2))/fake_windows_omega[-1,1] + 0.01, fake_omega_array_mean, yerr=fake_omega_array_sd, color="chartreuse", label="Generated")
    ax1.set_title('Windowed Omega')
    ax1.legend()
    ax1.set_xlabel("Position")
    ax1.set_ylabel(r"Kim and Nielsen's ($\omega$)")
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_windowed_omega.png"), bbox_inches="tight")
    plt.close(fig)

def plot_windowed_stats_paneled(fake_sites, fake_pos, in_sites, in_pos, curr_epoch, odir):
    plt.rcParams.update({'font.size': 18})

    fake_sites[fake_sites < 0] = 0
    fake_sites = fake_sites[:,0,:,:]
    fake_sites = np.rint(fake_sites)

    fake_positions = fake_pos * 10000
    fake_pos_sorted, fake_pos_indices = torch.sort(torch.tensor(fake_positions))
    fake_pos_sorted = np.rint(fake_pos_sorted.cpu().numpy())

    input_sites = in_sites[:,0,:,:].detach().cpu().numpy()

    input_positions = in_pos * 10000
    input_pos_sorted, input_pos_indices = torch.sort(input_positions)
    input_pos_sorted = np.rint(input_pos_sorted.detach().cpu().numpy())

    fake_pi_array = []
    fake_theta_array = []
    fake_tajD_array = []
    fake_omega_array = []

    in_pi_array = []
    in_theta_array = []
    in_tajD_array = []
    in_omega_array = []

    for aln in range(fake_sites.shape[0]):

        fake_ac = allel.HaplotypeArray(fake_sites[aln,:,:].astype(int)).count_alleles()
        in_ac = allel.HaplotypeArray(input_sites[aln,:,:].astype(int)).count_alleles()

        fake_pi, fake_windows_pi, fake_n_bases_pi, fake_counts_pi = allel.windowed_diversity(fake_pos_sorted[aln,0,:].astype(int), fake_ac, size=500, start=1, stop=10001,step=500)
        fake_theta, fake_windows_theta, fake_n_bases_theta, fake_counts_theta = allel.windowed_watterson_theta(fake_pos_sorted[aln,0,:].astype(int), fake_ac, size=500, start=1, stop=10001,step=500)
        fake_tajD, fake_windows_tajD, fake_counts_tajD = allel.windowed_tajima_d(fake_pos_sorted[aln,0,:].astype(int), fake_ac, size=500, start=1, stop=10001,step=500)
        fake_omega, fake_windows_omega, fake_counts_omega = calc_windowed_omega(fake_sites[aln,:,:].astype(int), fake_pos_sorted[aln,0,:].astype(int), 3000, start=1, stop=10001, step=500)
        
        in_pi, in_windows_pi, in_n_bases_pi, in_counts_pi = allel.windowed_diversity(input_pos_sorted[aln,0,:].astype(int), in_ac, size=500, start=1, stop=10001,step=500)
        in_theta, in_windows_theta, in_n_bases_theta, in_counts_theta = allel.windowed_watterson_theta(input_pos_sorted[aln,0,:].astype(int), in_ac, size=500, start=1, stop=10001,step=500)
        in_tajD, in_windows_tajD, in_counts_tajD = allel.windowed_tajima_d(input_pos_sorted[aln,0,:].astype(int), in_ac, size=500, start=1, stop=10001,step=500)
        in_omega, in_windows_omega, in_counts_omega = calc_windowed_omega(input_sites[aln,:,:].astype(int), input_pos_sorted[aln,0,:].astype(int), 3000, start=1, stop=10001, step=500)            
        
        fake_pi_array.append(fake_pi)
        fake_theta_array.append(fake_theta)
        fake_tajD_array.append(fake_tajD)
        fake_omega_array.append(fake_omega)
        
        in_pi_array.append(in_pi)
        in_theta_array.append(in_theta)
        in_tajD_array.append(in_tajD)
        in_omega_array.append(in_omega)


    fake_pi_array_mean = np.nanmean(fake_pi_array,0)
    fake_pi_array_sd = np.nanstd(fake_pi_array,0)
    fake_theta_array_mean = np.nanmean(fake_theta_array,0)
    fake_theta_array_sd = np.nanstd(fake_theta_array,0)
    fake_tajD_array_mean = np.nanmean(fake_tajD_array,0)
    fake_tajD_array_sd = np.nanstd(fake_tajD_array,0)
    fake_omega_array_mean = np.nanmean(fake_omega_array,0)
    fake_omega_array_sd = np.nanstd(fake_omega_array,0)

    in_pi_array_mean = np.nanmean(in_pi_array,0)
    in_pi_array_sd = np.nanstd(in_pi_array,0)
    in_theta_array_mean = np.nanmean(in_theta_array,0)
    in_theta_array_sd = np.nanstd(in_theta_array,0)
    in_tajD_array_mean = np.nanmean(in_tajD_array,0)
    in_tajD_array_sd = np.nanstd(in_tajD_array,0)
    in_omega_array_mean =np.nanmean(in_omega_array,0)
    in_omega_array_sd = np.nanstd(in_omega_array,0)

    pl_idx = np.cumsum(fake_n_bases_pi)

    fig, ax1 = plt.subplots(nrows=2, ncols=2, figsize = (20, 20))

    ax1[0,0].errorbar(pl_idx / pl_idx[-1], in_pi_array_mean, yerr=in_pi_array_sd, color="#440154FF", label="Input", linewidth=3, elinewidth=1, capsize=3, capthick=1)
    ax1[0,0].errorbar(pl_idx / pl_idx[-1] + 0.01, fake_pi_array_mean, yerr=fake_pi_array_sd, color="#29AF7FFF", label="Generated", linewidth=3, elinewidth=1, capsize=3, capthick=1)
    ax1[0,0].set_xlabel("Position", fontsize=16)
    ax1[0,0].set_ylabel(r"Nucleotide Diversity ($\pi$)", fontsize=16)
    ax1[0,0].legend()

    ax1[0,1].errorbar(pl_idx / pl_idx[-1], in_theta_array_mean, yerr=in_theta_array_sd, color="#440154FF", label="Input", linewidth=3, elinewidth=1, capsize=3, capthick=1)
    ax1[0,1].errorbar(pl_idx / pl_idx[-1] + 0.01, fake_theta_array_mean, yerr=fake_theta_array_sd, color="#29AF7FFF", label="Generated", linewidth=3, elinewidth=1, capsize=3, capthick=1)
    ax1[0,1].set_xlabel("Position", fontsize=16)
    ax1[0,1].set_ylabel(r"Watterson's Theta ($\theta$)", fontsize=16)


    ax1[1,0].errorbar(pl_idx / pl_idx[-1], in_tajD_array_mean, yerr=in_tajD_array_sd, color="#440154FF", label="Input", linewidth=3, elinewidth=1, capsize=3, capthick=1)
    ax1[1,0].errorbar(pl_idx / pl_idx[-1] + 0.01, fake_tajD_array_mean, yerr=fake_tajD_array_sd, color="#29AF7FFF", linewidth=3, elinewidth=1, capsize=3, capthick=1)
    ax1[1,0].set_xlabel("Position", fontsize=16)
    ax1[1,0].set_ylabel("Tajima's D", fontsize=16)

    ax1[1,1].errorbar((fake_windows_omega[:,0]+((fake_windows_omega[:,1] - fake_windows_omega[:,0])/2))/fake_windows_omega[-1,1], in_omega_array_mean, yerr=in_omega_array_sd, color="#440154FF", label="Input", linewidth=3, elinewidth=1, capsize=3, capthick=1)
    ax1[1,1].errorbar((fake_windows_omega[:,0]+((fake_windows_omega[:,1] - fake_windows_omega[:,0])/2))/fake_windows_omega[-1,1] + 0.01, fake_omega_array_mean, yerr=fake_omega_array_sd, color="#29AF7FFF", label="Generated", linewidth=3, elinewidth=1, capsize=3, capthick=1)
    ax1[1,1].set_xlabel("Position", fontsize=16)
    ax1[1,1].set_ylabel(r"Kim and Nielsen's $\omega$", fontsize=16)

    fig.suptitle('Windowed Diversity Statistics', fontsize=32)
    fig.set_size_inches(20, 20)
    fig.set_dpi(500)
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_windowed_stats_panel.png"), bbox_inches="tight", pad_inches=0.5)
    plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_windowed_stats_panel.svg"), bbox_inches="tight", pad_inches=0.5, format='svg')

    plt.close(fig)

def make_stairway_blueprint(fake_sfs, real_sfs, curr_epoch, odir):

    input_odir = os.path.join(odir, str(curr_epoch) + '/' + "stairway_in/")
    generated_odir = os.path.join(odir, str(curr_epoch) + '/' + "stairway_gen/")
    stairway_dir = "/proj/dschridelab/wwbooker/stairway_plot/stairway_plot_v2.1.1/stairway_plot_es/"

    if not os.path.exists(input_odir):
        os.mkdir(input_odir)
    else:
        os.system('rm -rf {0}'.format(input_odir))

    if not os.path.exists(generated_odir):
        os.mkdir(generated_odir)
    else:
        os.system('rm -rf {0}'.format(generated_odir))

    with open(os.path.join(input_odir, "input_" + str(curr_epoch) + "_stairway.blueprint"), 'w') as in_stairway_file:
        in_stairway_file.write("#ms blueprint\n")
        in_stairway_file.write("popid: input_"+ str(curr_epoch) +"\n")
        in_stairway_file.write("nseq: "+ str(len(real_sfs)+1) +"\n")
        in_stairway_file.write("L: 10000\n")
        in_stairway_file.write("whether_folded: false\n")
        in_stairway_file.write("SFS: "+'\t'.join(str(x) for x in real_sfs)+"\n")
        in_stairway_file.write("#smallest_size_of_SFS_bin_used_for_estimation: 1 # default is 1; to ignore singletons, uncomment this line and change this number to 2\n")
        in_stairway_file.write("#largest_size_of_SFS_bin_used_for_estimation: 63 # default is n-1; to ignore singletons, uncomment this line and change this number to nseq-2\n")
        in_stairway_file.write("pct_training: 0.67\n")
        in_stairway_file.write("nrand: 16 31 46 62\n")
        in_stairway_file.write("project_dir: "+str(input_odir)+"\n")
        in_stairway_file.write("stairway_plot_dir: "+str(stairway_dir)+"\n")
        in_stairway_file.write("ninput: 200\n")
        in_stairway_file.write("#random_seed: 6\n")
        in_stairway_file.write("#output setting\n")
        in_stairway_file.write("mu: 1.0e-8 # assumed mutation rate per site per generation\n")
        in_stairway_file.write("year_per_generation: 1 # assumed generation time (in years)\n")
        in_stairway_file.write("#plot setting\n")
        in_stairway_file.write("plot_title: " + "input_" + str(curr_epoch) + " # title of the plot\n")
        in_stairway_file.write("xrange: 0,0 # Time (1k year) range; format: xmin,xmax; 0,0 for default\n")
        in_stairway_file.write("yrange: 0,0 # Ne (1k individual) range; format: xmin,xmax; 0,0 for default\n")
        in_stairway_file.write("xspacing: 2 # X axis spacing\n")
        in_stairway_file.write("yspacing: 2 # Y axis spacing\n")
        in_stairway_file.write("fontsize: 12 # Font size\n")

    with open(os.path.join(input_odir, "run_input_stairway.sh"), 'w') as in_stairway_slurm_file:
        in_stairway_slurm_file.write("#!/bin/bash\n")
        in_stairway_slurm_file.write("#SBATCH -J stairway_plot\n")
        in_stairway_slurm_file.write("#SBATCH -p general\n")
        in_stairway_slurm_file.write("#SBATCH -N 1\n")
        in_stairway_slurm_file.write("#SBATCH -t 1-0\n")
        in_stairway_slurm_file.write("#SBATCH --mem=8g\n")
        in_stairway_slurm_file.write("#SBATCH --output=stairway_plot-%j.log\n\n")
        in_stairway_slurm_file.write("export CLASSPATH=$CLASSPATH:/proj/dschridelab/wwbooker/stairway_plot/stairway_plot_v2.1.1/stairway_plot_es/\n")
        in_stairway_slurm_file.write("java -cp /proj/dschridelab/wwbooker/stairway_plot/stairway_plot_v2.1.1/stairway_plot_es/ Stairbuilder " + "input_" + str(curr_epoch) + "_stairway.blueprint\n")
        in_stairway_slurm_file.write("bash " + "input_" + str(curr_epoch) + "_stairway.blueprint.sh\n")

    with open(os.path.join(generated_odir, "gen_" + str(curr_epoch) + "_stairway.blueprint"), 'w') as gen_stairway_file:
        gen_stairway_file.write("#ms blueprint\n")
        gen_stairway_file.write("popid: gen_"+ str(curr_epoch) +"\n")
        gen_stairway_file.write("nseq: "+ str(len(fake_sfs)+1) +"\n")
        gen_stairway_file.write("L: 10000\n")
        gen_stairway_file.write("whether_folded: false\n")
        gen_stairway_file.write("SFS: "+'\t'.join(str(x) for x in fake_sfs)+"\n")
        gen_stairway_file.write("#smallest_size_of_SFS_bin_used_for_estimation: 1 # default is 1; to ignore singletons, uncomment this line and change this number to 2\n")
        gen_stairway_file.write("#largest_size_of_SFS_bin_used_for_estimation: 63 # default is n-1; to ignore singletons, uncomment this line and change this number to nseq-2\n")
        gen_stairway_file.write("pct_training: 0.67\n")
        gen_stairway_file.write("nrand: 16 31 46 62\n")
        gen_stairway_file.write("project_dir: "+str(generated_odir)+"\n")
        gen_stairway_file.write("stairway_plot_dir: "+str(stairway_dir)+"\n")
        gen_stairway_file.write("ninput: 200\n")
        gen_stairway_file.write("#random_seed: 6\n")
        gen_stairway_file.write("#output setting\n")
        gen_stairway_file.write("mu: 1.0e-8 # assumed mutation rate per site per generation\n")
        gen_stairway_file.write("year_per_generation: 1 # assumed generation time (in years)\n")
        gen_stairway_file.write("#plot setting\n")
        gen_stairway_file.write("plot_title: " + "gen_" + str(curr_epoch) + " # title of the plot\n")
        gen_stairway_file.write("xrange: 0,0 # Time (1k year) range; format: xmin,xmax; 0,0 for default\n")
        gen_stairway_file.write("yrange: 0,0 # Ne (1k individual) range; format: xmin,xmax; 0,0 for default\n")
        gen_stairway_file.write("xspacing: 2 # X axis spacing\n")
        gen_stairway_file.write("yspacing: 2 # Y axis spacing\n")
        gen_stairway_file.write("fontsize: 12 # Font size\n")

    with open(os.path.join(generated_odir, "run_gen_stairway.sh"), 'w') as gen_stairway_slurm_file:
        gen_stairway_slurm_file.write("#!/bin/bash\n")
        gen_stairway_slurm_file.write("#SBATCH -J stairway_plot\n")
        gen_stairway_slurm_file.write("#SBATCH -p general\n")
        gen_stairway_slurm_file.write("#SBATCH -N 1\n")
        gen_stairway_slurm_file.write("#SBATCH -t 1-0\n")
        gen_stairway_slurm_file.write("#SBATCH --mem=8g\n")
        gen_stairway_slurm_file.write("#SBATCH --output=stairway_plot-%j.log\n\n")
        gen_stairway_slurm_file.write("export CLASSPATH=$CLASSPATH:/proj/dschridelab/wwbooker/stairway_plot/stairway_plot_v2.1.1/stairway_plot_es/\n")
        gen_stairway_slurm_file.write("java -cp /proj/dschridelab/wwbooker/stairway_plot/stairway_plot_v2.1.1/stairway_plot_es/ Stairbuilder " + "gen_" + str(curr_epoch) + "_stairway.blueprint\n")
        gen_stairway_slurm_file.write("bash " + "gen_" + str(curr_epoch) + "_stairway.blueprint.sh\n")


def make_stairway_blueprint_OoA2pop(fake_sites, in_dat, curr_epoch, odir):

    fake_sites[fake_sites < 0] = 0
    fake_sites = np.rint(fake_sites)
    #fake_sites = add_singletons(fake_sites)
    fake_sites_3d = fake_sites[:,0,:,:]
    #fake_sites_3d = np.reshape(fake_sites,(num_sfs,data_size,data_size))
    sfs_array_fake_1 = [[]]*fake_sites_3d.shape[0]
    sfs_array_fake_2 = [[]]*fake_sites_3d.shape[0]
    for f in range(fake_sites_3d.shape[0]):
        sfs_sum = ([sum(x) for x in zip(*fake_sites_3d[f,:32,:])])
        temp_sfs = allel.sfs(np.array(sfs_sum).astype(int))
        if np.array(temp_sfs).shape[0] < ((fake_sites_3d.shape[2]//2)+1):
            sfs_array_fake_1[f] = np.append(temp_sfs,np.zeros(((fake_sites_3d.shape[2]//2)+1)-np.array(temp_sfs).shape[0])).astype(int)
        else:
            sfs_array_fake_1[f] = temp_sfs

        sfs_sum = ([sum(x) for x in zip(*fake_sites_3d[f,32:,:])])
        temp_sfs = allel.sfs(np.array(sfs_sum).astype(int))
        if np.array(temp_sfs).shape[0] < ((fake_sites_3d.shape[2]//2)+1):
            sfs_array_fake_2[f] = np.append(temp_sfs,np.zeros(((fake_sites_3d.shape[2]//2)+1)-np.array(temp_sfs).shape[0])).astype(int)
        else:
            sfs_array_fake_2[f] = temp_sfs
    
    in_dat[in_dat < 0] = 0
    in_dat = in_dat[:,0,:,:]
    in_dat = np.rint(in_dat)

    sfs_array_real_1 = [[]]*in_dat.shape[0]
    sfs_array_real_2 = [[]]*in_dat.shape[0]
    for f in range(in_dat.shape[0]):
        sfs_sum = ([sum(x) for x in zip(*in_dat[f,:32,:])])
        temp_sfs = allel.sfs(np.array(sfs_sum).astype(int))
        if np.array(temp_sfs).shape[0] < ((in_dat.shape[2]//2)+1):
            sfs_array_real_1[f] = np.append(temp_sfs,np.zeros(((in_dat.shape[2]//2)+1)-np.array(temp_sfs).shape[0])).astype(int)
        else:
            sfs_array_real_1[f] = temp_sfs

        sfs_sum = ([sum(x) for x in zip(*in_dat[f,32:,:])])
        temp_sfs = allel.sfs(np.array(sfs_sum).astype(int))
        if np.array(temp_sfs).shape[0] < ((in_dat.shape[2]//2)+1):
            sfs_array_real_2[f] = np.append(temp_sfs,np.zeros(((in_dat.shape[2]//2)+1)-np.array(temp_sfs).shape[0])).astype(int)
        else:
            sfs_array_real_2[f] = temp_sfs
    
    sfs_fake_1 = np.average(sfs_array_fake_1,axis=0)
    sfs_fake_2 = np.average(sfs_array_fake_2,axis=0)
    sfs_real_1 = np.average(sfs_array_real_1,axis=0)
    sfs_real_2 = np.average(sfs_array_real_2,axis=0)

    sfs_fake_1 = sfs_fake_1[1:-1]
    sfs_fake_2 = sfs_fake_2[1:-1]
    sfs_real_1 = sfs_real_1[1:-1]
    sfs_real_2 = sfs_real_2[1:-1]

    input_odir = os.path.join(odir, str(curr_epoch) + '/' + "stairway_in/")
    generated_odir = os.path.join(odir, str(curr_epoch) + '/' + "stairway_gen/")
    stairway_dir = "/proj/dschridelab/wwbooker/stairway_plot/stairway_plot_v2.1.1/stairway_plot_es/"

    with open(os.path.join(input_odir, "input_pop1_" + str(curr_epoch) + "_stairway.blueprint"), 'w') as in_stairway_file:
        in_stairway_file.write("#ms blueprint\n")
        in_stairway_file.write("popid: input_pop1_"+ str(curr_epoch) +"\n")
        in_stairway_file.write("nseq: "+ str(len(sfs_real_1)+1) +"\n")
        in_stairway_file.write("L: 25400\n")
        in_stairway_file.write("whether_folded: false\n")
        in_stairway_file.write("SFS: "+'\t'.join(str(x) for x in sfs_real_1)+"\n")
        in_stairway_file.write("#smallest_size_of_SFS_bin_used_for_estimation: 1 # default is 1; to ignore singletons, uncomment this line and change this number to 2\n")
        in_stairway_file.write("#largest_size_of_SFS_bin_used_for_estimation: 31 # default is n-1; to ignore singletons, uncomment this line and change this number to nseq-2\n")
        in_stairway_file.write("pct_training: 0.67\n")
        in_stairway_file.write("nrand: 8 15 23 30\n")
        in_stairway_file.write("project_dir: "+str(input_odir)+"pop1\n")
        in_stairway_file.write("stairway_plot_dir: "+str(stairway_dir)+"\n")
        in_stairway_file.write("ninput: 200\n")
        in_stairway_file.write("#random_seed: 6\n")
        in_stairway_file.write("#output setting\n")
        in_stairway_file.write("mu: 2.36e-8 # assumed mutation rate per site per generation\n")
        in_stairway_file.write("year_per_generation: 25 # assumed generation time (in years)\n")
        in_stairway_file.write("#plot setting\n")
        in_stairway_file.write("plot_title: " + "input_pop1_" + str(curr_epoch) + " # title of the plot\n")
        in_stairway_file.write("xrange: 0,0 # Time (1k year) range; format: xmin,xmax; 0,0 for default\n")
        in_stairway_file.write("yrange: 0,0 # Ne (1k individual) range; format: xmin,xmax; 0,0 for default\n")
        in_stairway_file.write("xspacing: 2 # X axis spacing\n")
        in_stairway_file.write("yspacing: 2 # Y axis spacing\n")
        in_stairway_file.write("fontsize: 12 # Font size\n")

    with open(os.path.join(input_odir, "run_input_pop1_stairway.sh"), 'w') as in_stairway_slurm_file:
        in_stairway_slurm_file.write("#!/bin/bash\n")
        in_stairway_slurm_file.write("#SBATCH -J stairway_plot\n")
        in_stairway_slurm_file.write("#SBATCH -p general\n")
        in_stairway_slurm_file.write("#SBATCH -N 1\n")
        in_stairway_slurm_file.write("#SBATCH -t 1-0\n")
        in_stairway_slurm_file.write("#SBATCH --mem=8g\n")
        in_stairway_slurm_file.write("#SBATCH --output=stairway_plot-%j.log\n\n")
        in_stairway_slurm_file.write("export CLASSPATH=$CLASSPATH:/proj/dschridelab/wwbooker/stairway_plot/stairway_plot_v2.1.1/stairway_plot_es/\n")
        in_stairway_slurm_file.write("java -cp /proj/dschridelab/wwbooker/stairway_plot/stairway_plot_v2.1.1/stairway_plot_es/ Stairbuilder " + "input_pop1_" + str(curr_epoch) + "_stairway.blueprint\n")
        in_stairway_slurm_file.write("bash " + "input_pop1_" + str(curr_epoch) + "_stairway.blueprint.sh\n")

    with open(os.path.join(input_odir, "input_pop2_" + str(curr_epoch) + "_stairway.blueprint"), 'w') as in_stairway_file:
        in_stairway_file.write("#ms blueprint\n")
        in_stairway_file.write("popid: input_pop2_"+ str(curr_epoch) +"\n")
        in_stairway_file.write("nseq: "+ str(len(sfs_real_2)+1) +"\n")
        in_stairway_file.write("L: 25400\n")
        in_stairway_file.write("whether_folded: false\n")
        in_stairway_file.write("SFS: "+'\t'.join(str(x) for x in sfs_real_2)+"\n")
        in_stairway_file.write("#smallest_size_of_SFS_bin_used_for_estimation: 1 # default is 1; to ignore singletons, uncomment this line and change this number to 2\n")
        in_stairway_file.write("#largest_size_of_SFS_bin_used_for_estimation: 31 # default is n-1; to ignore singletons, uncomment this line and change this number to nseq-2\n")
        in_stairway_file.write("pct_training: 0.67\n")
        in_stairway_file.write("nrand: 8 15 23 30\n")
        in_stairway_file.write("project_dir: "+str(input_odir)+"pop2\n")
        in_stairway_file.write("stairway_plot_dir: "+str(stairway_dir)+"\n")
        in_stairway_file.write("ninput: 200\n")
        in_stairway_file.write("#random_seed: 6\n")
        in_stairway_file.write("#output setting\n")
        in_stairway_file.write("mu: 2.36e-8 # assumed mutation rate per site per generation\n")
        in_stairway_file.write("year_per_generation: 25 # assumed generation time (in years)\n")
        in_stairway_file.write("#plot setting\n")
        in_stairway_file.write("plot_title: " + "input_pop2_" + str(curr_epoch) + " # title of the plot\n")
        in_stairway_file.write("xrange: 0,0 # Time (1k year) range; format: xmin,xmax; 0,0 for default\n")
        in_stairway_file.write("yrange: 0,0 # Ne (1k individual) range; format: xmin,xmax; 0,0 for default\n")
        in_stairway_file.write("xspacing: 2 # X axis spacing\n")
        in_stairway_file.write("yspacing: 2 # Y axis spacing\n")
        in_stairway_file.write("fontsize: 12 # Font size\n")

    with open(os.path.join(input_odir, "run_input_pop2_stairway.sh"), 'w') as in_stairway_slurm_file:
        in_stairway_slurm_file.write("#!/bin/bash\n")
        in_stairway_slurm_file.write("#SBATCH -J stairway_plot\n")
        in_stairway_slurm_file.write("#SBATCH -p general\n")
        in_stairway_slurm_file.write("#SBATCH -N 1\n")
        in_stairway_slurm_file.write("#SBATCH -t 1-0\n")
        in_stairway_slurm_file.write("#SBATCH --mem=8g\n")
        in_stairway_slurm_file.write("#SBATCH --output=stairway_plot-%j.log\n\n")
        in_stairway_slurm_file.write("export CLASSPATH=$CLASSPATH:/proj/dschridelab/wwbooker/stairway_plot/stairway_plot_v2.1.1/stairway_plot_es/\n")
        in_stairway_slurm_file.write("java -cp /proj/dschridelab/wwbooker/stairway_plot/stairway_plot_v2.1.1/stairway_plot_es/ Stairbuilder " + "input_pop2_" + str(curr_epoch) + "_stairway.blueprint\n")
        in_stairway_slurm_file.write("bash " + "input_pop2_" + str(curr_epoch) + "_stairway.blueprint.sh\n")


    with open(os.path.join(generated_odir, "gen_pop1_" + str(curr_epoch) + "_stairway.blueprint"), 'w') as gen_stairway_file:
        gen_stairway_file.write("#ms blueprint\n")
        gen_stairway_file.write("popid: gen_pop1_"+ str(curr_epoch) +"\n")
        gen_stairway_file.write("nseq: "+ str(len(sfs_fake_1)+1) +"\n")
        gen_stairway_file.write("L: 25400\n")
        gen_stairway_file.write("whether_folded: false\n")
        gen_stairway_file.write("SFS: "+'\t'.join(str(x) for x in sfs_fake_1)+"\n")
        gen_stairway_file.write("#smallest_size_of_SFS_bin_used_for_estimation: 1 # default is 1; to ignore singletons, uncomment this line and change this number to 2\n")
        gen_stairway_file.write("#largest_size_of_SFS_bin_used_for_estimation: 31 # default is n-1; to ignore singletons, uncomment this line and change this number to nseq-2\n")
        gen_stairway_file.write("pct_training: 0.67\n")
        gen_stairway_file.write("nrand: 8 15 23 30\n")
        gen_stairway_file.write("project_dir: "+str(generated_odir)+"pop1\n")
        gen_stairway_file.write("stairway_plot_dir: "+str(stairway_dir)+"\n")
        gen_stairway_file.write("ninput: 200\n")
        gen_stairway_file.write("#random_seed: 6\n")
        gen_stairway_file.write("#output setting\n")
        gen_stairway_file.write("mu: 2.36e-8 # assumed mutation rate per site per generation\n")
        gen_stairway_file.write("year_per_generation: 25 # assumed generation time (in years)\n")
        gen_stairway_file.write("#plot setting\n")
        gen_stairway_file.write("plot_title: " + "gen_pop1_" + str(curr_epoch) + " # title of the plot\n")
        gen_stairway_file.write("xrange: 0,0 # Time (1k year) range; format: xmin,xmax; 0,0 for default\n")
        gen_stairway_file.write("yrange: 0,0 # Ne (1k individual) range; format: xmin,xmax; 0,0 for default\n")
        gen_stairway_file.write("xspacing: 2 # X axis spacing\n")
        gen_stairway_file.write("yspacing: 2 # Y axis spacing\n")
        gen_stairway_file.write("fontsize: 12 # Font size\n")

    with open(os.path.join(generated_odir, "run_gen_pop1_stairway.sh"), 'w') as gen_stairway_slurm_file:
        gen_stairway_slurm_file.write("#!/bin/bash\n")
        gen_stairway_slurm_file.write("#SBATCH -J stairway_plot\n")
        gen_stairway_slurm_file.write("#SBATCH -p general\n")
        gen_stairway_slurm_file.write("#SBATCH -N 1\n")
        gen_stairway_slurm_file.write("#SBATCH -t 1-0\n")
        gen_stairway_slurm_file.write("#SBATCH --mem=8g\n")
        gen_stairway_slurm_file.write("#SBATCH --output=stairway_plot-%j.log\n\n")
        gen_stairway_slurm_file.write("export CLASSPATH=$CLASSPATH:/proj/dschridelab/wwbooker/stairway_plot/stairway_plot_v2.1.1/stairway_plot_es/\n")
        gen_stairway_slurm_file.write("java -cp /proj/dschridelab/wwbooker/stairway_plot/stairway_plot_v2.1.1/stairway_plot_es/ Stairbuilder " + "gen_pop1_" + str(curr_epoch) + "_stairway.blueprint\n")
        gen_stairway_slurm_file.write("bash " + "gen_pop1_" + str(curr_epoch) + "_stairway.blueprint.sh\n")

    with open(os.path.join(generated_odir, "gen_pop2_" + str(curr_epoch) + "_stairway.blueprint"), 'w') as gen_stairway_file:
        gen_stairway_file.write("#ms blueprint\n")
        gen_stairway_file.write("popid: gen_pop2_"+ str(curr_epoch) +"\n")
        gen_stairway_file.write("nseq: "+ str(len(sfs_fake_2)+1) +"\n")
        gen_stairway_file.write("L: 25400\n")
        gen_stairway_file.write("whether_folded: false\n")
        gen_stairway_file.write("SFS: "+'\t'.join(str(x) for x in sfs_fake_2)+"\n")
        gen_stairway_file.write("#smallest_size_of_SFS_bin_used_for_estimation: 1 # default is 1; to ignore singletons, uncomment this line and change this number to 2\n")
        gen_stairway_file.write("#largest_size_of_SFS_bin_used_for_estimation: 31 # default is n-1; to ignore singletons, uncomment this line and change this number to nseq-2\n")
        gen_stairway_file.write("pct_training: 0.67\n")
        gen_stairway_file.write("nrand: 8 15 23 30\n")
        gen_stairway_file.write("project_dir: "+str(generated_odir)+"pop2\n")
        gen_stairway_file.write("stairway_plot_dir: "+str(stairway_dir)+"\n")
        gen_stairway_file.write("ninput: 200\n")
        gen_stairway_file.write("#random_seed: 6\n")
        gen_stairway_file.write("#output setting\n")
        gen_stairway_file.write("mu: 2.36e-8 # assumed mutation rate per site per generation\n")
        gen_stairway_file.write("year_per_generation: 25 # assumed generation time (in years)\n")
        gen_stairway_file.write("#plot setting\n")
        gen_stairway_file.write("plot_title: " + "gen_pop2_" + str(curr_epoch) + " # title of the plot\n")
        gen_stairway_file.write("xrange: 0,0 # Time (1k year) range; format: xmin,xmax; 0,0 for default\n")
        gen_stairway_file.write("yrange: 0,0 # Ne (1k individual) range; format: xmin,xmax; 0,0 for default\n")
        gen_stairway_file.write("xspacing: 2 # X axis spacing\n")
        gen_stairway_file.write("yspacing: 2 # Y axis spacing\n")
        gen_stairway_file.write("fontsize: 12 # Font size\n")

    with open(os.path.join(generated_odir, "run_gen_pop2_stairway.sh"), 'w') as gen_stairway_slurm_file:
        gen_stairway_slurm_file.write("#!/bin/bash\n")
        gen_stairway_slurm_file.write("#SBATCH -J stairway_plot\n")
        gen_stairway_slurm_file.write("#SBATCH -p general\n")
        gen_stairway_slurm_file.write("#SBATCH -N 1\n")
        gen_stairway_slurm_file.write("#SBATCH -t 1-0\n")
        gen_stairway_slurm_file.write("#SBATCH --mem=8g\n")
        gen_stairway_slurm_file.write("#SBATCH --output=stairway_plot-%j.log\n\n")
        gen_stairway_slurm_file.write("export CLASSPATH=$CLASSPATH:/proj/dschridelab/wwbooker/stairway_plot/stairway_plot_v2.1.1/stairway_plot_es/\n")
        gen_stairway_slurm_file.write("java -cp /proj/dschridelab/wwbooker/stairway_plot/stairway_plot_v2.1.1/stairway_plot_es/ Stairbuilder " + "gen_pop2_" + str(curr_epoch) + "_stairway.blueprint\n")
        gen_stairway_slurm_file.write("bash " + "gen_pop2_" + str(curr_epoch) + "_stairway.blueprint.sh\n")

def make_viridis_alignments(sites,curr_epoch,odir, input_align=False):
    fig, axes = plt.subplots(nrows = 8, ncols = 8)
    for i in range(len(axes.flatten())):
        im = axes.flatten()[i].imshow(sites[i])
        axes.flatten()[i].set_xticks([])
        axes.flatten()[i].set_yticks([])
        axes.flatten()[i].spines['top'].set_visible(False)
        axes.flatten()[i].spines['right'].set_visible(False)
        axes.flatten()[i].spines['bottom'].set_visible(False)
        axes.flatten()[i].spines['left'].set_visible(False)

    fig.tight_layout(pad=0.5, w_pad=0, h_pad=0.5)
    if input_align:
        plt.savefig(odir + "_viridis_alignments.png", dpi = 1000)
        plt.savefig(odir + "_viridis_alignments.svg", dpi = 1000, format = 'svg')
    else:
        plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_viridis_alignments.png"), dpi = 1000)
        plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_viridis_alignments.svg"), dpi = 1000, format = 'svg')
    plt.close(fig)

def make_center_line_alignments(sites,curr_epoch,odir, input_align=False):
    ins_col = np.repeat(0.5,64)
    ins_col.shape = (64,1)
    fig, axes = plt.subplots(nrows = 8, ncols = 8)
    for i in range(len(axes.flatten())):
        center_line_alignment = np.hstack((sites[i,:,:32], ins_col, sites[i,:,32:]))
        im = axes.flatten()[i].imshow(center_line_alignment)
        axes.flatten()[i].set_xticks([])
        axes.flatten()[i].set_yticks([])
        axes.flatten()[i].spines['top'].set_visible(False)
        axes.flatten()[i].spines['right'].set_visible(False)
        axes.flatten()[i].spines['bottom'].set_visible(False)
        axes.flatten()[i].spines['left'].set_visible(False)

    fig.tight_layout(pad=0.5, w_pad=0, h_pad=0.5)
    if input_align:
        plt.savefig(odir + "_viridis_center_alignments.png", dpi = 1000)
        plt.savefig(odir + "_viridis_center_alignments.svg", dpi = 1000, format = 'svg')
    else:
        plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_viridis_center_alignments.png"), dpi = 1000)
        plt.savefig(os.path.join(odir, str(curr_epoch) + "/" + str(curr_epoch) + "_viridis_center_alignments.svg"), dpi = 1000, format = 'svg')
    plt.close(fig)
