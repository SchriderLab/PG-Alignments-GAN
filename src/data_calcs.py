import numpy as np
from scipy.spatial.distance import squareform, cdist
import allel
from nn_adversarial_accuracy import *

def add_singletons(generated_sites):
    """
    adds singletons to a random individual for sites with all zero values

    Parameters
    ----------
    generated_sites: tensor, shape(n_alignments, n_channels, n_ind, n_sites)
        tensor as generated without added singletons

    Returns
    ----------
    generated_sites: tensor, shape(n_alignments, n_channels, n_ind, n_sites)
        tensor as generated with singletons added to ensure no sites have 0 for all individuals

    """
    for alignment in range(generated_sites.shape[0]):
        for site in range(generated_sites.shape[3]):
            if max(generated_sites[alignment,0,:,site]) == 0:
                generated_sites[alignment,0,np.random.randint(0,generated_sites.shape[2]),site] = int(1)
    
    return generated_sites

def calc_gametic_ld(sites_i, sites_j):
    """
    calculates gametic r2 values of linkage disequilibrium 

    Parameters
    ----------
    sites_i, sites_j: array_like, int, shape(n_sites)
        vector of derived and ancestral alleles (0 , 1) for two sites to calculate their r2 values

    Returns
    ----------
    r_2: float
        linkage disequilibrium as calculated between site i and j
    """
    if len(sites_i) != len(sites_j):
        raise ValueError("site lengths are not equal")
    
    if sites_i.sum() == 0 or sites_j.sum() == 0:
        raise ValueError("no variation at one or more sites")

    p_i = sites_i.sum() / len(sites_i)
    p_j = sites_j.sum() / len(sites_i)
    p_ij = np.logical_and(sites_i == 1, sites_j == 1).sum() / len(sites_i)
    
    #p_i = 0
    #p_j = 0
    #p_ij = 0
    #count = 0
    #for site in range(len(sites_i)):
    #    if sites_i[site] == 1:
    #        p_i += 1
    #    if sites_j[site] == 1:
    #       p_j +=1
    #    if sites_i[site] == 1 and sites_j[site] == 1:
    #        p_ij += 1
    #    count += 1
    
    #p_i = p_i / count
    #p_j = p_j / count
    #p_ij = p_ij / count    

    
    #p_ij = p_ij / len(sites_i)


    D_ij = p_ij - (p_i*p_j)
    num = (D_ij*D_ij)
    den = ((p_i*(1.0-p_i)) * (p_j*(1.0-p_j)))
    r_2 =  np.divide(num,den)
    
    return r_2

def get_ld_vec(alignment):
    """
    calculates gametic r2 values for every pairwise comparison in an alignment

    Parameters
    ----------
    alignment: array_like, int, shape(n_ind, n_sites)
        alignment of derived and ancestral alleles (0 , 1) for a number of individuals and sites

    Returns
    ----------
    r_2: ndarray, float, shape(n_sites * (n_sites - 1) // 2)
        linkage disequilibrium matrix in condensed form
    """

    r2 = []
    
    for i in range(alignment.shape[1]):
        for j in range(i+1, alignment.shape[1]):
            r2.append(calc_gametic_ld(alignment[:,i], alignment[:,j]))

    return r2

def calc_binned_ld_mean(alignment, positions, start, stop, window_size):
    """
    calculates r2 values binned in windows based on their proximity

    Parameters
    ----------
    alignment: array_like, int, shape(n_ind, n_sites)
        alignment of derived and ancestral alleles (0 , 1) for a number of individuals and sites
    
    positions: array_like, int, shape(n_sites)
        position of each site in the alignment

    start: int
        position to start the bins
    
    stop: int
        position to end the bins

    window_size:
        size of bins

    Returns
    ----------
    r_sq_mean: ndarray, shape(n_bins)
        average r2 value in each bin
    """

    #r = allel.rogers_huff_r(alignment.astype('int'))
    #r2 = r ** 2
    #ld_sq_form = squareform(r ** 2)

    r2 = get_ld_vec(alignment)

    dist_sq_form = cdist(positions.astype(int).reshape(-1,1),positions.astype(int).reshape(-1,1))
    dist_vec_form = squareform(dist_sq_form)
    dist_bins = np.arange(start,stop,window_size)
    idx = np.digitize(dist_vec_form,dist_bins,right=False)
    r_sq_sum = [0]*len(dist_bins)
    r_sq_mean = [0]*len(dist_bins)
    counts = [0]*len(dist_bins)
    for i in range(len(r2)):
        if not np.isnan(r2[i]):
            r_sq_sum[idx[i]-1] += r2[i]
            counts[idx[i]-1] += 1
    for i in range(len(counts)):
        if counts[i] > 0:
            r_sq_mean[i] = r_sq_sum[i]/counts[i]
    
    return r_sq_mean
    

def calc_omega(sites):

    """
    calculates the omega statistic from Kim and Nielsen (2004) used to detect selection by linkage disequilibrium

    Parameters
    ----------
    sites: array_like, int, shape(n_ind, n_sites)
        alignment of derived and ancestral alleles (0 , 1) for a number of individuals and sites

    Returns
    ----------
    omega_max: float
        maximum omega value in the alignment
    """

    omega_max = 0
    S_ = sites.shape[1]

    r = get_ld_vec(sites.astype(int))
    r2_sq_form = squareform(r)
    
    if S_ < 4:

        omega_max = 0

    else:

        for l_ in range(2, S_-1):

            sum_r2_L = 0
            sum_r2_R = 0
            sum_r2_LR = 0
            
            for i in range(S_):
                for j in range(i+1, S_):
                    #ld_calc = calc_gametic_ld(sites[:,i], sites[:,j])
                    ld_calc = r2_sq_form[i,j]
                    if i < l_ and j < l_:
                        sum_r2_L += ld_calc

                    elif i >= l_ and j >= l_:
                        sum_r2_R += ld_calc

                    elif i < l_ and j >= l_:
                        sum_r2_LR += ld_calc

            #l_ ## to keep the math right outside of indexing
            omega_numerator = (1 / ((l_*(l_-1)/2) + ((S_ - l_) * (S_ - l_ - 1)/2))) * (sum_r2_L + sum_r2_R)
            #omega_numerator = (1/(scipy.special.binom(l_, 2) + scipy.special.binom(S_ - l_, 2))) * (sum_r2_L + sum_r2_R)
            omega_denominator = (1/(l_ * (S_ - l_))) * sum_r2_LR
            
            if omega_denominator == 0:
                omega = 0
            else:
                omega = np.divide(omega_numerator, omega_denominator)

            if omega > omega_max:
                omega_max = omega
    
    return omega_max

def calc_windowed_omega(sites, positions, window_size, start=None, stop=None, step=None, fill=np.nan):
    """
    calculate omega in windows. Code modified from scikit-allel windowed_statistic.

    Parameters
    ----------
    sites: array_like, int, shape(n_ind, n_sites)
        alignment of derived and ancestral alleles (0 , 1) for a number of individuals and sites
    
    positions: ndarray, int, shape(n_sites)
        position of each site in the alignment

    start: int
        position to start the bins
    
    stop: int
        position to end the bins

    window_size: int
        size of bins
    
    step: int
        size of steps for window calculation

    fill: int
        what to fill in if the window is empty

    Returns
    ----------
    out : ndarray, shape (n_windows)
        omega in each window.

    windows : ndarray, int, shape (n_windows, 2)
        The windows used, as an array of (window_start, window_stop) positions,
        using 1-based coordinates.

    counts : ndarray, int, shape (n_windows)
        The number of items in each window.
    """
    windows = allel.position_windows(positions, window_size, start, stop, step)
    locs = allel.window_locations(positions, windows)

    out = []
    counts = []

    # iterate over windows
    for start_idx, stop_idx in locs:

        # calculate number of values in window
        n = stop_idx - start_idx

        if n == 0:
            # window is empty
            omega_max = fill

        else:

            omega_max = calc_omega(sites[:,start_idx:stop_idx])

        # store outputs
        out.append(omega_max)
        counts.append(n)

    # convert to arrays for output
    return np.asarray(out), windows, np.asarray(counts)

def compute_fst(alignment):
    """
    FST (for two populations)
    https://scikit-allel.readthedocs.io/en/stable/stats/fst.html
    """
    nvar = alignment.shape[1]

    pop_1 = alignment[:alignment.shape[0]//2,]
    pop_2 = alignment[alignment.shape[0]//2:,]

    pop_1_p = pop_1.sum(axis=0)
    pop_1_q = pop_1.shape[0] - pop_1_p
    pop_1_p.reshape(nvar,1)
    pop_1_q.reshape(nvar,1)
    pop_1_ac = np.dstack((pop_1_p, pop_1_q)).reshape(nvar,2)

    pop_2_p = pop_2.sum(axis=0)
    pop_2_q = pop_2.shape[0] - pop_2_p
    pop_2_p.reshape(nvar,1)
    pop_2_q.reshape(nvar,1)
    pop_2_ac = np.dstack((pop_2_p, pop_2_q)).reshape(nvar,2)

    num, den = allel.hudson_fst(pop_1_ac, pop_2_ac)
    fst = np.sum(num) / np.sum(den)
    return fst

def calc_aa_scores(in_sfs, gen_sfs):
    """
    calculates nearest neighbor adversarial accuracy for input and generated examples

    Parameters
    ----------
    in_sfs: array_like, int
        site frequency spectrum of input alignments

    gen_sfs: array_like, int
        site frequency spectrum of generated alignments

    Returns
    ----------
    aa_truth: float
        adversarial accuracy for input indexed

    aa_synth: float
        adversarial accuracy for generated indexed

    """

    nn_t = NearestNeighbors(n_neighbors=1).fit(np.asarray(in_sfs))
    nn_s = NearestNeighbors(n_neighbors=1).fit(np.asarray(gen_sfs))
    nn_t_t = nn_t.kneighbors()[0]
    nn_s_s = nn_s.kneighbors()[0]
    nn_t_s = nn_t.kneighbors(np.asarray(gen_sfs))[0]
    nn_s_t = nn_s.kneighbors(np.asarray(gen_sfs))[0]
    aa_truth = np.mean(nn_t_s > nn_t_t)
    aa_synth = np.mean(nn_s_t > nn_s_s)
    aa_ts = (aa_truth + aa_synth) / 2


    return aa_truth, aa_synth, aa_ts