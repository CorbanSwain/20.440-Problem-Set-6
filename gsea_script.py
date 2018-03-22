#!/bin/python3
# gsea_script.py
# Corban Swain, March 2018

# load required packages
import numpy as np
import os
import matplotlib.pyplot as plt
import re


# Problem 1 A #
# import data
datadir = 'data'
fname = 'mock_expression_data.txt'
ptype, genes, readings = ([], [], [])
with open(os.path.join(datadir, fname)) as data:
    for i_line, line in enumerate(data):
        if i_line is not 0:
            genes.append(line[0])
            numbers = re.split(r'\t+', line[1:].strip())
            numbers = [float(n) for n in numbers]
            readings.append(numbers)
        else:
            ptype = re.split(r'\t+', line.strip())
readings = np.array(readings)

# make a list of the different phenotypes
ptype_set = list(set(ptype))

# calculate some constants
n_genes = len(genes)
n_reads = len(ptype)
n_ptypes = len(ptype_set)

# get the column indices for the different phenotypes
set_idxs = np.array([ptype_set.index(p) for p in ptype])
set_selects = [np.where(set_idxs == i)[0] for i, _ in enumerate(ptype_set)]

# split up the readings into groups based on phenotype
split_readings = [readings[:, sel] for sel in set_selects]

# get the means and std. dev. across phenotype for each gene
means = [np.mean(p_read, 1) for p_read in split_readings]
means = np.column_stack(tuple(means))
stds = [np.std(p_read, 1) for p_read in split_readings]
stds = np.column_stack((tuple(stds)))

# calculate the snr for each gene
snr = np.divide(np.abs(np.diff(means, axis=1)),
                np.sum(stds, axis=1, keepdims=True))

# get the indices of the sorted genes, high snr to low
sort_idxs = np.squeeze(snr).argsort()
sort_idxs = np.flip(sort_idxs, 0)
ranked_snr = np.squeeze(snr[sort_idxs])

# make a dictionary with gene names for keys and ranks for values
gene_dict = {}
for i, si in enumerate(sort_idxs):
    g = genes[si]
    gene_dict[g] = i
    print('%s - %6.3f' % (g, snr[si]))


# Problem 1-B & C #
def enrichment_score(subset,
                     rank_dict=gene_dict,
                     ranked_scores=ranked_snr,
                     p_exp=1):
    """Calculate the running and max enrichment score (ES).

    :param subset: the subset of genes to analyze
    :param rank_dict: a dictionary with ranks for all genes
    :param ranked_scores: the correlation values in rank order
    :return: (np array of running ES, max ES)
    """
    n_subset = len(subset)
    n_genes = len(rank_dict)
    subset_ranks = np.array([rank_dict[g] for g in subset])
    subset_scores = ranked_scores[subset_ranks]
    subset_tot = np.sum(np.power(subset_scores, p_exp))
    miss_penalty = 1 / (n_genes - n_subset)
    deltas = np.zeros((n_genes,)) - miss_penalty
    deltas[subset_ranks] = np.power(subset_scores, p_exp) / subset_tot
    es = np.insert(np.cumsum(deltas), 0, 0)
    es_max = es[abs(es).argmax()]
    return es, es_max

# gene subset for part B
query = ['A', 'B', 'C']
es_abc, es_abc_max = enrichment_score(query)

# display enrichment score
print('ES for [A, B, C]: %5.2f' % es_abc_max)

# setup plot style
plt.style.use('seaborn-notebook')

# generate figure for running ES
plt.figure(0, (7, 4))
plt.plot(np.arange(0, 27), np.arange(0, 27) * 0, ':k')
plt.xlim((0, 26))
plt.plot(es_abc, label='[A, B, C]')
plt.legend()
plt.xlabel('Gene Rank')
plt.ylabel('Running Enrichment Score')
plt.tight_layout()
plt.savefig('figures/abc_es_fig.png', dpi=350)
plt.show(block=False)


# Problem 1-D
# four gene subsets
query_list = [['A', 'B', 'Y'],
              ['X', 'Y', 'Z'],
              ['A', 'H', 'O', 'V'],
              ['L', 'M', 'N', 'O']]

# plot running ES for each subset and print the max score
plt.figure(1, (7, 4))
for q in query_list:
    es, es_max = enrichment_score(q)
    q_str = '[' + ', '.join(q) + ']'
    plt.plot(es, label=q_str)
    print('ES for %15s is %5.2f' % (q_str, es_max))
plt.plot(np.arange(0, 27), np.arange(0, 27) * 0, ':k')
plt.xlim((0, 26))
plt.xlabel('Gene Rank')
plt.ylabel('Running Enrichment Score')
plt.legend()
plt.tight_layout()
plt.savefig('figures/multi_es_fig.png', dpi=350)
plt.show()

# Problem 1-E
# size of the null distribution
n_bootstraps = 1000

# prep gene list and ranks for scrambling
sort_gene_list = sorted(genes)
ref_ranks = np.arange(n_genes)

# loop through each of the four gene subsets
for q in query_list:
    _, es_obs = enrichment_score(q)

    # initialize an array of zeros
    es_null = np.zeros((n_bootstraps, ))

    # preform all the scrambled trials
    for i in range(n_bootstraps):
        rand_ranks = np.copy(ref_ranks)
        np.random.shuffle(rand_ranks)
        rand_rank_dict = dict(zip(sort_gene_list, rand_ranks))

        # extract the max ES from each trial
        _, es_null[i] = enrichment_score(q, rand_rank_dict)

    # select only values in the null distr. with the same sign as es_obs
    es_null_valid = np.where(np.sign(es_null) == np.sign(es_obs))[0]
    es_null = es_null[es_null_valid]

    # calculate the p-value for the observation by counting the number
    # of ESs in the null distr. greater than the observed ES
    p_val = len(np.where(abs(es_null) > abs(es_obs))[0]) / len(es_null)
    q_str = '[' + ', '.join(q) + ']'
    print('Significance of S: %15s is %6.3f' % (q_str, p_val))
