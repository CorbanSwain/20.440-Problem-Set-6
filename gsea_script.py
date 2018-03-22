#!/bin/python3
# gsea_script.py
# Corban Swain, March 2018

import numpy as np
import os
import matplotlib.pyplot as plt
import re

# Problem 1 A #
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
ptype_set = list(set(ptype))

n_genes = len(genes)
n_reads = len(ptype)
n_ptypes = len(ptype_set)
set_idxs = np.array([ptype_set.index(p) for p in ptype])
set_selects = [np.where(set_idxs == i)[0] for i, _ in enumerate(ptype_set)]
split_readings = [readings[:, sel] for sel in set_selects]

means = [np.mean(p_read, 1) for p_read in split_readings]
means = np.column_stack(tuple(means))
stds = [np.std(p_read, 1) for p_read in split_readings]
stds = np.column_stack((tuple(stds)))

snr = np.divide(np.abs(np.diff(means, axis=1)),
                np.sum(stds, axis=1, keepdims=True))

sort_idxs = np.squeeze(snr).argsort()
sort_idxs = np.flip(sort_idxs, 0)
ranked_snr = np.squeeze(snr[sort_idxs])
gene_dict = {}
for i, si in enumerate(sort_idxs):
    g = genes[si]
    gene_dict[g] = i
    print('%s - %6.3f' % (g, snr[si]))

# Problem 1-B & C
g_set = []
p_exp = 1


def enrichment_score(subset,
                     rank_dict=gene_dict,
                     ranked_scores=ranked_snr):
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


query = ['A', 'B', 'C']
es_abc, es_abc_max = enrichment_score(query)
print('ES for [A, B, C]: %5.2f' % es_abc_max)
plt.style.use('fivethirtyeight')
plt.figure(0)
plt.plot(es_abc)

# Problem 1-D
query_list = [['A', 'B', 'Y'],
              ['X', 'Y', 'Z'],
              ['A', 'H', 'O', 'V'],
              ['L', 'M', 'N', 'O']]

plt.figure(1)
for q in query_list:
    es, es_max = enrichment_score(q)
    plt.plot(es)
    q_str = '[' + ', '.join(q) + ']'
    print('ES for %15s is %5.2f' % (q_str, es_max))

plt.show()

# Problem 1-E
n_bootstraps = 1000
sort_gene_list = sorted(genes)
ref_ranks = np.arange(n_genes)
for q in query_list:
    _, es_obs = enrichment_score(q)
    es_null = np.zeros((n_bootstraps, ))
    for i in range(n_bootstraps):
        rand_ranks = np.copy(ref_ranks)
        np.random.shuffle(rand_ranks)
        rand_rank_dict = dict(zip(sort_gene_list, rand_ranks))
        _, es_null[i] = enrichment_score(q, rand_rank_dict)
    # FIXME - np.sign() will return zero for zero values, should zero be included?
    # could use             (        ...      in (np.sign(es_obs), 0))[0]
    es_null_valid = np.where(np.sign(es_null) == np.sign(es_obs))[0]
    es_null = es_null[es_null_valid]
    p_val = len(np.where(abs(es_null) > abs(es_obs))[0]) / len(es_null)
    q_str = '[' + ', '.join(q) + ']'
    print('Significance of S: %15s is %6.3f' % (q_str, p_val))
