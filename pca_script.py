#!/bin/python3
# pca_script.py
# Corban Swain, March 2018


import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import re

datadir = 'data'


def load_tab_list(fname):
    full_path = os.path.join(datadir, fname + '.txt')
    with open(full_path) as file:
        raw_str = file.read().strip('\'')
        str_list = re.split(r'\'\t+\'', raw_str)
    return np.array(str_list)

def load_line_list(fname):
    full_path = os.path.join(datadir, fname + '.txt')
    with open(full_path) as file:
        str_lst = [s[1:-2] if s[-1] is '\n' else s[1:-1]
                   for s in file.readlines()]
        return np.array(str_lst)

def load_matrix(fname):
    full_path = os.path.join(datadir, fname + '.txt')
    with open(full_path) as file:
        values = []
        for line in file.readlines():
            nums = re.split(r'\t+', line)
            nums = [float(num) for num in nums]
            values.append(nums)
        return np.array(values)


pps_signal = load_tab_list('cosgrove_phosphoprotein_signal')
pps = load_tab_list('cosgrove_phosphoproteins')
signal_type = load_tab_list('cosgrove_signal_type')
conditions = load_line_list('cosgrove_conditions')
cytokines = load_line_list('cosgrove_cytokines')
drugs = load_line_list('cosgrove_drugs')
signal_data = load_matrix('cosgrove_x')

ax1 = signal_data[:, 0]
ax1_label = pps_signal[0]
ax2 = signal_data[:, 1]
ax2_label = pps_signal[1]


def scatter_plot(x, y, xlabel, ylabel):
    plt.scatter(x, y, s=8 ** 2, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


plt.style.use('seaborn-notebook')
plt.figure(0)
plt.subplot(111, aspect='equal')
scatter_plot(ax1, ax2, ax1_label, ax2_label)

log_2_data = np.log2(signal_data + 0.001)
means = np.mean(log_2_data, 0)
stdevs = np.std(log_2_data, 0)
std_data = (log_2_data - means) / stdevs

ax1 = std_data[:, 0]
ax2 = std_data[:, 1]
plt.figure(1)
plt.subplot(111, aspect='equal')
scatter_plot(ax1, ax2, ax1_label, ax2_label)

# perform PCA
n_dims = np.size(std_data, 1)
pca = PCA(n_components=n_dims, svd_solver='full')
score = pca.fit_transform(std_data)
print('score.shape = ' + str(score.shape))
coeff = pca.components_.T
print('coeff.shape = ' + str(coeff.shape))
explained = pca.explained_variance_ratio_
print('explained.shape = ' + str(explained.shape))


def group_selecs(items):
    item_set = list(set(items))
    item_set.sort()
    sels = [np.where(items == i)[0] for i in item_set]
    return item_set, sels


drug_set, drug_sels = group_selecs(drugs)
cytokine_set, cytokine_sels = group_selecs(cytokines)

plt.figure(2)
ax = plt.subplot(121, aspect='equal')
for i, (label, sel) in enumerate(zip(drug_set, drug_sels)):
    x = score[sel, 0]
    y = score[sel, 1]
    if i < 6: mk = 'o'
    else: mk = 'v'
    plt.plot(x, y, mk, label=label)
ax.grid(True, linestyle='--')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(title='Drugs', bbox_to_anchor=(0.5, 1.02), loc=8,
           ncol=4, fontsize='x-small')
ax = plt.subplot(122, aspect='equal')
for label, sel in zip(cytokine_set, cytokine_sels):
    x = score[sel, 0]
    y = score[sel, 1]
    plt.plot(x, y, 'o', label=label)
ax.grid(True, linestyle='--')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(title='Cytokines', bbox_to_anchor=(0.5, 1.02), loc=8,
           ncol=3, fontsize='x-small')

pps_set, pps_sels = group_selecs(pps)
st_set, st_sels = group_selecs(signal_type)
plt.figure(3)
ax = plt.subplot(121, aspect='equal')
for i, (label, sel) in enumerate(zip(pps_set, pps_sels)):
    x = coeff[sel, 0]
    y = coeff[sel, 1]
    if i < 6: mk = 'o'
    elif i < 12: mk = 'v'
    else: mk = 's'
    plt.plot(x, y, mk, label=label)
ax.grid(True, linestyle='--')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(title='Phosphoproteins', bbox_to_anchor=(0.5, 1.02), loc=8,
           ncol=5, fontsize='x-small')
ax = plt.subplot(122, aspect='equal')
for i, (label, sel) in enumerate(zip(st_set, st_sels)):
    x = coeff[sel, 0]
    y = coeff[sel, 1]
    if i < 6: mk = 'o'
    elif i < 12: mk = 'v'
    else: mk = 's'
    plt.plot(x, y, mk, label=label)
ax.grid(True, linestyle='--')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(title='Signal types', bbox_to_anchor=(0.5, 1.02), loc=8,
           ncol=3, fontsize='x-small')

data_approx = np.matmul(score[:, 0:2], coeff.T[0:2, :])
pick_near = 'p-p90RSK_24hr'
pick_far = 'p-p38_48hr'
near_idx = np.where(pps_signal == pick_near)[0]
far_idx = np.where(pps_signal == pick_far)[0]

plt.figure(4)
ax = plt.subplot(121, aspect='equal')
ax.grid(True, linestyle='--')
plt.plot(np.arange(-3, 5), np.arange(-3, 5), 'k:')
plt.plot(std_data[:, near_idx], data_approx[:, near_idx], 'o')
plt.title('Comparison for %s, LOW PC Loadings' % pick_near)
plt.xlabel('Actual Reading (Normalized)')
plt.ylabel('Approximated Reading')
plt.xlim((-2.5, 3.5))
plt.ylim((-1.5, 2.5))
ax = plt.subplot(122, aspect='equal')
ax.grid(True, linestyle='--')
plt.plot(np.arange(-3, 5), np.arange(-3, 5), 'k:')
plt.plot(std_data[:, far_idx], data_approx[:, far_idx], 'o')
plt.title('Comparison for %s, HIGH PC Loadings' % pick_far)
plt.xlabel('Actual Reading (Normalized)')
plt.ylabel('Approximated Reading')
plt.xlim((-1.5, 3.5))
plt.ylim((-1.5, 2.5))
plt.show()