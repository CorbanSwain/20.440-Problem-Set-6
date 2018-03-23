#!/bin/python3
# pca_script.py
# Corban Swain, March 2018

# import required packages
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import re


# load in data
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


# Problem 2-1 #
def scatter_plot(x, y, xlabel, ylabel):
    """Creates a scatter plot with axis labels."""
    plt.scatter(x, y, s=8 ** 2, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


# extract the first two  raw data columns
ax1 = signal_data[:, 0]
ax1_label = pps_signal[0]
ax2 = signal_data[:, 1]
ax2_label = pps_signal[1]

# set plot defaults
plt.style.use('seaborn-notebook')

# scatter plot raw data
fignum = 0
plt.figure(fignum, (6, 6))
plt.subplot(111, aspect='equal')
scatter_plot(ax1, ax2, ax1_label, ax2_label)
plt.savefig('figures/basic_data_fig.png', dpi=350)
# plt.show()


# Problem 2-2 #
# standardize data by first nudging up then taking the log
log_2_data = np.log2(signal_data + 0.001)

# then  by z-scoring the log-transformed data
means = np.mean(log_2_data, 0)
stdevs = np.std(log_2_data, 0)
std_data = (log_2_data - means) / stdevs

# create another scatter plot with the standardized data
ax1 = std_data[:, 0]
ax2 = std_data[:, 1]
fignum += 1
plt.figure(fignum, (6, 6))
plt.subplot(111, aspect='equal')
scatter_plot(ax1, ax2, ax1_label, ax2_label)
plt.savefig('figures/std_data_fig.png', dpi=350)


# Problem 2-3 #
# Performing principle component analysis calculation
n_dims = np.size(std_data, 1)
pca = PCA(n_components=n_dims, svd_solver='full')

# the data in PC space
score = pca.fit_transform(std_data)

# the loading coefficients for each PC in feature space
coeff = pca.components_.T

# ratio of the variance explained by each axis
explained = pca.explained_variance_ratio_

# Making a Scree plot, the % of variance vs PC number
fignum += 1
plt.figure(fignum, (7, 4))
plt.plot(np.arange(len(explained)) + 1, explained * 100,
         'o-', markersize=7, clip_on=False, zorder=100)
plt.xlim((1, len(explained)))
plt.ylim((0, 30))
plt.xlabel('Principal Component Number')
plt.ylabel('% Variance Explained')
plt.tight_layout()
plt.savefig('figures/pc_explained_fig.png', dpi=350)


# functions for making grouped scatter plots
def group_selecs(items, item_set=None):
    """
    Gets the slices for a categorical lists.

    :param items: list with category names
    :param item_set: ordered non-repeating list of categories
    :return: (item_set, slices for each category in item_set)
    """
    if item_set is None:
        item_set = list(set(items))
        item_set.sort()
    sels = [np.where(items == i)[0] for i in item_set]
    return item_set, sels


xy = (np.arange(-100, 100) * 0, np.arange(-100, 100))


def plot_xy_axes():
    """Plots the lines x=0 and y=0 in black."""
    plt.autoscale(False)
    plt.plot(xy[0], xy[1], 'k-', zorder=0, linewidth=0.75)
    plt.plot(xy[1], xy[0], 'k-', zorder=0, linewidth=0.75)


# Get categorical selections
drug_set, drug_sels = group_selecs(drugs)
cytokine_set, cytokine_sels = group_selecs(cytokines)

# Plot data along the first two principal components
# coloring by drug
fignum += 1
plt.figure(fignum, (13, 6))
plt.subplot(121, aspect='equal')
for i, (label, sel) in enumerate(zip(drug_set, drug_sels)):
    x = score[sel, 0]
    y = score[sel, 1]
    if i < 6: mk = 'o'
    else: mk = 'v'
    plt.plot(x, y, mk, label=label)
plot_xy_axes()
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(title='Drugs', bbox_to_anchor=(0.5, 1.02), loc=8,
           ncol=4, fontsize='x-small')

# and coloring by cytokine
plt.subplot(122, aspect='equal')
for label, sel in zip(cytokine_set, cytokine_sels):
    x = score[sel, 0]
    y = score[sel, 1]
    plt.plot(x, y, 'o', label=label)
plot_xy_axes()
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(title='Cytokines', bbox_to_anchor=(0.5, 1.02), loc=8,
           ncol=3, fontsize='x-small')
pps_set, pps_sels = group_selecs(pps)
st_set = ['0.33hr', '4hr', '24hr', '48hr', 'lateAvg', 'integral']
st_set, st_sels = group_selecs(signal_type, st_set)
plt.savefig('figures/pc_scores_fig.png', dpi=350)

# plot the coefficients of the first to principal components
# color by phosphoprotein
fignum += 1
plt.figure(fignum, (13, 6))
plt.subplot(121, aspect='equal')
for i, (label, sel) in enumerate(zip(pps_set, pps_sels)):
    x = coeff[sel, 0]
    y = coeff[sel, 1]
    if i < 6: mk = 'o'
    elif i < 12: mk = 'v'
    else: mk = 's'
    plt.plot(x, y, mk, label=label)
plot_xy_axes()
plt.xlabel('Loadings for PC 1')
plt.ylabel('Loadings for PC 2')
plt.legend(title='Phosphoproteins', bbox_to_anchor=(0.5, 1.02), loc=8,
           ncol=5, fontsize='x-small')

# and color by signal type
plt.subplot(122, aspect='equal')
for i, (label, sel) in enumerate(zip(st_set, st_sels)):
    x = coeff[sel, 0]
    y = coeff[sel, 1]
    if i < 6: mk = 'o'
    elif i < 12: mk = 'v'
    else: mk = 's'
    plt.plot(x, y, mk, label=label)
plot_xy_axes()
plt.xlabel('Loadings for PC 1')
plt.ylabel('Loadings for PC 2')
plt.legend(title='Signal types', bbox_to_anchor=(0.5, 1.02), loc=8,
           ncol=3, fontsize='x-small')
plt.savefig('figures/pc_loadings_fig.png', dpi=350)

# rebuild the data just from the first two principal components
data_approx = np.matmul(score[:, 0:2], coeff.T[0:2, :])

# choose a feature with low loadings
pick_near = 'p-p90RSK_24hr'

# and one with high loadings
pick_far = 'p-p38_48hr'

# extract the column indices for each of those
near_idx = np.where(pps_signal == pick_near)[0]
far_idx = np.where(pps_signal == pick_far)[0]

# plot the approx. data vs. the actual data for each test point
fignum += 1
plt.figure(fignum, (13, 6))
plt.subplot(121, aspect='equal')
plt.plot(np.arange(-3, 5), np.arange(-3, 5), 'k:')
plt.plot(std_data[:, near_idx], data_approx[:, near_idx], 'o')
plt.title('Comparison for %s, LOW PC Loadings' % pick_near)
plt.xlabel('Actual Reading (Normalized)')
plt.ylabel('Approximated Reading')
plt.xlim((-2.5, 3.5))
plt.ylim((-2, 3))
plot_xy_axes()
plt.subplot(122, aspect='equal')
plt.plot(np.arange(-3, 5), np.arange(-3, 5), 'k:')
plt.plot(std_data[:, far_idx], data_approx[:, far_idx], 'o')
plt.title('Comparison for %s, HIGH PC Loadings' % pick_far)
plt.xlabel('Actual Reading (Normalized)')
plt.ylabel('Approximated Reading')
plt.xlim((-1.5, 3.5))
plt.ylim((-1.5, 2.5))
plot_xy_axes()
plt.savefig('figures/data_approx_compare_fig.png', dpi=350)
plt.show()