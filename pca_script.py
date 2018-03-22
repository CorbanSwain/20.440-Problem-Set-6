#!/bin/python3
# pca_script.py
# Corban Swain, March 2018

import os
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
import numpy as np
import re

datadir = 'data'


def load_cell_arr(fname):
    full_path = os.path.join(datadir, fname + '.txt')
    with open(full_path) as file:
        raw_str = file.read().strip('\'')
        str_list = re.split(r'\'\t+\'', raw_str)
    return str_list


def load_matrix(fname):
    full_path = os.path.join(datadir, fname + '.txt')
    with open(full_path) as file:
        values = []
        for line in file.readlines():
            nums = re.split(r'\t+', line)
            nums = [float(num) for num in nums]
            values.append(nums)
        return np.array(values)


pps_signal = load_cell_arr('cosgrove_phosphoprotein_signal')
pps = load_cell_arr('cosgrove_phosphoproteins')
signal_type = load_cell_arr('cosgrove_signal_type')
conditions = load_cell_arr('cosgrove_conditions')
cytokines = load_cell_arr('cosgrove_cytokines')
drugs = load_cell_arr('cosgrove_drugs')
signal_data = load_matrix('cosgrove_x')

ax1 = signal_data[:, 0]
ax1_label = pps_signal[0]
ax2 = signal_data[:, 1]
ax2_label = pps_signal[1]


def scatter_plot(x, y, xlabel, ylabel):
    plt.scatter(ax1, ax2, s=8 ** 2, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


plt.style.use('fivethirtyeight')
plt.figure(0)
plt.subplot(111, aspect='equal')
scatter_plot(ax1, ax2, ax1_label, ax2_label)

log_2_data = np.log2(signal_data + 0.001)
means = np.mean(log_2_data, 0)
stdevs = np.std(log_2_data, 0)
std_data = (log_2_data - means) / stdevs

ax1 = std_data[:, 1]
ax2 = std_data[:, 2]
plt.figure(1)
plt.subplot(111, aspect='equal')
scatter_plot(ax1, ax2, ax1_label, ax2_label)
plt.show()

# perform PCA
