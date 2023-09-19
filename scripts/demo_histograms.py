#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:52:57 2023

@author: lwright
"""



import numpy as np
# import os
# print("package version = ", vhb.__version__)
import matplotlib.pyplot as plt
import os.path as op
# how many voxels are there per dimension of the MRI?
import pyemma.plots as pyplot

def meanogram(stats, metric, bincount):
    means = stats[:,metric]
    voxelcount = len(means)
    plt.hist(means, bins=bincount)  # arguments are passed to np.histogram
    title_string = "Metric " + str(metric) + " histogram"
    plt.title(title_string)
    save_string = "histogram_vox"+ str(voxelcount) +"_metr"+str(metric)+".png"
    plt.savefig(save_string)
    plt.show()
    

import_from = '/home/lwright/anaconda3/envs/networktoy/output/HypoBrains_Y2023_M09_D12_H13_M30_S53_v359938_r0.3/means.csv'
stats = np.loadtxt(import_from, delimiter=',')

# meanogram(stats, 1, 100)


i = 0


counts = np.ones(stats[:,i].shape)
print(stats[:,i].shape)
print(counts.shape)
features = len(stats[0,:])


font = {'size'   : 6}

fig, axes = plt.subplots(features-1, features, figsize=(10, 4), sharex=True, sharey=True)
for i in range(features): # i = x axis variable
    a = 0 # a = y axis variable
    for j in range(features):
        if i==j:
            a = 1
        else:
            pyplot.plot_free_energy(stats[:,i], stats[:,j], ax=axes[j-a,i], nbins=100, vmin=0, vmax=10,  cbar=True)
            axes[j-a,i].set_xlabel(str(i))
            axes[j-a,i].set_ylabel(str(j))
plt.tight_layout()
plt.subplots_adjust(top=0.989,bottom=0.042,left=0.023,right=0.983,hspace=0.15,wspace=0.1)
plt.rc('font', **font)
plt.show()

