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
import pyemma.plots as pyemmaplots
# import matplotlib.rcParams as pltrc

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

# import_from = '/home/lwright/anaconda3/envs/networktoy/output/HypoBrains_Y2023_M09_D12_H12_M15_S54_v1073_r0.5/means.csv'
stats = np.loadtxt(import_from, delimiter=',')

# meanogram(stats, 1, 100)



features = len(stats[0,:])
voxelcount = len(stats[:,0])


vmin=0
vmax=6
nbins=100
border = 1

font = {'size'   : 6}
fig1,ax1 = plt.subplots(nrows=1, ncols=1)
im = ax1.imshow(np.array([[0.1,vmax],[vmax/4.,0.1]]), vmin=vmin, vmax=vmax, cmap='nipy_spectral')
cbar_ax = fig1.add_axes([0.85, 0.15, 0.05, 0.7])
fig1.colorbar(im, cax=cbar_ax)

fig, axes = plt.subplots(features-1, features, figsize=(24.0, 12.0), sharex=True, sharey=True)
for i in range(features): # i = x axis variable
    a = 0 # a = y axis variable
    for j in range(features):
        if i==j:
            a = 1
        else:
            pyemmaplots.plot_free_energy(stats[:,i], stats[:,j], ax=axes[j-a,i], 
                                    nbins=nbins, vmin=vmin, vmax=vmax, cbar=False)
                                    # extend='both')
            axes[j-a,i].set_xlabel(str(i))
            axes[j-a,i].set_ylabel(str(j))
            # spanx = max(stats[:,i])-min(stats[:,i])
            # spany = max(stats[:,j])-min(stats[:,j])
            axes[j-a,i].set_xlim(min(stats[:,i])-border,#spanx/nbins,
                                 max(stats[:,i])+border)#spanx/nbins)
            axes[j-a,i].set_ylim(min(stats[:,j])-border,#spany/nbins,
                                 max(stats[:,j])+border)#spany/nbins)

# set xticks, xtick labels - missing


fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.92, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.rc('font', **font)

# plt.tight_layout()
plt.subplots_adjust(top=0.96,bottom=0.042,left=0.024,
                    right=0.91,hspace=0.169,wspace=0.096)
title_string = import_from + " nbins=" + str(nbins) + " vmin=" + str(vmin) + " vmax=" + str(vmax)
fig.suptitle(title_string)    
save_string = import_from +"FES_vox"+ str(voxelcount) + "_nbins" + str(nbins) + "_vmin" + str(vmin) + "_vmax" + str(vmax) + ".png"
plt.savefig(save_string)
plt.show()

# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.92, 0.15, 0.05, 0.7])
# fig.colorbar(im, cax=cbar_ax)
# plt.tight_layout()
# plt.subplots_adjust(top=0.991,bottom=0.042,left=0.024,right=0.91,
#                     hspace=0.169,wspace=0.096)
# plt.rc('font', **font)
# plt.show()

