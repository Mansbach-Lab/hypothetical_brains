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

def meanogram(stats, metric):
    means = stats[:,metric]
    voxelcount = len(means)
    bincount = int(voxelcount/100)
    print(bincount)
    plt.hist(means, bins=bincount)  # arguments are passed to np.histogram
    title_string = "Metric " + str(metric) + " histogram"
    plt.title(title_string)
    save_string = "histogram_vox"+ str(voxelcount) +"_metr"+str(metric)+".png"
    plt.savefig(save_string)
    plt.show()


import_from = '/home/lwright/anaconda3/envs/networktoy/output/HypoBrains_Y2023_M09_D12_H12_M15_S54/means.csv'
stats = np.loadtxt(import_from, delimiter=',')

meanogram(stats, 1, 100)



