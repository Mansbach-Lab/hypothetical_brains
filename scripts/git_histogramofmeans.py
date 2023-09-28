#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 12:27:26 2023

@author: lwright
"""
from __future__ import absolute_import, division, print_function
# import os.path as op
import hypotheticalbrains as hb
import numpy as np
import pyemma.plots as pyplot

directory = "/home/lwright/anaconda3/envs/networktoy/output/HypoBrains_Y2023_M09_D12_H13_M30_S53_v359938_r0.3/"
import_from = directory + 'means.csv'
stats = np.loadtxt(import_from, delimiter=',')
features = len(stats[0])

for i in range(features):
    print(i)
    hb.meanogram(stats, i, 1000, directory)
    
hb.free_energy_surface_allfeatures(stats, vmin=0, vmax=10, nbins=100, border = 1)
