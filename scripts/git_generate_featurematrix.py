#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 13:25:01 2023

@author: lwright
"""

import hypotheticalbrains as hb

# how many voxels are there per dimension of the MRI?
samples = 20 # whole brain = 256

# how many features per voxel?
feature_number = 7

# for importing and preparing features data
stringname = 'brain_july26_features'

# Load data of each feature within WM mask
features_dir = '/home/lwright/Desktop/TrialData/'

hb.generate_feature_matrix(features_dir, stringname, samples, feature_number)