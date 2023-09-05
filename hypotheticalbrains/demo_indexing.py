#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 12:29:14 2023

@author: lwright
"""

import numpy as np

features = 3

feature_mat_scaled = np.random.randint(50, size=(10,features))
neighbour_idx = np.array([3,5,7,8])


# total number of neighbours in ith cluster
neighbour_count = len(neighbour_idx)

# collect the cluster (voxels, neighbour_idx+features)
cluster = np.zeros((neighbour_count,features+1))
cluster[:,0] = neighbour_idx

cluster[:,1::] = feature_mat_scaled[neighbour_idx,:]
