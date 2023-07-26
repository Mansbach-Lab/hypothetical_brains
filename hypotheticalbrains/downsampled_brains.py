#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:29:57 2023

@author: lwright
"""

#import sys
#sys.path.append('../')
#import c2m2_new as ccm
#import mvcomp
# import os
# import glob
from os.path import join
# from os.path import basename
# import matplotlib.pyplot as plt
import numpy as np
import nibabel as nb
# import math
# import importlib
# import matplotlib.patches as patches

#import seaborn as sns
import pandas as pd
# from matplotlib.colors import LogNorm, Normalize
# import subprocess as subproc
import networkx as nx
import graph_tool.all as gt
# import cupy as cp
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import time

# Make features array in whole WM mask (voxels to be compared) in a test subject
feature_number = 7

# Load data of each feature within WM mask
features_dir = '/home/lwright/Desktop/TrialData/'

samples = 20
filter_threshold = 0.3
stringname = 'brain_june07_'


save_matrix_as = join('brain_may17_'+str(samples)+'.npy')
save_graph_as = join('brain_may17_'+str(samples)+'.gexf')
save_features_as = join(stringname+'features'+str(samples)+'.csv')


WM_mask = nb.load(join(features_dir, 'WM_mask_'+str(samples)+'.nii')).get_fdata().astype('bool')

AD = nb.load(join(features_dir, 'AD_'+str(samples)+'.nii')).get_fdata()
FA = nb.load(join(features_dir, 'FA_'+str(samples)+'.nii')).get_fdata()
MD = nb.load(join(features_dir, 'MD_'+str(samples)+'.nii')).get_fdata()
RD = nb.load(join(features_dir, 'RD_'+str(samples)+'.nii')).get_fdata()
ICVF = nb.load(join(features_dir, 'ICVF_'+str(samples)+'.nii')).get_fdata()
OD = nb.load(join(features_dir, 'OD_'+str(samples)+'.nii')).get_fdata()
ISOVF = nb.load(join(features_dir, 'ISOVF_'+str(samples)+'.nii')).get_fdata()

AD_WM = AD[WM_mask]
FA_WM = FA[WM_mask]
MD_WM = MD[WM_mask]
RD_WM = RD[WM_mask]
ICVF_WM = ICVF[WM_mask]
OD_WM = OD[WM_mask]
ISOVF_WM = ISOVF[WM_mask]

column_labels = ['AD_WM', 'FA_WM', 'MD_WM', 'RD_WM', 
                                          'ICVF_WM', 'OD_WM', 'ISOVF_WM']

# Make array containing all features (voxels in WM X 10 features)
feature_mat = np.zeros((AD_WM.shape[0],feature_number))

# Replace zeros in each column by the data of one of the metrics 
feature_mat[:,0] = AD_WM
feature_mat[:,1] = FA_WM
feature_mat[:,2] = MD_WM
feature_mat[:,3] = RD_WM
feature_mat[:,4] = ICVF_WM
feature_mat[:,5] = OD_WM
feature_mat[:,6] = ISOVF_WM




df = pd.DataFrame(feature_mat, columns = column_labels)
print(feature_mat.shape)
voxels_howmany = len(df.iloc[:,0])


# AD_WM_df = pd.DataFrame(AD_WM_row)
# AD_WM_corrmat = AD_WM_df.corr()

# brain = nx.Graph()
    
# for i in range(len(feature_mat)):
#     brain.add_node(i, AD_WM=feature_mat[i,0], FA_WM=feature_mat[i,1], MD_WM=feature_mat[i,2], RD_WM=feature_mat[i,3],
#                 ICVF_WM=feature_mat[i,4], OD_WM=feature_mat[i,5], ISOVF_WM=feature_mat[i,6])
    
start = time.time()
scaler = StandardScaler()
scaler.fit(feature_mat)
feature_mat_scaled = scaler.transform(feature_mat)


print("time 1: ", (time.time() - start))

labeled_matrix = np.zeros((AD_WM.shape[0]+1,feature_number))
labeled_matrix[0,:] = np.arange(0,7,1)
labeled_matrix[1::,:] = feature_mat_scaled

np.savetxt(save_features_as,labeled_matrix, delimiter=",")

# np.savetxt("AD_WM.csv",labeled_matrix[:,0], delimiter=",")
# np.savetxt("FA_WM.csv",labeled_matrix[:,1], delimiter=",")
# np.savetxt("MD_WM.csv",labeled_matrix[:,2], delimiter=",")
# np.savetxt("RD_WM.csv",labeled_matrix[:,3], delimiter=",")
# np.savetxt("ICVF_WM.csv",labeled_matrix[:,4], delimiter=",")
# np.savetxt("OD_WM.csv",labeled_matrix[:,5], delimiter=",")
# np.savetxt("ISOVF_WM.csv",labeled_matrix[:,6], delimiter=",")
print("Attributes saved")

distance_matrix_flat = pdist(feature_mat_scaled)
print("time 2: ", (time.time() - start))

weight_matrix_flat = np.exp(-1.*distance_matrix_flat)
print("time 3: ", (time.time() - start))

# mask_filtering = weight_matrix_flat > filter_threshold

myfunc = np.vectorize(lambda a : 0.0 if (a < filter_threshold) else a)
filtered_weights = myfunc(weight_matrix_flat)
# filtered_weights = weight_matrix_flat[mask_filtering]

distance_matrix_square = squareform(filtered_weights)

# zeroed_weights = np.ma.MaskedArray(weight_matrix_flat, mask_filtering, fill_value = np.nan)
# zeroed_weights.filled()

# distance_matrix_square = squareform(zeroed_weights)

print("time 4: ", (time.time() - start))

np.save(save_matrix_as,distance_matrix_square)

brain2 = nx.from_numpy_matrix(distance_matrix_square)

nx.write_gexf(brain2, save_graph_as)
print("time 5: ", (time.time() - start))


"""

print("Nodes: ", len(brain.nodes())) 
print("Edges: ", len(brain.edges())) 
    
brain2 = nx.from_numpy_matrix(distance_matrix_square)

nx.write_gexf(brain2, "brain2.gexf")
print("time 5: ", (time.time() - start))


# maybe_matrix = nx.adjacency_matrix(brain)
# brain2 = nx.from_numpy_matrix(maybe_matrix)

print("Nodes: ", len(brain2.nodes())) 
print("Edges: ", len(brain2.edges())) 

# brain3 = nx.complete_graph(brain)

# print("Nodes: ", len(brain3.nodes())) 
# print("Edges: ", len(brain3.edges())) 

# gtbrain = gt.Graph(directed=False)

# nx.write_gexf(brain, "brain.gexf")

need to make parallel graphs, make correlation matricies, sum them to make edges

# for i in range(voxels_howmany):
#     for j in np.arange(i,voxels_howmany+1,1):
#         name = 'corr_matrix' + str(i)
#         correlation_matrix = df.iloc[i,1:].corr()

# brain2 = nx.from_numpy_matrix(adjacency_matrix)


print("Nodes: ", len(brain.nodes())) 

nx.write_gexf(brain, "brain.gexf")

"""