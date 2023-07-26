#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:29:57 2023

@author: lwright
"""


from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nb
import seaborn as sns
from sklearn.cluster import OPTICS, cluster_optics_dbscan

# import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import ward, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from scipy.stats import shapiro
import time
from scipy import stats
# import plotly.graph_objects as go

# Make features array in whole WM mask (voxels to be compared) in a test subject
feature_number = 7

# Load data of each feature within WM mask
features_dir = '/home/lwright/Desktop/TrialData/'

# how many voxels are there per dimension of the MRI?
samples = 20

# minimum edge weight to include
filter_threshold = 0.1

# for save file titles, update to today's date
stringname = 'brain_june27_'

# automated save file titles
save_matrix_as = join(stringname+str(samples)+'.npy')
save_features_as = join(stringname+'features'+str(samples)+'.csv')
save_graph_as = join(stringname+str(samples)+'.gexf')

# creating brain mask from mask file
WM_mask = nb.load(join(features_dir, 'WM_mask_'+str(samples)+'.nii')).get_fdata().astype('bool')

# importing attributes from files
AD = nb.load(join(features_dir, 'AD_'+str(samples)+'.nii')).get_fdata()
FA = nb.load(join(features_dir, 'FA_'+str(samples)+'.nii')).get_fdata()
MD = nb.load(join(features_dir, 'MD_'+str(samples)+'.nii')).get_fdata()
RD = nb.load(join(features_dir, 'RD_'+str(samples)+'.nii')).get_fdata()
ICVF = nb.load(join(features_dir, 'ICVF_'+str(samples)+'.nii')).get_fdata()
OD = nb.load(join(features_dir, 'OD_'+str(samples)+'.nii')).get_fdata()
ISOVF = nb.load(join(features_dir, 'ISOVF_'+str(samples)+'.nii')).get_fdata()

# masking attributes
AD_WM = AD[WM_mask]
FA_WM = FA[WM_mask]
MD_WM = MD[WM_mask]
RD_WM = RD[WM_mask]
ICVF_WM = ICVF[WM_mask]
OD_WM = OD[WM_mask]
ISOVF_WM = ISOVF[WM_mask]

# need to figure out how to add titles to matrices - for attributes csv
column_labels = ['AD_WM', 'FA_WM', 'MD_WM', 'RD_WM', 
                                          'ICVF_WM', 'OD_WM', 'ISOVF_WM']

# Make array containing all features (shape is N voxels in WM X feature_number)
voxel_count = AD_WM.shape[0]
feature_mat = np.zeros((voxel_count,feature_number))

# Replace zeros in each column by the data of one of the metrics 
feature_mat[:,0] = AD_WM
feature_mat[:,1] = FA_WM
feature_mat[:,2] = MD_WM
feature_mat[:,3] = RD_WM
feature_mat[:,4] = ICVF_WM
feature_mat[:,5] = OD_WM
feature_mat[:,6] = ISOVF_WM

print(feature_mat.shape)

# duplicate the matrix to record the scaled feature/attribute data
feature_mat_scaled = np.copy(feature_mat)

# timing the proceses to see how they scale
start = time.time()

# scale each feature set
for i in range(feature_number):
    scaler = MinMaxScaler()
    current_feature = np.reshape(feature_mat[:,i],(voxel_count,1))
    scaler.fit(current_feature)
    feature_mat_scaled[:,i] = scaler.transform(current_feature).flatten()
    
    # print distribution of each feature
    # sns.displot(feature_mat_scaled[:,i])
    title = str(i) + " - feature number"
    # plt.title(title)
print("time 1: ", (time.time() - start))

# saving attributes; labeling each column with sequential integer
labeled_matrix = np.zeros((voxel_count+1,feature_number))
labeled_matrix[0,:] = np.arange(0,7,1)
labeled_matrix[1::,:] = feature_mat_scaled
np.savetxt(save_features_as,labeled_matrix, delimiter=",")
print("Attributes saved")

#%%

# calculating squared distances between each node
distance_matrix_flat = pdist(feature_mat_scaled, 'sqeuclidean')
print("time 2: ", (time.time() - start))

# delete random samples from set for graphing efficacy's sake
# distance_matrix_flat_graphing = np.random.choice(distance_matrix_flat, size=30000000, replace=False)
distance_matrix_flat_graphing = distance_matrix_flat

# check if it returns indices ^^ otherwise there are other ways to do this
print(distance_matrix_flat.shape)

print("time 2.4: ", (time.time() - start))

# histogram of squared distances
sns.displot(distance_matrix_flat_graphing)
plt.title("Distogram")

print("time 2.6: ", (time.time() - start))

# checking out the effects of widths on weights
width = 1.0
weight_matrix_flat_graphing = np.exp(-1.*distance_matrix_flat_graphing/width)
weight_matrix_flat = np.exp(-1.*distance_matrix_flat/width)

# for i in [0.5, 1.0, 1.5, 3.0]:
#     weight_matrix_flat = np.exp(-1.*distance_matrix_flat/i)
#     title = str(i) + " - width"
#     plt.title(title)
print("time 3: ", (time.time() - start))

# pruning edges less than filter_threshold
myfunc = np.vectorize(lambda a : np.nan if (a < filter_threshold) else a)
filtered_weights = myfunc(weight_matrix_flat)
filtered_weights_graphing = myfunc(weight_matrix_flat_graphing)


# checking the stats of the weights
k2, p = stats.normaltest(filtered_weights)
print("time 4: ", (time.time() - start))

# histogram of filtered squared weights
filtered_weights_std = np.nanstd(filtered_weights)
filtered_weights_std_graphing = np.nanstd(filtered_weights_graphing)

sns.displot(filtered_weights_graphing)
title = str(filtered_weights_std_graphing) + " - SD, filtered weights"
plt.title(title)
print("time 5: ", (time.time() - start))






# Z = ward(distance_matrix_flat)
# Z2 = ward(1/weight_matrix_flat)

# plt.figure()
# dendrogram(Z, p=10, truncate_mode='level')
# plt.title("dendrogram distances")

# plt.figure()
# dendrogram(Z2, p=10, truncate_mode='level')
# plt.title("dendrogram weights inverse")



# print(shapiro(filtered_weights[0:100000]))

# # mask_filtering = weight_matrix_flat > filter_threshold

# myfunc = np.vectorize(lambda a : 0.0 if (a < filter_threshold) else a)
# filtered_weights = myfunc(weight_matrix_flat)

# print("time 4: ", (time.time() - start))
