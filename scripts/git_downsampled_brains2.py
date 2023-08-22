#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:29:57 2023

@author: lwright
"""



from os.path import join
import numpy as np
import nibabel as nb
import networkx as nx
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix, csr_array, lil_matrix
from scipy.spatial import cKDTree
# from sklearn.metrics import pairwise_distances
import time

# Make features array in whole WM mask (voxels to be compared) in a test subject
feature_number = 7
samples = 256
filter_threshold = 0.3

# Load data of each feature within WM mask
features_dir = '/home/lwright/Desktop/TrialData/'

stringname = 'brain_aug17_'

save_matrix_as = join(stringname+str(samples)+'.npy')
save_graph_as = join(stringname+str(samples)+'.gexf')
save_features_as = join(stringname+'features'+str(samples)+'.csv')
# WM_mask = nb.load(join(features_dir, 'WM_mask_'+str(samples)+'.nii')).get_fdata().astype('bool')
# AD = nb.load(join(features_dir, 'AD_'+str(samples)+'.nii')).get_fdata()
# FA = nb.load(join(features_dir, 'FA_'+str(samples)+'.nii')).get_fdata()
# MD = nb.load(join(features_dir, 'MD_'+str(samples)+'.nii')).get_fdata()
# RD = nb.load(join(features_dir, 'RD_'+str(samples)+'.nii')).get_fdata()
# ICVF = nb.load(join(features_dir, 'ICVF_'+str(samples)+'.nii')).get_fdata()
# OD = nb.load(join(features_dir, 'OD_'+str(samples)+'.nii')).get_fdata()
# ISOVF = nb.load(join(features_dir, 'ISOVF_'+str(samples)+'.nii')).get_fdata()


AD = nb.load(join(features_dir,'sub-071_P_AD_WarpedToMNI.nii')).get_fdata()
FA = nb.load(join(features_dir,'sub-071_P_FA_WarpedToMNI.nii')).get_fdata()
MD = nb.load(join(features_dir,'sub-071_P_MD_WarpedToMNI.nii')).get_fdata()
RD = nb.load(join(features_dir,'sub-071_P_RD_WarpedToMNI.nii')).get_fdata()
ICVF = nb.load(join(features_dir,'sub-071_P_ICVF_WarpedToMNI.nii')).get_fdata()
OD = nb.load(join(features_dir,'sub-071_P_OD_WarpedToMNI.nii')).get_fdata()
ISOVF = nb.load(join(features_dir,'sub-071_P_ISOVF_WarpedToMNI.nii')).get_fdata()
WM_mask = nb.load(join(features_dir,'Group_mean_CIRM_57_ACTION_5_MPRAGE0p9_T1w_brain_reg2DWI_0p9_T1_5tt_vol2_WM_WarpedToMNI_thr0p95_bin.nii')).get_fdata().astype('bool')




AD_WM = AD[WM_mask]
FA_WM = FA[WM_mask]
MD_WM = MD[WM_mask]
RD_WM = RD[WM_mask]
ICVF_WM = ICVF[WM_mask]
OD_WM = OD[WM_mask]
ISOVF_WM = ISOVF[WM_mask]

voxel_number = AD_WM.shape[0]

# Make array containing all features (voxels in WM X features)
feature_mat = np.zeros((voxel_number,feature_number))

# Replace zeros in each column by the data of one of the metrics 
feature_mat[:,0] = AD_WM
feature_mat[:,1] = FA_WM
feature_mat[:,2] = MD_WM
feature_mat[:,3] = RD_WM
feature_mat[:,4] = ICVF_WM
feature_mat[:,5] = OD_WM
feature_mat[:,6] = ISOVF_WM

start = time.time()

# scaling the MRI feature data
scaler = StandardScaler()
scaler.fit(feature_mat)
feature_mat_scaled = scaler.transform(feature_mat)

print("time 1: ", (time.time() - start))

labeled_matrix = np.zeros((AD_WM.shape[0]+1,feature_number))
labeled_matrix[0,:] = np.arange(0,feature_number,1)
labeled_matrix[1::,:] = feature_mat_scaled

np.savetxt(save_features_as,labeled_matrix, delimiter=",")

print("Attributes saved")


width = 1.0
distance_threshold = 1000 # for ckdtree only


"""
kd_tree1 = cKDTree(feature_mat_scaled)
kd_tree2 = cKDTree(feature_mat_scaled)
ckdtree_distance = kd_tree1.sparse_distance_matrix(kd_tree2, distance_threshold).tocsr()
max_dist = 0
for i in range(feature_number):
    curr_dist = ckdtree_distance.max()
    if curr_dist > max_dist:
        max_dist = curr_dist
print(max_dist)

def ckd_made_matrix(feature_mat_scaled, width, distance_threshold):
    # method 1 for sparse matrix; does distance only, cannot make weights
    kd_tree1 = cKDTree(feature_mat_scaled)
    kd_tree2 = cKDTree(feature_mat_scaled)
    ckdtree_distance = kd_tree1.sparse_distance_matrix(kd_tree2, distance_threshold).tocsr()
    ckdtree_weight = ckdtree_distance.copy()
    ckdtree_weight.data = np.exp(-1.*myfunc(ckdtree_distance.data)/width)
    return ckdtree_weight

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
# or cdist?
# can consolidate steps into Y = pdist(X, f) where f is user defined function.
# Computes the distance between all pairs of vectors in X using the user supplied 2-arity function f. 

def loop_made_matrix(feature_mat_scaled, width):
    distance_matrix_flat = pdist(feature_mat_scaled, 'sqeuclidean')
    weight_matrix_flat = np.exp(-1.*distance_matrix_flat/width)
    
    # pruning edges less than filter_threshold
    filtered_weights = myfunc(weight_matrix_flat)

    # collecting data; two different approaches lil_matrix or for direct to CSR
    weights_sparse_lil = lil_matrix((voxel_number, voxel_number)) 
    # row_idx, col_idx = [],[]
    
    count = 0
    for row in range(voxel_number-1):
        for col in range(row+1, voxel_number):
            # row_idx.append(row)
            # col_idx.append(col)
            weights_sparse_lil[row, col] = filtered_weights[count]
            count += 1
    print("exit loop")
    
    # method 2 for making sparse array of weights - think this is best
    weights_sparse_csr = weights_sparse_lil.tocsr()
    # method 3 for making sparse array of weights - seems memory-heavy
    # rows = np.array(row_idx)
    # columns = np.array(col_idx)
    # distances_sparse = csr_array((filtered_weights, (row_idx, col_idx)), shape=(voxel_number, voxel_number)) 
    return weights_sparse_csr

# method 4 this bit takes too much memory for higher orders
def squareform_made_matrix(feature_mat_scaled, width):
    distance_matrix_square = squareform(myfunc(np.exp(-1.*pdist(feature_mat_scaled, 'sqeuclidean')/width)))
    return csr_matrix(distance_matrix_square)
"""

"""np.save(save_matrix_as,distance_matrix_square)
brain = nx.from_numpy_matrix(distance_matrix_square)
nx.write_gexf(brain, save_graph_as)

print("done")"""
