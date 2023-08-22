#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 14:09:53 2023

@author: lwright
"""
import numpy as np
# import pandas as pd
# import scipy.optimize as opt
# from scipy.special import erf

from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial import cKDTree
import numpy.testing as npt

from memory_profiler import profile, memory_usage
import requests



def thresholding_weight(weight_matrix_flat, weight_threshold):
    # myfunc = np.vectorize(lambda a : 0.0 if (a < filter_threshold) else a)

    for i in range(len(weight_matrix_flat)):
        if weight_matrix_flat[i] < weight_threshold:
            weight_matrix_flat[i] = 0
    return weight_matrix_flat

# def thresholding_distance(distance_matrix_flat, distance_threshold):
#     for i in range(len(distance_matrix_flat)):
#         if distance_matrix_flat[i] > distance_threshold:
#             distance_matrix_flat[i] = 1
#     return distance_matrix_flat

def ckd_made_weights(feature_mat_scaled, width, distance_threshold, weight_threshold):
    # method 1 for sparse matrix; does distance only, cannot make weights
    kd_tree1 = cKDTree(feature_mat_scaled)
    kd_tree2 = cKDTree(feature_mat_scaled)
    ckdtree_distance = kd_tree1.sparse_distance_matrix(kd_tree2, distance_threshold).tocsr()
    ckdtree_weight = ckdtree_distance.copy()
    ckdtree_weight.data = thresholding_weight(np.exp(-1.*(ckdtree_distance.power(2)).data/width), weight_threshold)
    ckdtree_weight.eliminate_zeros()
    return ckdtree_weight

def ckd_made_distance(feature_mat_scaled, width, distance_threshold):
    # method 1 for sparse matrix; does distance only, cannot make weights
    kd_tree1 = cKDTree(feature_mat_scaled)
    kd_tree2 = cKDTree(feature_mat_scaled)
    ckdtree_distance = kd_tree1.sparse_distance_matrix(kd_tree2, distance_threshold).tocsr()
    ckdtree_distance = ckdtree_distance.power(2)
    ckdtree_distance.eliminate_zeros()
    return ckdtree_distance

    
def loop_made_weights(feature_mat_scaled, width, weight_threshold):
    distance_matrix_flat = pdist(feature_mat_scaled, 'sqeuclidean')
    weight_matrix_flat = np.exp(-1.*distance_matrix_flat/width)
    
    # pruning edges less than filter_threshold
    filtered_weights = thresholding_weight(weight_matrix_flat, weight_threshold)

    # collecting data
    voxel_number=feature_mat_scaled.shape[0]
    weights_sparse_lil = lil_matrix((voxel_number, voxel_number)) 
    
    count = 0
    for row in range(voxel_number-1):
        
        weights_sparse_lil[row, row] = 1.
        
        for col in range(row+1, voxel_number): 
            
            # for upper triangle elements
            weights_sparse_lil[row, col] = (
                filtered_weights[count])
            
            # to add lower triangular elements
            weights_sparse_lil[col,row] = (
                filtered_weights[count])

            count += 1
    weights_sparse_lil[feature_mat_scaled.shape[0]-1,feature_mat_scaled.shape[0]-1] = 1.0
    return weights_sparse_lil.tocsr() 
    
def loop_made_distance(feature_mat_scaled, width):
    distance_matrix_flat = pdist(feature_mat_scaled, 'sqeuclidean')

    # collecting data; two different approaches lil_matrix or for direct to CSR
    voxel_number=feature_mat_scaled.shape[0]
    distances_sparse_lil = lil_matrix((voxel_number, voxel_number)) 

    count = 0
    for row in range(voxel_number-1):
        for col in range(row+1, voxel_number): 
            
            # for upper triangle elements
            distances_sparse_lil[row, col] = (
                distance_matrix_flat[count])
            
            # to add lower triangular elements
            distances_sparse_lil[col,row] = (
                distance_matrix_flat[count])
            
            count += 1
     
    return distances_sparse_lil.tocsr()

# method 4 this bit takes too much memory for higher orders
def squareform_made_weights(feature_mat_scaled, width, weight_threshold):
    weight_matrix_square = squareform(
        thresholding_weight(np.exp(-1.*pdist(feature_mat_scaled, 'sqeuclidean')/width),
    weight_threshold))
    sq_csr = csr_matrix(weight_matrix_square)
    sq_csr.setdiag(1.0)
    return sq_csr

def squareform_made_distance(feature_mat_scaled):
    weight_matrix_square = squareform(pdist(feature_mat_scaled, 'sqeuclidean'))
    return csr_matrix(weight_matrix_square)

"""def ckd_made_distance(feature_mat_scaled, width, distance_threshold):
    # method 1 for sparse matrix; does distance only, cannot make weights
    kd_tree = cKDTree(feature_mat_scaled)
    ckdtree_distance = kd_tree.sparse_distance_matrix(kd_tree, distance_threshold).tocsr()
    return ckdtree_distance.power(2)

def squareform_made_distance(feature_mat_scaled):
    weight_matrix_square = squareform(pdist(feature_mat_scaled, 'sqeuclidean'))
    return csr_matrix(weight_matrix_square)

def thresholding_weight(weight_matrix_flat, weight_threshold):
    # myfunc = np.vectorize(lambda a : 0.0 if (a < filter_threshold) else a)

    for i in range(len(weight_matrix_flat)):
        if weight_matrix_flat[i] < weight_threshold:
            weight_matrix_flat[i] = 0
    return weight_matrix_flat

def ckd_made_weights(feature_mat_scaled, width, distance_threshold, weight_threshold):
    # method 1 for sparse matrix; does distance only, cannot make weights
    kd_tree1 = cKDTree(feature_mat_scaled)
    kd_tree2 = cKDTree(feature_mat_scaled)
    ckdtree_distance = kd_tree1.sparse_distance_matrix(kd_tree2, distance_threshold).tocsr()
    ckdtree_weight = ckdtree_distance.copy()
    ckdtree_weight.data = thresholding_weight(np.exp(-1.*(ckdtree_distance.power(2)).data/width), weight_threshold)
    return ckdtree_weight

def loop_made_weights(feature_mat_scaled, width, weight_threshold):
    distance_matrix_flat = pdist(feature_mat_scaled, 'sqeuclidean')
    weight_matrix_flat = np.exp(-1.*distance_matrix_flat/width)
    
    # pruning edges less than filter_threshold
    filtered_weights = thresholding_weight(weight_matrix_flat, weight_threshold)

    # collecting data
    voxel_number=feature_mat_scaled.shape[0]
    weights_sparse_lil = lil_matrix((voxel_number, voxel_number)) 
    
    count = 0
    for row in range(voxel_number-1):
        for col in range(row+1, voxel_number): 
            
            # for upper triangle elements
            weights_sparse_lil[row, col] = (
                filtered_weights[count])
            
            # to add lower triangular elements
            weights_sparse_lil[col,row] = (
                filtered_weights[count])

            count += 1

    return weights_sparse_lil.tocsr() 

def squareform_made_weights(feature_mat_scaled, width, weight_threshold):
    weight_matrix_square = squareform(
        thresholding_weight(np.exp(-1.*pdist(feature_mat_scaled, 'sqeuclidean')/width),
    weight_threshold))
    sq_csr = csr_matrix(weight_matrix_square)
    sq_csr.setdiag(1.0)
    return sq_csr"""

mat1 = np.array([0,1,2,3,4,5]).reshape((6, 1))
# mat2 = np.array([0,0,0,2,2,2]).reshape((6, 1))

tree1 = cKDTree(mat1)
tree2 = cKDTree(mat1)

distance_threshold = 1000
weight_threshold = 0.3
width = 1.0

# square dist, weights
sq_dist = squareform_made_distance(mat1)
sq_weight = squareform_made_weights(mat1, width, weight_threshold)
print(sq_dist.nnz)
print(sq_weight.nnz)

# loop dist, weights
loop_dist = loop_made_distance(mat1, width)
loop_weights = loop_made_weights(mat1, width, weight_threshold)
print(loop_dist.nnz)
print(loop_weights.nnz)
# print(csr_matrix.todense(loop_made_weights(mat1, width, weight_threshold)))

# ckdTree dist, weights

ckd_dist = ckd_made_distance(mat1, width, distance_threshold)
ckd_weight = ckd_made_weights(mat1, width, distance_threshold, weight_threshold)
print(ckd_dist.nnz)
print(ckd_weight.nnz)

# print(sq_dist.todense())
# print(loop_dist.todense())
# print(ckd_dist.todense())

print(sq_weight.todense())
print(loop_weights.todense())
print(ckd_weight.todense())

mem = max(memory_usage(proc=hb.loop_made_weights(mat1, width, weight_threshold)))
print("Maximum memory used: {} MiB".format(mem))

mem = max(memory_usage(proc=hb.loop_made_distance(mat1, width)))
print("Maximum memory used: {} MiB".format(mem))