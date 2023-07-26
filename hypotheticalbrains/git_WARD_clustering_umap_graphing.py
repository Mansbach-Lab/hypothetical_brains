#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:55:20 2023

@author: lwright
"""

# from sklearn.cluster import OPTICS, cluster_optics_dbscan
# from sklearn import manifold, datasets
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
from sklearn.cluster import AgglomerativeClustering
from os.path import join
import numpy as np
import umap
from time import time

start_time = time()

# how many voxels are there per dimension of the MRI?
samples = 100

# set number of data points per cluster
cluster_members = 200


# for importing features file
stringname = 'brain_june27_'
import_data_from = join('/home/lwright/anaconda3/envs/networktoy/' 
                        + stringname + 'features' + str(samples)+'.csv')
data = np.loadtxt(import_data_from, delimiter=',')

# data = np.loadtxt('/home/lwright/anaconda3/envs/networktoy/brain_june27_features65.csv', delimiter=',')

# automated save file titles
save_fig1_as = join(stringname+str(samples)+'_1.png')
save_fig2_as = join(stringname+str(samples)+'_2.png')

# data = np.loadtxt('/home/lwright/brain_june27_features65.csv', delimiter=',')
data = data[1::,:]
voxel_number = np.arange(0, len(data),1)
features = len(data[0])

cluster_count = int(len(voxel_number)/cluster_members)

reducer = umap.UMAP()
embedding = reducer.fit_transform(data)
embedding.shape


## Plotting different clustering linikages on UMAP coordinates

rows,cols = 2,2
linkages = [["ward", "average"], ["complete", "single"]]
global cluster_ward, cluster_average, cluster_complete, cluster_single
cluster_list = ["cluster_ward", "cluster_average", "cluster_complete", 
                "cluster_single"]
plt.figure(figsize=(24.0, 12.0))
for i in range(rows):
    for j in range(cols):
        linkage = linkages[i][j]
        graph_number = i*cols+j
        plt.subplot(2,2,graph_number+1)
        cluster_list[graph_number] = AgglomerativeClustering(linkage=linkage, 
                                                    n_clusters=cluster_count)
        t0 = time()
        cluster_list[graph_number].fit(data)
        print("%s :\t%.2fs" % (linkage, time() - t0))
        plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_list[graph_number].labels_,
                            marker='.', cmap='tab20')
        plt.title("%s linkage" % linkage, size=17)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
plt.subplots_adjust(top=0.96,bottom=0.033,left=0.021,right=0.988,hspace=0.148,
                    wspace=0.043)
plt.savefig(save_fig1_as)
# plt.show()


## Plotting one clustering linkage across pairs of features

link = 0 # 0=ward, 1=average, 2=complete, 3=single

font = {'size'   : 2}
plt.figure(figsize=(24.0, 12.0))

for i in range(features): # i = x axis variable
    a = 0 # a = y axis variable
    for j in range(features):
        if i==j:
            a = 1
        else:
            ax=plt.subplot2grid((features-1,features), (j-a,i))
            ax.scatter(data[:, i], data[:, j], c=cluster_list[link].labels_,
                            marker='.', cmap='tab20')
            plt.xlabel(i, fontsize=10)
            plt.ylabel(j, fontsize=10)            
plt.subplots_adjust(top=0.997,bottom=0.048,left=0.03,right=0.995,
                    hspace=0.402,wspace=0.289)
plt.rc('font', **font)
plt.savefig(save_fig2_as)
# plt.show()
run_time = time() - start_time
print("Done! Total run time :\t%.2fs" % (run_time))