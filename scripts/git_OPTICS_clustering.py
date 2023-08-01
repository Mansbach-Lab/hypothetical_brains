#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:10:17 2023

@author: lwright
"""

from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import umap
# import itertools

data = np.loadtxt('/home/lwright/brain_june14_features20.csv', delimiter=',')
data = data[1::,:]
labels = np.arange(0, len(data),1)
reducer = umap.UMAP()

features = len(data[0])

embedding = reducer.fit_transform(data)
embedding.shape

eps1, eps2 = 0.5, 1.05
minsamples=50
XI=0.02 # 0.015
minclustersize=None

setmaxcluster = 12

clust = OPTICS(min_samples=minsamples, xi=XI, min_cluster_size=minclustersize)
clust.fit(data)


space = np.arange(len(embedding))

labels_1 = cluster_optics_dbscan(
    reachability=clust.reachability_,
    core_distances=clust.core_distances_,
    ordering=clust.ordering_,
    eps=eps1,
)
labels_2 = cluster_optics_dbscan(
    reachability=clust.reachability_,
    core_distances=clust.core_distances_,
    ordering=clust.ordering_,
    eps=eps2,
)

reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]

plt.figure(figsize=(10, 7))
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :])
ax2 = plt.subplot(G[1, 0])
ax3 = plt.subplot(G[1, 1])
ax4 = plt.subplot(G[1, 2])

# Reachability plot
colors = ['xkcd:hot pink', 'xkcd:violet', 'xkcd:teal', 
          'xkcd:orange', 'xkcd:true blue', 'xkcd:bright red',
          'xkcd:apple green', 'xkcd:golden yellow', 'xkcd:barney purple']



for klass, color in zip(range(0, setmaxcluster), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    ax1.plot(Xk, Rk, color, alpha=0.3)
ax1.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)
ax1.plot(space, np.full_like(space, 2.0, dtype=float), "k--", alpha=0.5)
ax1.plot(space, np.full_like(space, 1.0, dtype=float), "k-", alpha=0.5)
ax1.plot(space, np.full_like(space, 0.5, dtype=float), "k-.", alpha=0.5)
ax1.set_ylabel("Reachability (epsilon distance)")
ax1.set_title("Reachability Plot: min_samples="+str(minsamples)+
              "; xi=" +str(XI)+"; min_cluster_size="+str(minclustersize))
plt.subplots_adjust(top=0.964,bottom=0.04,left=0.035,right=0.988,
                    hspace=0.186,wspace=0.086)


# OPTICS

ax2.scatter(embedding[np.invert(clust.labels_==-1),0],
            embedding[np.invert(clust.labels_==-1),1],
            c=clust.labels_[np.invert(clust.labels_==-1)],
            cmap='tab20')
ax2.plot(embedding[clust.labels_ == -1, 0], embedding[clust.labels_ == -1, 1], "k+", alpha=0.1)
ax2.set_title("Automatic Clustering\nOPTICS")
"""
for klass, color in zip(range(0, setmaxcluster), colors):
    Xk = embedding[clust.labels_ == klass]
    ax2.plot(Xk[:, 0], Xk[:, 1], color, marker='.', linestyle='None', alpha=0.3)
ax2.plot(embedding[clust.labels_ == -1, 0], embedding[clust.labels_ == -1, 1], "k+", alpha=0.1)
ax2.set_title("Automatic Clustering\nOPTICS")
"""
# DBSCAN at 0.5
for klass, color in zip(range(0, setmaxcluster), colors):
    Xk = embedding[labels_1 == klass]
    ax3.plot(Xk[:, 0], Xk[:, 1], color, marker='.', linestyle='None', alpha=0.3)
ax3.plot(embedding[labels_1 == -1, 0], embedding[labels_1 == -1, 1], "k+", alpha=0.1)
ax3.set_title("Clustering at "+ str(eps1) +" epsilon cut\nDBSCAN")

# DBSCAN at 2.
for klass, color in zip(range(0, setmaxcluster), colors):
    Xk = embedding[labels_2 == klass]
    ax4.plot(Xk[:, 0], Xk[:, 1], color, marker='.',linestyle='None', alpha=0.3)
ax4.plot(embedding[labels_2 == -1, 0], embedding[labels_2 == -1, 1], "k+", alpha=0.1)
ax4.set_title("Clustering at "+ str(eps2)+ " epsilon cut\nDBSCAN")

plt.tight_layout()
plt.show()



font = {'size'   : 2}

plt.figure(2)
for i in range(features): # i = x axis variable
    a = 0 # a = y axis variable
    for j in range(features):
        if i==j:
            a = 1
        else:
            plt.subplot2grid((features-1,features), (j-a,i))
            for klass, color in zip(range(0, 9), colors):
                Xk = data[labels_1 == klass]
                plt.plot(Xk[:, i], Xk[:, j], color, marker='.', linestyle='None', alpha=0.3)
            plt.plot(data[:,0][labels_1 == -1], data[:,1][labels_1 == -1], "k+", alpha=0.1)
            plt.xlabel(i, fontsize=10)
            plt.ylabel(j, fontsize=10)            
plt.tight_layout()
plt.subplots_adjust(top=0.997,bottom=0.048,left=0.03,right=0.995,
                    hspace=0.402,wspace=0.289)
plt.rc('font', **font)
plt.show()

# https://stackoverflow.com/questions/37424530/how-to-make-more-than-10-subplots-in-a-figure

# axes = list(itertools.combinations(range(features), 2))
# print(len(axes))
# 
# for pairs in range(len(axes)):
#     x_axis, y_axis = axes[pairs][0], axes[pairs][1]
#     plt.figure()
#     for klass, color in zip(range(0, 9), colors):
#         Xk = data[labels_050 == klass]
#         plt.plot(Xk[:, x_axis], Xk[:, y_axis], color, marker='.', linestyle='None', alpha=0.3)
#     plt.plot(data[:,0][labels_200 == -1], data[:,1][labels_200 == -1], "k+", alpha=0.1)
#     plt.xlabel(x_axis)
#     plt.ylabel(y_axis)


# for i in range(features):
#     for j in range(features):
#         if (i!=j):
#             plt.figure()
#             for klass, color in zip(range(0, 9), colors):
#                 Xk = data[labels_050 == klass]
#                 print(Xk.shape)
#                 plt.plot(Xk[:, i], Xk[:, j], color, marker='.', linestyle='None', alpha=0.3)
#             plt.plot(data[:,0][labels_200 == -1], data[:,1][labels_200 == -1], "k+", alpha=0.1)
#             plt.xlabel(i)
#             plt.ylabel(j)