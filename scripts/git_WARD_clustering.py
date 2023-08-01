#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:55:20 2023

@author: lwright
"""

# from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn import manifold, datasets
import matplotlib.pyplot as plt
import numpy as np
import umap


data = np.loadtxt('/home/lwright/brain_june14_features20.csv', delimiter=',')
data = data[1::,:]
labels = np.arange(0, len(data),1)
reducer = umap.UMAP()

embedding = reducer.fit_transform(data)
embedding.shape

x, y = embedding[:, 0], embedding[:, 1]

plt.scatter(x,y)

clust = OPTICS(min_samples=50, xi=0.015, min_cluster_size=0.01)
clust.fit(embedding)

# ----------------------------------------------------------------------
# Visualize the clustering
def plot_clustering(X_red, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for digit in digits.target_names:
        plt.scatter(
            *X_red[y == digit].T,
            marker=f"${digit}$",
            s=50,
            c=plt.cm.nipy_spectral(labels[y == digit] / 10),
            alpha=0.5,
        )

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# ----------------------------------------------------------------------
# 2D embedding of the digits dataset
print("Computing embedding")
X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
print("Done.")

from sklearn.cluster import AgglomerativeClustering

for linkage in ("ward", "average", "complete", "single"):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
    t0 = time()
    clustering.fit(X_red)
    print("%s :\t%.2fs" % (linkage, time() - t0))

    plot_clustering(X_red, clustering.labels_, "%s linkage" % linkage)


plt.show()