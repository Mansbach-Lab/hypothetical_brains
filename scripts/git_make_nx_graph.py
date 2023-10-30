#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:17:53 2023

@author: lwright
"""

import numpy as np
import networkx as nx
from sklearn.decomposition import PCA


feature_list = ['FA', 'MD', 'OD', 'OS']
graph_dict = {}

array = np.array([[1, 2, 3, 5], [6, 7, 8, 10], [11, 12, 13, 15]])
feature_no = len(array[0])
for i in range(feature_no):
    feature = str(feature_list[i])
    graph_dict[feature] = array[:,i]
    
# attrs = {0: {"attr1": 20, "attr2": "nothing"}, 1: {"attr2": 3}}


# graph_dict = dict(zip(keys, array.T))
print(graph_dict)

G = nx.Graph()

nx.set_node_attributes(G, graph_dict, name=None)

G.nodes[10]["a"]

# dict of dict via pandas dataframes

# add edges from distance metric