#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:17:53 2023

@author: lwright
"""

import numpy as np
import networkx as nx
from sklearn.decomposition import PCA


import_from = '/home/lwright/anaconda3/envs/networktoy/output/cluster5000.gexf'
G = nx.read_gexf(import_from)
Ns = G.number_of_nodes()

print(Ns, "  , ", str(Ns*Ns))

print(G.number_of_edges())

# https://stackoverflow.com/questions/28281850/list-of-attributes-available-to-use-in-networkx
attrs = set(np.array([list(G.nodes[n].keys()) for n in G.nodes()]).flatten())
print(attrs)
# nx.draw(G, with_labels=True, font_weight='bold')

G.nodes['81'][1]



"""

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

# add edges from distance metric"""