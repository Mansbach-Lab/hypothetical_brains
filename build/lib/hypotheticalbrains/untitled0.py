#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 09:20:16 2023

@author: lwright
"""
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA

dt_string = "/home/lwright/anaconda3/envs/networktoy/output/HypoBrains_Y2023_M10_D03_H15_M03_S44_v359938_r0.3" 
graphtoy = nx.read_gexf(dt_string+'/cluster0.gexf')


feature_list = ['FA', 'MD', 'OD', 'OS']
graph_dict = {}

array = np.array([[1, 2, 3, 5], [6, 7, 8, 10], [11, 12, 13, 15]])
feature_no = len(array[0])
for i in range(feature_no):
    feature = str(feature_list[i])
    graph_dict[feature] = array[:,i]
   
# attrs = {0: {"attr1": 20, "attr2": "nothing"}, 1: {"attr2": 3}}

desrattrib = {0: {"FA":1, "MD": 6,"OD": 11},
              1: {"FA":2, "MD": 7, "OD": 12},
              2: {"FA":3, "MD": 8, "OD": 13},
              3: {"FA": 5, "MD": 10, "OD": 15}}

# graph_dict = dict(zip(keys, array.T))
print(graph_dict)

G = nx.Graph()

nx.set_node_attributes(G, desrattrib, name=None)

G.nodes[0]["FA"]

# dict of dict via pandas dataframes

# add edges from distance metric