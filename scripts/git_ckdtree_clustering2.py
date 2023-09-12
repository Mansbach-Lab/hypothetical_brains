from __future__ import absolute_import, division, print_function
import os.path as op
# import hypotheticalbrains.hypotheticalbrains as hb
# import hypotheticalbrains.version as vhb
import hypotheticalbrains as hb
#from .hypotheticalbrains import *
import numpy as np
# import os
# print("package version = ", vhb.__version__)

# how many voxels are there per dimension of the MRI?
samples = 256 # whole brain = 256, likes r = 0.3 ish
# for importing and preparing features data
stringname = 'brain_aug17_features' 
import_data_from = op.join('/home/lwright/anaconda3/envs/networktoy/' 
                                            + stringname + str(samples)+'.csv')

# samples = 20
# stringname = 'brain_july26_features' # 'brain_aug17_features'
# import_data_from = op.join('/home/lwright/anaconda3/envs/networktoy/' 
#                                            + stringname + str(samples)+'.csv')

data = np.loadtxt(import_data_from, delimiter=',')
feature_mat_scaled = data[1::,:]

# maximum radius from central point in cluster
r = 0.3

# hb.simple_function()
# input("continue? [ctrl+c to exit]")

hb.generate_clusters(feature_mat_scaled, r, samples)

