from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import pandas as pd
import numpy.testing as npt
import hypotheticalbrains as hb

data_path = op.join(hb.__path__[0], 'data')


def test_dummy():
    
    from os.path import join
    import networkx as nx
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import pdist, squareform
    from scipy.sparse import csr_matrix, csr_array, lil_matrix
    from scipy.spatial import cKDTree
    
    # how many voxels are there per dimension of the MRI?
    samples = 20
    
    # set number of data points per cluster
    cluster_members = 200
     
    # for importing features file
    stringname = 'brain_june07_'
    import_data_from = join('/home/lwright/anaconda3/envs/networktoy/' 
                            + stringname + 'features' + str(samples)+'.csv')
    data = np.loadtxt(import_data_from, delimiter=',')
    feature_mat_scaled = data[1::,:]
    voxel_number = np.arange(0, len(data),1)
    features = len(data[0])
    
    width = 1.
    distance_threshold = 100
    
    
    #make matrix one way
    A = hb.squareform_made_matrix(feature_mat_scaled, width)
    #make matrix second way
    B = hb.loop_made_matrix(feature_mat_scaled, width)
    C = hb.ckd_made_matrix(feature_mat_scaled, width, distance_threshold)
    npt.assert_array_almost_equal(A, B)
