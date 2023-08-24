
from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
# import pandas as pd
import numpy.testing as npt
import hypotheticalbrains as hb

# from memory_profiler import profile, memory_usage
# import requests

data_path = op.join(hb.__path__[0], 'data')


def test_compare_distance_methods():

    # how many voxels are there per dimension of the MRI?
    samples = 20
    
    # for importing features file
    stringname = 'brain_june07_'
    import_data_from = op.join('/home/lwright/anaconda3/envs/networktoy/' 
                            + stringname + 'features' + str(samples)+'.csv')
    data = np.loadtxt(import_data_from, delimiter=',')
    feature_mat_scaled = data[1::,:]

    width = 1.
    distance_threshold = 1000
    weight_threshold = 0.3
    
    
    #make matrix one way
    squareform_weights = hb.squareform_made_weights(feature_mat_scaled, width, weight_threshold)
    squareform_distances = hb.squareform_made_distance(feature_mat_scaled)
    
    # mem = max(memory_usage(proc=hb.squareform_made_weights(feature_mat_scaled, width, weight_threshold)))
    # print("Maximum memory used: {} MiB".format(mem))
    
    # mem = max(memory_usage(proc=hb.squareform_made_distance(feature_mat_scaled)))
    # print("Maximum memory used: {} MiB".format(mem))
    
    
    #make matrix second way
    # logging.warning("B is upper triangular")
    weights_sparse_csr = hb.loop_made_weights(feature_mat_scaled, width, weight_threshold) 
    distances_sparse_csr = hb.loop_made_distance(feature_mat_scaled, width)
    
    # mem = max(memory_usage(proc=hb.loop_made_weights(feature_mat_scaled, width, weight_threshold)))
    # print("Maximum memory used: {} MiB".format(mem))
    
    # mem = max(memory_usage(proc=hb.loop_made_distance(feature_mat_scaled, width)))
    # print("Maximum memory used: {} MiB".format(mem))
    
    
    #make matrix third way
    ckdtree_weight = hb.ckd_made_weights(feature_mat_scaled, width, distance_threshold, weight_threshold)
    ckdtree_distance = hb.ckd_made_distance(feature_mat_scaled, width, distance_threshold)
        
    # mem = max(memory_usage(proc=hb.ckd_made_weights(feature_mat_scaled, width, distance_threshold, weight_threshold)))
    # print("Maximum memory used: {} MiB".format(mem))
    
    # mem = max(memory_usage(proc=hb.ckd_made_distance(feature_mat_scaled, width, distance_threshold)))
    # print("Maximum memory used: {} MiB".format(mem))
    
    
     
    #compare matrices
    # npt.assert_array_almost_equal(weights_sparse_csr.todense(), squareform_weights.todense())
    # npt.assert_array_almost_equal(weights_sparse_csr.todense(), ckdtree_weight.todense())
    # npt.assert_array_almost_equal(distances_sparse_csr.todense(), squareform_distances.todense())
    npt.assert_array_almost_equal(ckdtree_distance.todense(), distances_sparse_csr.todense())

def test_generate_clusters():
    
    feature_mat_scaled, r = np.array([[0,0],[1,1],[4,4],[5,5]]), 3 # demo case
    minimum,maximum,average = hb.generate_clusters(feature_mat_scaled, r)
    if minimum ==2 and maximum ==2 and average ==2:
        assert True
    else:
        assert False

