
from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import pandas as pd
import numpy.testing as npt
import hypotheticalbrains as hb

data_path = op.join(hb.__path__[0], 'data')

"""
def test_dummy():
    

    
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
    
    # A_mat = hb.making_distance_matrix()
    # B_mat = hb.making_distance_matrix()
    # C_mat = hb.making_distance_matrix()
    
    #make matrix one way
    A = hb.squareform_made_matrix(feature_mat_scaled, width)
    #make matrix second way
    B = hb.loop_made_matrix(feature_mat_scaled, width)
    #make matrix third way
    C = hb.ckd_made_matrix(feature_mat_scaled, width, distance_threshold)
    npt.assert_array_almost_equal(A, B)
"""

def test_silly_function():
    a = 1
    b = 2
    c = hb.silly_function(a,b)
    assert(c==(a+b))