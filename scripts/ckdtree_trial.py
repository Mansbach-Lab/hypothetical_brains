

import os.path as op
import numpy as np
from scipy.spatial import cKDTree
from datetime import datetime
from os import mkdir
from os.path import join
from time import perf_counter #time


start_time = perf_counter()

# how many voxels are there per dimension of the MRI?
samples = 20 # whole brain = 256
# for importing features file
stringname = 'brain_july26_'
import_data_from = op.join('/home/lwright/anaconda3/envs/networktoy/' 
                        + stringname + 'features' + str(samples)+'.csv')
data = np.loadtxt(import_data_from, delimiter=',')


feature_mat_scaled = data[1::,:]
voxelcount = len(feature_mat_scaled)
features = len(feature_mat_scaled[0])

# feature_mat_scaled, r = np.array([[0,0],[1,1],[4,4],[5,5]]), 3 # demo case

r = 0.9

now = datetime.now()
dt_string = now.strftime("HypoBrains_Y%Y_M%m_D%d_H%H_M%M_S%S")
mkdir(dt_string)
tree = cKDTree(feature_mat_scaled)
# mrRogers = []
maximum,minimum,average = 0, len(feature_mat_scaled), 0

stats = np.zeros((voxelcount,features,2))

# means = np.zeros((voxelcount,features))
# stdevs = np.zeros((voxelcount,features))


for i in range(voxelcount):
    neighbour_idx = np.array(cKDTree.query_ball_point(tree,feature_mat_scaled[i],r,p=2., 
                    eps=0, workers=-1, return_sorted=True, return_length=False))
    temp = len(neighbour_idx)
    cluster = np.zeros((temp,features+1))
    for j in range(temp):
        cluster[j,0] = neighbour_idx[j]
        cluster[j,1::] = feature_mat_scaled[neighbour_idx[j],:]
    if maximum < temp:
        maximum = temp
    if minimum > temp:
        minimum = temp
    average += temp
    stats[i,:,0]=np.mean(cluster[:,1::],axis=0)
    stats[i,:,1]=np.std(cluster[:,1::],axis=0)
    # means[i,:]=np.mean(cluster[:,1::],axis=0)
    # stdevs[i,:]=np.std(cluster[:,1::],axis=0)

    # mrRogers.append(neighbour_idx)
    
    loc = join("./"+ dt_string+ "/cluster"+ str(i)+ ".txt")
    np.savetxt(loc, cluster, delimiter=",")
    print("time saved voxel ", str(i),"/", str(voxelcount), " : ", (perf_counter() - start_time))

average/=voxelcount
print(maximum, " ", minimum)
time1 = perf_counter() - start_time
print("time 1: ", (perf_counter() - start_time))

summary_notes = (  "Project HypoBrains"  
                 + "\nRead me: summary of parameters for " +dt_string
                 + "\nSource file: " +import_data_from
                 + "\nSamples: "+str(samples)
                 + "\nTotal number of voxels: " +str(voxelcount)
                 + "\nTotal number of features: " +str(features)
                 + "\nRadius, r="+str(r)
                 + "\nMax number of voxels per cluster: "+str(maximum)
                 + "\nMin number of voxels per cluster: "+str(minimum)
                 + "\nAverage number of voxels per cluster: "+str(average)
                 + "\nNumber of clusters = number of voxels by cKDTree definition"
                 + "\nRun time: {:.2f}".format(time1)
                 )

loc_readme = join("./"+ dt_string+ "/"+str(dt_string)+"_readme.txt")

readme = open(loc_readme, "w")
n = readme.write(summary_notes)
readme.close()
print(dt_string)



#%%
"""def ckd_made_weights(feature_mat_scaled, width, distance_threshold, weight_threshold):
    # method 1 for sparse matrix; does distance only, cannot make weights
    kd_tree1 = cKDTree(feature_mat_scaled)
    kd_tree2 = cKDTree(feature_mat_scaled)
    ckdtree_distance = kd_tree1.sparse_distance_matrix(kd_tree2, distance_threshold).tocsr()
    ckdtree_weight = ckdtree_distance.copy()
    ckdtree_weight.data = thresholding_weight(np.exp(-1.*(ckdtree_distance.power(2)).data/width), weight_threshold)
    ckdtree_weight.eliminate_zeros()
    return ckdtree_weight

def ckd_made_distance(feature_mat_scaled, width, distance_threshold):
    # method 1 for sparse matrix; does distance only, cannot make weights
    kd_tree1 = cKDTree(feature_mat_scaled)
    kd_tree2 = cKDTree(feature_mat_scaled)
    ckdtree_distance = kd_tree1.sparse_distance_matrix(kd_tree2, distance_threshold).tocsr()
    ckdtree_distance = ckdtree_distance.power(2)
    ckdtree_distance.eliminate_zeros()
    return ckdtree_distance

#make matrix one way
squareform_weights = hb.squareform_made_weights(feature_mat_scaled, width, weight_threshold)
squareform_distances = hb.squareform_made_distance(feature_mat_scaled)

mem = max(memory_usage(proc=hb.squareform_made_weights(feature_mat_scaled, width, weight_threshold)))
print("Maximum memory used: {} MiB".format(mem))

mem = max(memory_usage(proc=hb.squareform_made_distance(feature_mat_scaled)))
print("Maximum memory used: {} MiB".format(mem))


#make matrix second way
# logging.warning("B is upper triangular")
weights_sparse_csr = hb.loop_made_weights(feature_mat_scaled, width, weight_threshold) 
distances_sparse_csr = hb.loop_made_distance(feature_mat_scaled, width)

mem = max(memory_usage(proc=hb.loop_made_weights(feature_mat_scaled, width, weight_threshold)))
print("Maximum memory used: {} MiB".format(mem))

mem = max(memory_usage(proc=hb.loop_made_distance(feature_mat_scaled, width)))
print("Maximum memory used: {} MiB".format(mem))


#make matrix third way
ckdtree_weight = hb.ckd_made_weights(feature_mat_scaled, width, distance_threshold, weight_threshold)
ckdtree_distance = hb.ckd_made_distance(feature_mat_scaled, width, distance_threshold)
    
mem = max(memory_usage(proc=hb.ckd_made_weights(feature_mat_scaled, width, distance_threshold, weight_threshold)))
print("Maximum memory used: {} MiB".format(mem))

mem = max(memory_usage(proc=hb.ckd_made_distance(feature_mat_scaled, width, distance_threshold)))
print("Maximum memory used: {} MiB".format(mem))


 
#compare matrices
# npt.assert_array_almost_equal(weights_sparse_csr.todense(), squareform_weights.todense())
# npt.assert_array_almost_equal(weights_sparse_csr.todense(), ckdtree_weight.todense())
# npt.assert_array_almost_equal(distances_sparse_csr.todense(), squareform_distances.todense())
npt.assert_array_almost_equal(ckdtree_distance.todense(), distances_sparse_csr.todense())

"""
