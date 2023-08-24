# import numpy as np
# from scipy.spatial import cKDTree
# from datetime import datetime
# from os import mkdir
# from os.path import join
# from time import perf_counter #time

import os.path as op
import hypotheticalbrains as hb
import numpy as np

# how many voxels are there per dimension of the MRI?
samples = 256 # whole brain = 256

# for importing and preparing features data
stringname = 'brain_aug17_features'
import_data_from = op.join('/home/lwright/anaconda3/envs/networktoy/' 
                                           + stringname + str(samples)+'.csv')
data = np.loadtxt(import_data_from, delimiter=',')
feature_mat_scaled = data[1::,:]

# maximum radius from central point in cluster
r = 0.9


hb.generate_clusters(feature_mat_scaled, samples, r)








"""data = np.loadtxt(import_data_from, delimiter=',')
feature_mat_scaled = data[1::,:]

# feature_mat_scaled, r = np.array([[0,0],[1,1],[4,4],[5,5]]), 3 # demo case

voxelcount = len(feature_mat_scaled)
features = len(feature_mat_scaled[0])

start_time = perf_counter()
now = datetime.now()
dt_string = now.strftime("HypoBrains_Y%Y_M%m_D%d_H%H_M%M_S%S")
mkdir(dt_string)
tree = cKDTree(feature_mat_scaled)
# mrRogers = []
maximum,minimum,average = 0, len(feature_mat_scaled), 0

stats = np.zeros((voxelcount,features,2))

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
print(dt_string)"""
