"""

Created on Fri Sep  1 12:07:11 2023

@author: lwright

This code is to demonstrate my method of data file organisation. The clustering
algorithm generates one csv file of data per cluster, up to 350 000 files.
To keep them organised, I create a directory labeled with the project code and
the run starttime, and save the output files to this directory. Additionally,
two extra csv output files containing the run's cluster metrics are added. 
Finally, a readme file containing important run parameters is saved to the directory.

"""

import numpy as np
from scipy.spatial import cKDTree
from datetime import datetime
from os import mkdir
from os.path import join
from time import perf_counter#, time #time


def generate_clusters(feature_mat_scaled, r, samples=0, import_data_from=''):
    """
    
    Parameters
    ----------
    
    import_data_from : str
        Path to csv file containing (voxel, feature) sacled matrix brain MRI data.
        
    samples : int
        Dimensions of brain MRI in voxels, e.g. 256x256x256 and sample = 256.
        
    r : float
        Maximum radius for inclusion in nearest neighbour cluster
        as calculated in featurespace, euclidean distance.

    Returns
    -------
    
    minimum,maximum,average
        integer values for minimum, maximum, and average number of neighbours
        in any given cluster of a single brain.

    """
    
    # how many voxels total, how many features per voxel    
    voxel_count = len(feature_mat_scaled)
    features = len(feature_mat_scaled[0])
    
    # run time, start stopwatch    
    start_time = perf_counter()
    
    # creating a directory labeled with project name and start date+time of run
    now = datetime.now()
    dateonly = now.strftime("Y%Y_M%m_D%d_H%H_M%M_S%S")
    dt_string = "/home/lwright/anaconda3/envs/networktoy/output/HypoBrains_" + dateonly
    mkdir(dt_string)
    
    # create the tree from which the neighbours can be found
    tree = cKDTree(feature_mat_scaled)
    
    # variables to collect cluster metrics
    maximum,minimum,average = 0, voxel_count, 0
    stats = np.zeros((voxel_count,features,2))
    
    # find the nearest neighbours of each voxel in the brain
    for i in range(voxel_count):
        
        # gives the indices in feature_mat_scaled of all nearest neighbours to voxel i
        neighbour_idx = np.array(cKDTree.query_ball_point(tree,
                                feature_mat_scaled[i],r,p=2., 
                                eps=0, workers=-1, return_sorted=True, 
                                return_length=False))
        
        # total number of neighbours in ith cluster
        neighbour_count = len(neighbour_idx)
        
        # collect the cluster (voxels, neighbour_idx+features)
        cluster = np.zeros((neighbour_count,features+1))
        cluster[neighbour_idx,0] = neighbour_idx
        cluster[neighbour_idx,1::] = feature_mat_scaled[neighbour_idx,:]
        
        # collecting cluster metrics                     
        if maximum < neighbour_count:
            maximum = neighbour_count
        if minimum > neighbour_count:
            minimum = neighbour_count
        average += neighbour_count 
        stats[i,:,0]=np.mean(cluster[:,1::],axis=0)
        stats[i,:,1]=np.std(cluster[:,1::],axis=0)
            
        # create a csv file containing cluster data, save to run directory
        loc = join(dt_string+ "/cluster"+ str(i)+ ".csv")
        np.savetxt(loc, cluster, delimiter=",")
        print("time saved voxel ", str(i),"/", str(voxel_count), " : ", 
                                          (perf_counter() - start_time))
    
    # saving the run's cluster metrics
    loc_means = join(dt_string+ "/means.csv")
    loc_sd = join(dt_string+ "/sd.csv")
    np.savetxt(loc_means, stats[:,:,0], delimiter=",")
    np.savetxt(loc_sd, stats[:,:,1], delimiter=",")
    
    # calculate average cluster size
    average/=voxel_count
    
    #final runtime
    runtime = perf_counter() - start_time 
    print("Total runtime: ", runtime)
    
    # save values of important parameters to readme file in run directory
    summary_notes = (  "Project HypoBrains"  
                     + "\nRead me: summary of parameters for " +dt_string
                     + "\nSource file: " +import_data_from
                     + "\nSamples: "+str(samples)
                     + "\nTotal number of voxels: " +str(voxel_count)
                     + "\nTotal number of features: " +str(features)
                     + "\nRadius, r="+str(r)
                     + "\nMax number of voxels per cluster: "+str(maximum)
                     + "\nMin number of voxels per cluster: "+str(minimum)
                     + "\nAverage number of voxels per cluster: "+str(average)
                     + "\nNumber of clusters = number of voxels by cKDTree definition"
                     + "\nRun time: {:.2f}".format(runtime)
                     )
    
    loc_readme = join(dt_string+ "/"+str(dateonly)+"_readme.txt")
    readme = open(loc_readme, "w")
    n = readme.write(summary_notes)
    readme.close()
    print(dt_string)
    return minimum,maximum,average
