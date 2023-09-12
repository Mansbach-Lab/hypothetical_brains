from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.special import erf
from .due import due, Doi

from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial import cKDTree

# import os.path as op
from datetime import datetime
from os import mkdir
from os.path import join
from time import perf_counter#, time #time

import nibabel as nb
# import networkx as nx
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


__all__ = [
    "simple_function",
    "thresholding_weight",
    
#    "thresholding_distance",

    "ckd_made_weights",
    "ckd_made_distance",
    
    "loop_made_weights",
    "loop_made_distance",
    
    "squareform_made_weights",
    "squareform_made_distance",
    "generate_clusters",
    "generate_feature_matrix",
    "meanogram",
    
    "Model", 
    "Fit", 
    "opt_err_func", 
    "transform_data", 
    "cumgauss"
    ]


# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(Doi("10.1167/13.9.30"),
         description="Template project for small scientific Python projects",
         tags=["reference-implementation"],
         path='hypotheticalbrains')

def simple_function():
    print("huzzah!")
    return 42


def thresholding_weight(weight_matrix_flat, weight_threshold):
    """
    

    Parameters
    ----------
    weight_matrix_flat : TYPE
        DESCRIPTION.
    weight_threshold : TYPE
        DESCRIPTION.

    Returns
    -------
    weight_matrix_flat : TYPE
        DESCRIPTION.

    """
    # myfunc = np.vectorize(lambda a : 0.0 if (a < filter_threshold) else a)

    for i in range(len(weight_matrix_flat)):
        if weight_matrix_flat[i] < weight_threshold:
            weight_matrix_flat[i] = 0
    return weight_matrix_flat

# def thresholding_distance(distance_matrix_flat, distance_threshold):
#     for i in range(len(distance_matrix_flat)):
#         if distance_matrix_flat[i] > distance_threshold:
#             distance_matrix_flat[i] = 1
#     return distance_matrix_flat

def ckd_made_weights(feature_mat_scaled, width, distance_threshold, weight_threshold):
    """
    

    Parameters
    ----------
    feature_mat_scaled : TYPE
        DESCRIPTION.
    width : TYPE
        DESCRIPTION.
    distance_threshold : TYPE
        DESCRIPTION.
    weight_threshold : TYPE
        DESCRIPTION.

    Returns
    -------
    ckdtree_weight : TYPE
        DESCRIPTION.

    """
    # method 1 for sparse matrix; does distance only, cannot make weights
    kd_tree1 = cKDTree(feature_mat_scaled)
    kd_tree2 = cKDTree(feature_mat_scaled)
    ckdtree_distance = kd_tree1.sparse_distance_matrix(kd_tree2, distance_threshold).tocsr()
    ckdtree_weight = ckdtree_distance.copy()
    ckdtree_weight.data = thresholding_weight(np.exp(-1.*(ckdtree_distance.power(2)).data/width), weight_threshold)
    ckdtree_weight.eliminate_zeros()
    return ckdtree_weight

def ckd_made_distance(feature_mat_scaled, width, distance_threshold):
    """
    

    Parameters
    ----------
    feature_mat_scaled : TYPE
        DESCRIPTION.
    width : TYPE
        DESCRIPTION.
    distance_threshold : TYPE
        DESCRIPTION.

    Returns
    -------
    ckdtree_distance : TYPE
        DESCRIPTION.

    """
    # method 1 for sparse matrix; does distance only, cannot make weights
    kd_tree1 = cKDTree(feature_mat_scaled)
    kd_tree2 = cKDTree(feature_mat_scaled)
    ckdtree_distance = kd_tree1.sparse_distance_matrix(kd_tree2, distance_threshold).tocsr()
    ckdtree_distance = ckdtree_distance.power(2)
    ckdtree_distance.eliminate_zeros()
    return ckdtree_distance

    
def loop_made_weights(feature_mat_scaled, width, weight_threshold):
    """
    

    Parameters
    ----------
    feature_mat_scaled : TYPE
        DESCRIPTION.
    width : TYPE
        DESCRIPTION.
    weight_threshold : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    distance_matrix_flat = pdist(feature_mat_scaled, 'sqeuclidean')
    weight_matrix_flat = np.exp(-1.*distance_matrix_flat/width)
    
    # pruning edges less than filter_threshold
    filtered_weights = thresholding_weight(weight_matrix_flat, weight_threshold)

    # collecting data
    voxel_number=feature_mat_scaled.shape[0]
    weights_sparse_lil = lil_matrix((voxel_number, voxel_number)) 
    
    count = 0
    for row in range(voxel_number-1):
        
        weights_sparse_lil[row, row] = 1.
        
        for col in range(row+1, voxel_number): 
            
            # for upper triangle elements
            weights_sparse_lil[row, col] = (
                filtered_weights[count])
            
            # to add lower triangular elements
            weights_sparse_lil[col,row] = (
                filtered_weights[count])

            count += 1
    weights_sparse_lil[feature_mat_scaled.shape[0]-1,feature_mat_scaled.shape[0]-1] = 1.0
    return weights_sparse_lil.tocsr() 
    
def loop_made_distance(feature_mat_scaled, width):
    """
    

    Parameters
    ----------
    feature_mat_scaled : TYPE
        DESCRIPTION.
    width : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    distance_matrix_flat = pdist(feature_mat_scaled, 'sqeuclidean')

    # collecting data; two different approaches lil_matrix or for direct to CSR
    voxel_number=feature_mat_scaled.shape[0]
    distances_sparse_lil = lil_matrix((voxel_number, voxel_number)) 

    count = 0
    for row in range(voxel_number-1):
        for col in range(row+1, voxel_number): 
            
            # for upper triangle elements
            distances_sparse_lil[row, col] = (
                distance_matrix_flat[count])
            
            # to add lower triangular elements
            distances_sparse_lil[col,row] = (
                distance_matrix_flat[count])
            
            count += 1
     
    return distances_sparse_lil.tocsr()

# method 4 this bit takes too much memory for higher orders
def squareform_made_weights(feature_mat_scaled, width, weight_threshold):
    """
    

    Parameters
    ----------
    feature_mat_scaled : TYPE
        DESCRIPTION.
    width : TYPE
        DESCRIPTION.
    weight_threshold : TYPE
        DESCRIPTION.

    Returns
    -------
    sq_csr : TYPE
        DESCRIPTION.

    """
    weight_matrix_square = squareform(
        thresholding_weight(np.exp(-1.*pdist(feature_mat_scaled, 'sqeuclidean')/width),
    weight_threshold))
    sq_csr = csr_matrix(weight_matrix_square)
    sq_csr.setdiag(1.0)
    return sq_csr

def squareform_made_distance(feature_mat_scaled):
    """
    

    Parameters
    ----------
    feature_mat_scaled : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    weight_matrix_square = squareform(pdist(feature_mat_scaled, 'sqeuclidean'))
    return csr_matrix(weight_matrix_square)

def generate_feature_matrix(features_dir, stringname, samples, feature_number):
    """
    

    Parameters
    ----------
    stringname : TYPE
        DESCRIPTION.
    samples : TYPE
        DESCRIPTION.
    feature_number : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
        
    save_features_as = join(stringname+'features'+str(samples)+'.csv')
    # WM_mask = nb.load(join(features_dir, 'WM_mask_'+str(samples)+'.nii')).get_fdata().astype('bool')
    # AD = nb.load(join(features_dir, 'AD_'+str(samples)+'.nii')).get_fdata()
    # FA = nb.load(join(features_dir, 'FA_'+str(samples)+'.nii')).get_fdata()
    # MD = nb.load(join(features_dir, 'MD_'+str(samples)+'.nii')).get_fdata()
    # RD = nb.load(join(features_dir, 'RD_'+str(samples)+'.nii')).get_fdata()
    # ICVF = nb.load(join(features_dir, 'ICVF_'+str(samples)+'.nii')).get_fdata()
    # OD = nb.load(join(features_dir, 'OD_'+str(samples)+'.nii')).get_fdata()
    # ISOVF = nb.load(join(features_dir, 'ISOVF_'+str(samples)+'.nii')).get_fdata()
    
    AD = nb.load(join(features_dir,'sub-071_P_AD_WarpedToMNI.nii')).get_fdata()
    FA = nb.load(join(features_dir,'sub-071_P_FA_WarpedToMNI.nii')).get_fdata()
    MD = nb.load(join(features_dir,'sub-071_P_MD_WarpedToMNI.nii')).get_fdata()
    RD = nb.load(join(features_dir,'sub-071_P_RD_WarpedToMNI.nii')).get_fdata()
    ICVF = nb.load(join(features_dir,'sub-071_P_ICVF_WarpedToMNI.nii')).get_fdata()
    OD = nb.load(join(features_dir,'sub-071_P_OD_WarpedToMNI.nii')).get_fdata()
    ISOVF = nb.load(join(features_dir,'sub-071_P_ISOVF_WarpedToMNI.nii')).get_fdata()
    WM_mask = nb.load(join(features_dir,'Group_mean_CIRM_57_ACTION_5_MPRAGE0p9_T1w_brain_reg2DWI_0p9_T1_5tt_vol2_WM_WarpedToMNI_thr0p95_bin.nii')).get_fdata().astype('bool')
    
    AD_WM = AD[WM_mask]
    FA_WM = FA[WM_mask]
    MD_WM = MD[WM_mask]
    RD_WM = RD[WM_mask]
    ICVF_WM = ICVF[WM_mask]
    OD_WM = OD[WM_mask]
    ISOVF_WM = ISOVF[WM_mask]
    
    voxel_number = AD_WM.shape[0]
    
    # Make array containing all features (voxels in WM X features)
    feature_mat = np.zeros((voxel_number,feature_number))
    
    # Replace zeros in each column by the data of one of the metrics 
    feature_mat[:,0] = AD_WM
    feature_mat[:,1] = FA_WM
    feature_mat[:,2] = MD_WM
    feature_mat[:,3] = RD_WM
    feature_mat[:,4] = ICVF_WM
    feature_mat[:,5] = OD_WM
    feature_mat[:,6] = ISOVF_WM
        
    # scaling the MRI feature data
    scaler = StandardScaler()
    scaler.fit(feature_mat)
    feature_mat_scaled = scaler.transform(feature_mat)
        
    labeled_matrix = np.zeros((AD_WM.shape[0]+1,feature_number))
    labeled_matrix[0,:] = np.arange(0,feature_number,1)
    labeled_matrix[1::,:] = feature_mat_scaled
    
    np.savetxt(save_features_as,labeled_matrix, delimiter=",")
    
    return print("Attributes saved")


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
    dt_string = "/home/lwright/anaconda3/envs/networktoy/output/HypoBrains_" + dateonly + "_v" + str(voxel_count) + "_r" + str(r)
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
        cluster[:,0] = neighbour_idx
        cluster[:,1::] = feature_mat_scaled[neighbour_idx,:]
        
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
    # plt.hist(stats[:,:,1], bins=100000, range=None, density=None, weights=None)
    # plt.savefig(join(dt_string+ "/histogram.png"))
    
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

def meanogram(stats, metric, bincount, directory):
    means = stats[:,metric]
    voxelcount = len(means)
    plt.hist(means, bins=bincount)  # arguments are passed to np.histogram
    title_string = "Histogram of Means for metric=" + str(metric) + ", voxels=" + str(voxelcount) + ", bins=" + str(bincount)
    plt.title(title_string)
    save_string = directory +"histogram_vox"+ str(voxelcount) + "_metr" + str(metric) + "bins" + str(bincount) + ".png"
    plt.savefig(save_string)
    plt.show()










def transform_data(data):
    """
    Function that takes experimental data and gives us the
    dependent/independent variables for analysis.

    Parameters
    ----------
    data : Pandas DataFrame or string.
        If this is a DataFrame, it should have the columns `contrast1` and
        `answer` from which the dependent and independent variables will be
        extracted. If this is a string, it should be the full path to a csv
        file that contains data that can be read into a DataFrame with this
        specification.

    Returns
    -------
    x : array
        The unique contrast differences.
    y : array
        The proportion of '2' answers in each contrast difference
    n : array
        The number of trials in each x,y condition
    """
    if isinstance(data, str):
        data = pd.read_csv(data)

    contrast1 = data['contrast1']
    answers = data['answer']

    x = np.unique(contrast1)
    y = []
    n = []

    for c in x:
        idx = np.where(contrast1 == c)
        n.append(float(len(idx[0])))
        answer1 = len(np.where(answers[idx[0]] == 1)[0])
        y.append(answer1 / n[-1])
    return x, y, n


def cumgauss(x, mu, sigma):
    """
    The cumulative Gaussian at x, for the distribution with mean mu and
    standard deviation sigma.

    Parameters
    ----------
    x : float or array
       The values of x over which to evaluate the cumulative Gaussian function

    mu : float
       The mean parameter. Determines the x value at which the y value is 0.5

    sigma : float
       The variance parameter. Determines the slope of the curve at the point
       of Deflection

    Returns
    -------

    g : float or array
        The cumulative gaussian with mean $\\mu$ and variance $\\sigma$
        evaluated at all points in `x`.

    Notes
    -----
    Based on:
    http://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function

    The cumulative Gaussian function is defined as:

    .. math::

        \\Phi(x) = \\frac{1}{2} [1 + erf(\\frac{x}{\\sqrt{2}})]

    Where, $erf$, the error function is defined as:

    .. math::

        erf(x) = \\frac{1}{\\sqrt{\\pi}} \\int_{-x}^{x} e^{t^2} dt
    """
    return 0.5 * (1 + erf((x - mu) / (np.sqrt(2) * sigma)))


def opt_err_func(params, x, y, func):
    """
    Error function for fitting a function using non-linear optimization.

    Parameters
    ----------
    params : tuple
        A tuple with the parameters of `func` according to their order of
        input

    x : float array
        An independent variable.

    y : float array
        The dependent variable.

    func : function
        A function with inputs: `(x, *params)`

    Returns
    -------
    float array
        The marginals of the fit to x/y given the params
    """
    return y - func(x, *params)


class Model(object):
    """Class for fitting cumulative Gaussian functions to data"""
    def __init__(self, func=cumgauss):
        """ Initialize a model object.

        Parameters
        ----------
        data : Pandas DataFrame
            Data from a subjective contrast judgement experiment

        func : callable, optional
            A function that relates x and y through a set of parameters.
            Default: :func:`cumgauss`
        """
        self.func = func

    def fit(self, x, y, initial=[0.5, 1]):
        """
        Fit a Model to data.

        Parameters
        ----------
        x : float or array
           The independent variable: contrast values presented in the
           experiment
        y : float or array
           The dependent variable

        Returns
        -------
        fit : :class:`Fit` instance
            A :class:`Fit` object that contains the parameters of the model.

        """
        params, _ = opt.leastsq(opt_err_func, initial,
                                args=(x, y, self.func))
        return Fit(self, params)


class Fit(object):
    """
    Class for representing a fit of a model to data
    """
    def __init__(self, model, params):
        """
        Initialize a :class:`Fit` object.

        Parameters
        ----------
        model : a :class:`Model` instance
            An object representing the model used

        params : array or list
            The parameters of the model evaluated for the data

        """
        self.model = model
        self.params = params

    def predict(self, x):
        """
        Predict values of the dependent variable based on values of the
        indpendent variable.

        Parameters
        ----------
        x : float or array
            Values of the independent variable. Can be values presented in
            the experiment. For out-of-sample prediction (e.g. in
            cross-validation), these can be values
            that were not presented in the experiment.

        Returns
        -------
        y : float or array
            Predicted values of the dependent variable, corresponding to
            values of the independent variable.
        """
        return self.model.func(x, *self.params)
