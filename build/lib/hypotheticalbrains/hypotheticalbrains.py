from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.special import erf
from .due import due, Doi

# from os.path import join
# import networkx as nx
# from sklearn.preprocessing import StandardScaler
# from scipy.spatial.distance import pdist, squareform
# from scipy.sparse import csr_matrix, csr_array, lil_matrix
from scipy.spatial import cKDTree
    
__all__ = ["silly_function","Model", "Fit", "opt_err_func", "transform_data", "cumgauss"]


# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(Doi("10.1167/13.9.30"),
         description="Template project for small scientific Python projects",
         tags=["reference-implementation"],
         path='shablona')

def silly_function(a,b):
    c = a+b
    return c

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



myfunc = np.vectorize(lambda a : 0.0 if (a < filter_threshold) else a)


def ckd_made_matrix(feature_mat_scaled, width, distance_threshold):
    # method 1 for sparse matrix; does distance only, cannot make weights
    kd_tree1 = cKDTree(feature_mat_scaled)
    kd_tree2 = cKDTree(feature_mat_scaled)
    ckdtree_distance = kd_tree1.sparse_distance_matrix(kd_tree2, distance_threshold).tocsr()
    ckdtree_weight = ckdtree_distance.copy()
    ckdtree_weight.data = np.exp(-1.*myfunc(ckdtree_distance.data)/width)
    return ckdtree_weight

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
# or cdist?
# can consolidate steps into Y = pdist(X, f) where f is user defined function.
# Computes the distance between all pairs of vectors in X using the user supplied 2-arity function f. 

def loop_made_matrix(feature_mat_scaled, width):
    distance_matrix_flat = pdist(feature_mat_scaled, 'sqeuclidean')
    weight_matrix_flat = np.exp(-1.*distance_matrix_flat/width)
    
    # pruning edges less than filter_threshold
    filtered_weights = myfunc(weight_matrix_flat)

    # collecting data; two different approaches lil_matrix or for direct to CSR
    weights_sparse_lil = lil_matrix((voxel_number, voxel_number)) 
    # row_idx, col_idx = [],[]
    
    count = 0
    for row in range(voxel_number-1):
        for col in range(row+1, voxel_number):
            # row_idx.append(row)
            # col_idx.append(col)
            weights_sparse_lil[row, col] = filtered_weights[count]
            count += 1
    print("exit loop")
    
    # method 2 for making sparse array of weights - think this is best
    weights_sparse_csr = weights_sparse_lil.tocsr()
    # method 3 for making sparse array of weights - seems memory-heavy
    # rows = np.array(row_idx)
    # columns = np.array(col_idx)
    # distances_sparse = csr_array((filtered_weights, (row_idx, col_idx)), shape=(voxel_number, voxel_number)) 
    return weights_sparse_csr

# method 4 this bit takes too much memory for higher orders
def squareform_made_matrix(feature_mat_scaled, width):
    distance_matrix_square = squareform(myfunc(np.exp(-1.*pdist(feature_mat_scaled, 'sqeuclidean')/width)))
    return csr_matrix(distance_matrix_square)

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
