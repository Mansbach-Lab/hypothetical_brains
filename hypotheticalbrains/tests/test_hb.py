from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import pandas as pd
import numpy.testing as npt
import hypotheticalbrains as hb

data_path = op.join(hb.__path__[0], 'data')


def test_dummy():
    #make matrix one way
    A = matrix made one way
    #make matrix second way
    B = matrix made second way
    B = reshaped matrix made second way
    npt.assert_array_almost_equal(A, B)
