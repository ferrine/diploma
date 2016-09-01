from theano import tensor as tt


def lags(x, k=1, axis=-1):
    """Implements Lag operations over main axis
    Notice if you create probabilistic model
    don't forget first k-1 observations

    Parameters
    ----------
    x : tensor
    k : int - lag order
    axis : int - axis to extend
    Returns
    -------
    tensor

    Example
    -------
    #>>> import numpy as np
    #>>> data = np.matrix([np.arange(8), np.arange(8)]).T
    #>>> data.shape
    (8, 2)
    #>>> m = tt.matrix('m')
    #>>> lgs = lags(m, 3)
    #>>> lgs.eval({m:data}).shape
    (6, 2, 3)
    #>>> vdata = np.arange(8)
    #>>> vdata.shape
    (8,)
    #>>> v = tt.vector('v')
    #>>> lgsv = lags(v, 3)
    #>>> lgsv.eval({v:vdata}).shape
    (6, 3)
    #>>> vdata
    array([0, 1, 2, 3, 4, 5, 6, 7])
    #>>> lgsv.eval({v:vdata})
    array([[ 2.,  1.,  0.],
       [ 3.,  2.,  1.],
       [ 4.,  3.,  2.],
       [ 5.,  4.,  3.],
       [ 6.,  5.,  4.],
       [ 7.,  6.,  5.]])
    """
    m = x.shape[0]  # total length
    slices = []
    for i in range(k):
        slices.append(x[k - i - 1:m - i])
    return tt.stack(slices, axis=axis)