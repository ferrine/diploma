import unittest

import numpy as np
import theano.tensor as tt

from diploma.utils import lags


class TestUtils(unittest.TestCase):
    def test_lags(self):
        data = np.matrix([np.arange(8), np.arange(8)]).T
        vdata = np.arange(8)
        m = tt.matrix('m')
        v = tt.vector('v')
        lgs = lags(m, 3)
        lgsv = lags(v, 3)
        self.assertEqual(lgsv.eval({v: vdata}).shape, (6, 3), msg='shape got wrong(1d)')
        self.assertEqual(lgs.eval({m: data}).shape, (6, 2, 3), msg='shape got wrong(2d)')
        self.assertEqual(lgsv.eval({v: vdata}).tolist(),
                         [[2.,  1.,  0.],
                          [3.,  2.,  1.],
                          [4.,  3.,  2.],
                          [5.,  4.,  3.],
                          [6.,  5.,  4.],
                          [7.,  6.,  5.]],
                         msg='bad lag computation'
                         )
