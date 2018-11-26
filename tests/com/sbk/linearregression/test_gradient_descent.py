import unittest

import numpy as np
import pandas as pd

from com.sbk.linearregression.stochastic_gradient_descent import vanilla_gradient_descent, coefficient_of_determination


class TestGradientDescent(unittest.TestCase):

    def test_vanilla_gradient_descent(self):
        df_adv = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
        x = np.around(df_adv['TV'], decimals=0)
        y = np.around(df_adv['radio'], decimals=0)
        z = np.around(df_adv['sales'], decimals=0)
        r = np.column_stack((x, y, z))
        theta = vanilla_gradient_descent(r, start_theta=(1, 0, 0), alpha=0.00005, max_iter=5500)
        pred_c = theta[0] + theta[1] * x + theta[2] * y
        cd = coefficient_of_determination(y, pred_c)
        self.assertAlmostEqual(cd, -0.0908, places=3)
        rmse = np.sqrt(np.mean((pred_c - z) ** 2))
        self.assertAlmostEqual(rmse, 1.841, places=2)