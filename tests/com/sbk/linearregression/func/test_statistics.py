import unittest
import numpy as np
from parameterized import parameterized
from sklearn.metrics import r2_score
from com.sbk.linearregression.func.statistics import coefficient_of_determination, pearson_correlation_coefficient


class TestStatistics(unittest.TestCase):

    def test_coefficient_of_determination_generated(self):
        len = 10
        max = 20
        y = np.random.randint(max, size=len)
        p_y = np.random.randint(max, size=len)
        expected = r2_score(y, p_y)
        actual = coefficient_of_determination(y, p_y)
        self.assertAlmostEqual(actual, expected)

    @parameterized.expand([
        ["first", (4, 15, 3, 10, 15, 9), (9, 17, 8, 11, 14, 12), .5125],
        ["second", (1, 2, 3), (1, 2, 3), 1],
        ["third", (1, 2, 3), (3, 2, 1), -3],
        ["fourth", (1, 2, 3), (2, 2, 2), 0],
    ])
    def test_coefficient_of_determination_parametrized(self, name, y, p_y, expected):
        actual = coefficient_of_determination(y, p_y)
        self.assertEqual(actual, expected)

    def test_pearson_correlation_coefficient(self):
        len = 10
        max = 20
        y = np.random.randint(max, size=len)
        p_y = np.random.randint(max, size=len)
        expected = np.corrcoef(y, p_y)[1,0]
        actual = pearson_correlation_coefficient(y, p_y)
        self.assertAlmostEqual(actual, expected)