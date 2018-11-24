import numpy as np


def coefficient_of_determination(y, p_y):
    assert (len(y) == len(p_y)), "Vectors must be same length"
    y_m = np.mean(y)
    s_s_tot = np.sum((y - y_m)**2)
    residuals = np.diff(np.array([y, p_y]), axis=0)
    s_s_res = np.sum(residuals**2)
    return 1 - s_s_res / s_s_tot