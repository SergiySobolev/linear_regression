import numpy as np


def coefficient_of_determination(y, p_y):
    assert (len(y) == len(p_y)), "Vectors must be same length"
    y_m = np.mean(y)
    s_s_tot = np.sum((y - y_m)**2)
    residuals = np.diff(np.array([y, p_y]), axis=0)
    s_s_res = np.sum(residuals**2)
    return 1 - s_s_res / s_s_tot


def pearson_correlation_coefficient(y, p_y):
    assert (len(y) == len(p_y)), "Vectors must be same length"
    y_m = np.mean(y)
    p_y_m = np.mean(p_y)
    v1 = np.sum((y - y_m)*(p_y - p_y_m))
    v2 = np.sum((y - y_m)**2)
    v3 = np.sum((p_y - p_y_m)**2)
    return v1 / np.sqrt(v2 * v3)


def linear_correlation_coefficient(y, p_y):
    assert (len(y) == len(p_y)), "Vectors must be same length"
    n = len(y)
    y_s = np.sum(y)
    p_y_s = np.sum(p_y)
    v1 = n*np.sum(np.dot(y, p_y)) - y_s * p_y_s
    v2 = np.sqrt(n * np.dot(y,y) - y_s**2)
    v3 = np.sqrt(n * np.dot(p_y,p_y) - p_y_s ** 2)
    return v1/(v2*v3)