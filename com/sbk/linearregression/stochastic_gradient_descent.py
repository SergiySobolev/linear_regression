import numpy as np


def coefficient_of_determination(y, p_y):
    assert (len(y) == len(p_y)), "Vectors must be same length"
    y_m = np.mean(y)
    s_s_tot = np.sum((y - y_m)**2)
    residuals = np.diff(np.array([y, p_y]), axis=0)
    s_s_res = np.sum(residuals**2)
    return 1 - s_s_res / s_s_tot


def compute_cost_function(theta, x, y):
    x_arr = np.asarray(x) if type(x) is list else x
    m = x_arr.shape[0]
    ones = np.ones((m, 1))
    xa = np.hstack((ones, x))
    approx_value = np.sum(theta * xa, axis=1)
    dif = approx_value - y
    return sum(dif ** 2) / (2 * m)


def compute_gradient(theta, x, y):
    x_arr = np.asarray(x) if type(x) is list else x
    m = x_arr.shape[0]
    ones = np.ones((m, 1))
    xa = np.hstack((ones, x))
    approx_value = np.sum(theta * xa, axis=1)
    dif = approx_value - y
    grad0 = np.asarray([sum(dif) / m])
    gradn = np.dot(np.asarray(dif), x) / m
    grad = np.concatenate((grad0, gradn), axis=0)
    return grad


def propose_theta(data):
    return np.ones(data.shape[1])


def update_theta(theta, gradient, alpha):
    return theta - alpha * gradient


def calc_theta_for_batch(batch_data, min_theta, alpha):
    batch_n = batch_data.shape[1] - 1
    batch_x = batch_data[:, range(batch_n)]
    batch_y = batch_data[:, batch_n]
    g = compute_gradient(min_theta, batch_x, batch_y)
    return update_theta(min_theta, g, alpha)


def vanilla_gradient_descent(data, start_theta=None, alpha=None, max_iter=None):

    if start_theta is None:
        start_theta = propose_theta(data)

    if alpha is None:
        alpha = 0.001

    if max_iter is None:
        max_iter = 40


    f_n = data.shape[1] - 1
    x = data[:, range(f_n)]
    y = data[:, f_n]
    iter_num = 0
    cur_theta = start_theta
    while iter_num < max_iter:
        gradient = compute_gradient(cur_theta, x, y)
        cur_theta = update_theta(cur_theta, gradient, alpha)
        iter_num += 1
    return cur_theta


def mini_batch_gradient_descent(data, start_theta=None, alpha=None, max_iter=None, batch_size=5):
    if start_theta is None:
        start_theta = propose_theta(data)

    if alpha is None:
        alpha = 0.001

    if max_iter is None:
        max_iter = 100

    min_theta = start_theta
    min_cost = float("inf")
    f_n = data.shape[1] - 1

    iteration_without_improvement = 0
    cost_history = []
    cur_alpha = alpha
    while iteration_without_improvement < max_iter:
        np.random.shuffle(data)
        batch_data = data[0:batch_size]
        cur_theta = calc_theta_for_batch(batch_data, min_theta, cur_alpha)
        x = data[:, range(f_n)]
        y = data[:, f_n]
        cur_cost = compute_cost_function(cur_theta, x, y)
        cost_history.append(cur_cost)
        if cur_cost < min_cost:
            min_cost = cur_cost
            min_theta = cur_theta
            iteration_without_improvement = 0
            cur_alpha *= 0.95
        else:
            iteration_without_improvement+=1

    return min_theta, min(cost_history)