import numpy as np
from . import common as c

def compute_cost_function(x_data, y_data, theta):
    coefficient = 0.5 / len(x_data)
    hyp_data = np.array([c.hyp_value(x, theta) for x in x_data])
    return sum(((y_data - hyp_data) ** 2)) * coefficient


def compute_partial_derivative(func, theta, theta_zero=False):
    result = 0.0
    for key in func:
        temp_sum = c.hyp_value(key, theta) - func[key]
        result += temp_sum if theta_zero else temp_sum * key

    return result / len(func)


def gradient_step(func, theta, alpha):
    theta_zero = theta[0] - alpha * compute_partial_derivative(func, theta, True)
    theta_one = theta[1] - alpha * compute_partial_derivative(func, theta)
    theta[0] = theta_zero
    theta[1] = theta_one
    return theta


def gradient_descent(x_data, y_data, theta, alpha, iterations):
    temp_dict = dict(zip(x_data, y_data))
    prev_theta = list(theta)
    for i in range(1, iterations):
        theta = gradient_step(temp_dict, theta, alpha)
        if c.compare_theta(prev_theta, theta):
            print('Estimate iterations: %d' % i)
            break

        prev_theta = list(theta)

    print('Theta 0: %s' % str(theta[0]))
    print('Theta 1: %s' % str(theta[1]))
    return theta
