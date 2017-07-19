import numpy as np
from demo.init.coursera.linearregression.multivariableregression import multivatiable_common as mc


def compute_cost_function(x_data, y_data, theta):
    coefficient = 0.5 / len(x_data)
    hyp_data = np.array([mc.hyp_value(x, theta) for x in x_data])
    return sum(((hyp_data - y_data) ** 2)) * coefficient


def compute_partial_derivative(x_data, y_data, theta, j):
    result = 0.0
    for i in range(len(x_data)):
        result += (mc.hyp_value(x_data[i], theta) - y_data[i]) * x_data[i][j]

    return result / len(x_data)


def gradient_step(x_data, y_data, theta, alpha):
    return [theta[i] - alpha * compute_partial_derivative(x_data, y_data, theta, i) for i in range(len(theta))]


def gradient_descent(x_data, y_data, theta, alpha, iterations):
    prev_theta = list(theta)
    convergence_plot_data = []
    for i in range(1, iterations):
        theta = gradient_step(x_data, y_data, theta, alpha)
        if mc.compare_theta(prev_theta, theta):
            print('\tEstimated iterations: %d\n' % i)
            break

        prev_theta = list(theta)
        convergence_plot_data.append([i, compute_cost_function(x_data, y_data, theta)])

    if convergence_plot_data:
        mc.display_convergence(convergence_plot_data)
    return theta
