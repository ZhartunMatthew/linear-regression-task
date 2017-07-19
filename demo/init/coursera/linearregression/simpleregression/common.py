import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    input_file = open(path)
    output_data = []
    for line in input_file:
        one_row = line.split(',')
        output_data.append([float(one_row[0]), float(one_row[1])])
    return np.array(output_data)


def display_convergence(data):
    descent_plot_data = np.array(data)
    descent_plot = plt.figure().add_subplot(111)
    plt.xlabel('Iteration, N')
    plt.ylabel('Cost function, J')
    plt.title('Cost function convergence')
    descent_plot.plot(descent_plot_data[:, 0], descent_plot_data[:, 1])


def display_regression(data, hypothesis=None):
    x_data = data[:, 0]
    y_data = data[:, 1]
    plot_regression = plt.figure().add_subplot(111)
    plot_regression.scatter(x_data, y_data, s=10)
    plt.xlabel('Population size, 10.000')
    plt.ylabel('Profit, $10.000')
    plt.title('Linear regression')
    if hypothesis is not None:
        plot_regression.plot(x_data, hypothesis, color='red')
    plt.show()


def hyp_value(x, theta):
    return theta[0] + x * theta[1]


def compare_theta(a, b):
    delta = 0.000001
    return True if abs(a[0] - b[0]) < delta and abs(a[1] - b[1]) < delta else False
