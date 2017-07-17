import numpy as np
import matplotlib.pyplot as plt


def warm_up():
    identity_matrix = np.identity(5, int)
    return identity_matrix


def load_data(path):
    input_file = open(path)
    output_data = []
    for line in input_file:
        one_row = line[:-2].split(',')
        output_data.append([float(one_row[0]), float(one_row[1])])
    return np.array(output_data)


def display_data(data, hypothesis=None):
    x_data = data[:, 0]
    y_data = data[:, 1]
    plt.scatter(x_data, y_data, s=10)
    plt.xlabel('Population size, 10.000')
    plt.ylabel('Profit, $10.000')
    plt.title('Plot')
    if hypothesis is not None:
        plt.plot(x_data, hypothesis, color='red')
    plt.show()


def hyp_value(x, theta):
    return theta[0] + x * theta[1]


def compare_theta(a, b):
    delta = 0.000001
    return True if abs(a[0] - b[0]) < delta and abs(a[1] - b[1]) < delta else False
