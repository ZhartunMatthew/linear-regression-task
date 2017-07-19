import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    input_file = open(path)
    output_data = []
    for line in input_file:
        one_row = ['1']
        one_row += line.split(',')
        output_data.append([float(x) for x in one_row])

    print('Input data info.')
    print('\tVariable amount: %d, training set size %d' % (len(output_data[0]) - 1, len(output_data)))
    return np.array(output_data)


def display_logistic_data(data, hypothesis=None, labels=None):
    a_data = data[data[:, -1] == 0]
    b_data = data[data[:, -1] == 1]

    plot_regression = plt.figure().add_subplot(111)
    plot_regression.scatter(a_data[:, 1], a_data[:, 2], s=10, color='blue')
    plot_regression.scatter(b_data[:, 1], b_data[:, 2], s=10, color='red')

    if labels is not None:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.title(labels[2])

    # if hypothesis is not None:
    #     plot_regression.plot(x_data, hypothesis, color='red')


def display_all_plots():
    plt.show()


def compare_theta(a, b):
    delta = 0.000001
    for i in range(len(a)):
        if abs(a[i] - b[i]) > delta:
            return False

    return True


def display_convergence(data):
    convergence_plot_data = np.array(data)
    convergence_plot = plt.figure().add_subplot(111)
    plt.xlabel('Iteration, N')
    plt.ylabel('Cost function, J')
    plt.title('Cost function convergence')
    convergence_plot.plot(convergence_plot_data[:, 0], convergence_plot_data[:, 1])
