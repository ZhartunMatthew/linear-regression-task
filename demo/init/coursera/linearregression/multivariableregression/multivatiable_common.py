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


def display_convergence(data):
    convergence_plot_data = np.array(data)
    convergence_plot = plt.figure().add_subplot(111)
    plt.xlabel('Iteration, N')
    plt.ylabel('Cost function, J')
    plt.title('Cost function convergence')
    convergence_plot.plot(convergence_plot_data[:, 0], convergence_plot_data[:, 1])


def hyp_value(x, theta):
    return sum(np.array(x) * np.array(theta))


def normalize_all(data):
    for i in range(1, len(data[0])):
        data[:, i] = scale_normalize(data[:, i])

    return data


def scale_normalize(lst):
    return lst / np.mean(lst)


def compare_theta(a, b):
    delta = 0.000001
    for i in range(len(a)):
        if abs(a[i] - b[i]) > delta:
            return False

    return True


def display_results(data, theta, hypothesis=None, labels=None):
    display_regression_coefficients(theta)
    if len(data[0]) == 3:
        display_regression_plot(data, hypothesis, labels)


def display_all_plots():
    plt.show()


def display_regression_plot(data, hypothesis=None, labels=None):
    x_data = data[:, 1]
    y_data = data[:, 2]
    plot_regression = plt.figure().add_subplot(111)
    plot_regression.scatter(x_data, y_data, s=10)

    if labels is not None:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.title(labels[2])

    if hypothesis is not None:
        plot_regression.plot(x_data, hypothesis, color='red')


def display_regression_coefficients(theta):
    print('\nCalculated regression coefficients:')
    [print('\tTheta[%d]: %f' % (i, theta[i])) for i in range(len(theta))]
