from demo.init.coursera.multivariableregression import multivatiable_common as mc
from demo.init.coursera.multivariableregression import multivariable_linear_regression as mlr
import random


def compute_linear_regression():
    data = mc.load_data('data/ex1data1.txt')

    # getting separated x and y data for descent and cost function
    # and normalization x-data
    x_data = mc.normalize_all(data[:, :-1])
    y_data = data[:,  -1]

    # generating random start theta (or coefficients on linear regression)
    theta = [random.uniform(0, 1) for i in range(len(x_data[0]))]

    # descent speed
    # decrease alpha if:
    #   - script threw an overflow warning
    #   - J-function on plot increase instead of decrease
    # increase alpha if:
    #   - J-function on plot decrease too slow or looks like linear function
    #   - descent finished after first iteration or very fast
    #   - descent finished all iterations, but regression looks even not close to true
    # for ex1data1.txt alpha = 0.01 is perfect choice (if data not normalize)
    # for ex1data2.txt alpha = 0.01 if too big, 10**(-10) - 10**(-7) is better choice (is not normalize)
    # if data normalized alpha = 0.1 is perfect choice for both examples
    alpha = pow(10, -1)

    # if cost function not converged, descent will stop after all iterations
    iterations = 10000

    # computing gradient descent
    print('Gradient params.')
    print('\tRandomly generated theta vector: ', end='')
    [print('[%d - %.4f]' % (i, theta[i]), end=' ') for i in range(len(theta))]
    print('\n\tAlpha:                           %f' % alpha)
    print('\tMax iteration count:             %d' % iterations)
    print('\nGradient descent started...')

    # computing gradient descent
    theta = mlr.gradient_descent(x_data, y_data, theta, alpha, iterations)

    # displaying results
    # if regression with one variable plot will be displayed
    prediction = [mc.hyp_value(x, theta) for x in x_data]
    labels = ['X-axis', 'Y-axis', 'Title']
    mc.display_results(data, theta, prediction, labels)

compute_linear_regression()
