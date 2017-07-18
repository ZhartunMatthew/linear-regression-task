from demo.init.coursera.multivariableregression import multivatiable_common as mc
from demo.init.coursera.multivariableregression import multivariable_linear_regression as mlr
import random


def compute_linear_regression():
    data = mc.load_data('data/ex1data1.txt')

    # getting separated x and y data for descent and cost function
    x_data = data[:, :-1]
    y_data = data[:,  -1]

    # generating random start theta (or coefficients on linear regression)
    theta = [random.uniform(0, 1) for i in range(len(x_data[0]))]

    # descent speed
    # decrease alpha if:
    #   - script threw an overflow warning
    #   - J-function on plot increase instead of decrease
    # increase alpha if:
    #   - J-function on plot decrease too slow or looks like linear function
    #   - convergence finished after first iteration
    alpha = pow(10, -4)

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
    mc.display_results(data, theta)

compute_linear_regression()
