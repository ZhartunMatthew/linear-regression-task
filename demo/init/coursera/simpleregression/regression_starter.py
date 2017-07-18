from demo.init.coursera.simpleregression import common as com
from demo.init.coursera.simpleregression import linear_regression as lr
import random


def compute_linear_regression():
    # loading exercise data
    data = com.load_data('data/ex1data1.txt')

    # getting separated x and y data for descent and cost function
    x_data = data[:, 0]
    y_data = data[:, 1]

    # testing compute_cost function
    print('For theta [0, 0] expected cost function result: 32.0')
    print('For theta [0, 0] computed cost function result: %.4f' % lr.compute_cost_function(x_data, y_data, [0, 0]))
    print('For theta [-1, 2] expected cost function result: 54.5')
    print('For theta [-1, 2] computed cost function result: %.4f' % lr.compute_cost_function(x_data, y_data, [-1, 2]))

    # generating random start theta (or coefficients on linear regression)
    theta = [random.uniform(0, 1), random.uniform(0, 1)]

    # descent speed
    # alpha should be between 0.005 and 0.2, or cost function will not converge
    # if alpha > 0.2 cost function will increase instead of decrease
    # if alpha < 0.005 cost function will never converge faster then in 10.000 iteration
    alpha = 0.01

    # if cost function not converged, descent will stop after all iterations
    iterations = 10000

    # computing gradient descent
    print('Started gradient descent with theta zero: %.4f, theta one: %.4f' % (theta[0], theta[1]))
    print('Expected theta zero: %.4f, theta one: %.4f (approximately)' % (-3.6303, 1.1664))
    theta = lr.gradient_descent(x_data, y_data, theta, alpha, iterations)
    print('Computed theta zero: %.4f, theta one: %.4f' % (theta[0], theta[1]))

    # displaying results
    com.display_regression(data, [com.hyp_value(x, theta) for x in x_data])

compute_linear_regression()
