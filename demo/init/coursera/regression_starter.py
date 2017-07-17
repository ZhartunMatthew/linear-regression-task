from . import common as com
from . import linear_regression as lr
import random

# loading exercise data
data = com.load_data('data/ex1data1.txt')

# testing compute_cost function
print('For theta [0, 0] cost function result smth like 32.0')
print('For computed result for theta [0, 0]: %.4f' % lr.compute_cost_function(data[:, 0], data[:, 1], [0, 0]))
print('For theta [-1, 2] cost function result smth like 54.5')
print('For computed result for theta [-1, 2]: %.4f' % lr.compute_cost_function(data[:, 0], data[:, 1], [-1, 2]))

# getting separated x and y data for descent
x_data = data[:, 0]
y_data = data[:, 1]

# generating random start theta (or coefficients on linear regression)
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

# descent speed
# alpha should be between 0.005 and 0.2, or cost function will not converge
# if alpha > 0.2 cost function will increase instead of decrease
# if alpha < 0.005 cost function will never converge faster then in 10.000 iteration
alpha = 0.01

# if cost function not converged, descent will stop after all iterations
iterations = 10000

# computing gradient descent
print('Started gradient descent with theta zero: %.4f, theta one: %.4f' % (theta[0], theta[1]))
theta = lr.gradient_descent(data[:, 0], data[:, 1], theta, alpha, iterations)

# displaying results
com.display_regression(data, [com.hyp_value(x, theta) for x in x_data])
