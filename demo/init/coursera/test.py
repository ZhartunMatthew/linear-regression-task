from . import common as com
from . import linear_regression as lr
import random

data = com.load_data('data/ex1data1.txt')
# com.display_data(data)


# for theta [0, 0] cost = 32.07
# for theta [-1, 2] cost = 54.24
def check_cost_function_default(theta, expectation):
    result = lr.compute_cost_function(data[:, 0], data[:, 1], theta)
    print('Result: ', result)
    print('Expected: ', expectation)


# print(lr.compute_cost_function(data[:, 0], data[:, 1], [0, 0]))
# print(lr.compute_cost_function(data[:, 0], data[:, 1], [-1, 2]))

# theta_0: -3.6303
# theta_1: 1.1664

x_data = data[:, 0]
y_data = data[:, 1]
theta = [random.uniform(-5, 5), random.uniform(-5, 5)]
alpha = 0.01
iterations = 10000

theta = lr.gradient_descent(data[:, 0], data[:, 1], theta, alpha, iterations)

com.display_data(data, [com.hyp_value(x, theta) for x in x_data])
