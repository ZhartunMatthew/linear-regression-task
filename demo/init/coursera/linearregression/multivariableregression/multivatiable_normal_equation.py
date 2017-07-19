import numpy.linalg as lin


def compute_normal_equation(x_data, y_data):
    # xT - transposed x-matrix
    x_transpose = x_data.transpose()
    # X' * X
    x_mul = x_transpose @ x_data
    # (X' * X) ** (-1)
    x_mul_inv = lin.inv(x_mul)
    # ((X' * X) ** (-1))' * X'
    x_mul_on_transpose = x_mul_inv @ x_transpose
    # ((X' * X) ** (-1))' * X' * y
    return x_mul_on_transpose @ y_data

