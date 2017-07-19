from demo.init.coursera.logisticregression.simpleclassification import logistic_regression_common as lrc
from demo.init.coursera.logisticregression.simpleclassification import logistic_regression as lr
import numpy as np
from os.path import abspath
import random
data_path = abspath('coursera\logisticregression\data\ex2data1.txt')


def compute_linear_regression():
    data = lrc.load_data(data_path)
    lrc.display_logistic_data(data, labels=['First exam', 'Second exam', 'Exams'])
    theta = np.array([random.uniform(0, 1) for i in range(len(data[0]) - 1)])
    print(theta)


compute_linear_regression()
lrc.display_all_plots()
