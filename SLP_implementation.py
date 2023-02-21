import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#
# Perceptron implementation
#
class CustomPerceptron(object):

    def __init__(self, n_iterations=100, random_state=1, learning_rate=0.01):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.learning_rate = learning_rate
    '''
    Stochastic Gradient Descent

    1. Weights are updated based on each training examples.
    2. Learning of weights can continue for multiple iterations
    3. Learning rate needs to be defined
    '''

    def fit(self, X, y,check):
        if check == True :
            rgen = np.random.RandomState(self.random_state)
            self.coef_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
            self.errors_ = []
            Xi = np.array(X)
            yi = np.array(y)
            for _ in range(self.n_iterations):
                errors = 0
                for i in range(Xi.shape[0]):
                    predicted_value = self.predict(Xi[i])
                    self.coef_[1:] = self.coef_[1:] + self.learning_rate * (
                                yi[i] - predicted_value) * Xi[i]  # w1 = w0 + l( y-h(x)) * Xi
                    self.coef_[0] = self.coef_[0] + self.learning_rate * (
                                yi[i] - predicted_value) * 1  # b1 = b0 + l( y-h(x))
                    update = self.learning_rate * (yi[i] - predicted_value)
                    errors += int(update != 0.0)
                self.errors_.append(errors)
        else :
            rgen = np.random.RandomState(self.random_state)
            self.coef_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
            self.errors_ = []
            Xi = np.array(X)
            yi = np.array(y)
            for _ in range(self.n_iterations):
                errors = 0
                for i in range(Xi.shape[0]):
                    predicted_value = self.predict(Xi[i])
                    self.coef_[1:] = self.coef_[1:] + self.learning_rate * (
                            yi[i] - predicted_value) * Xi[i]  # w1 = w0 + l( y-h(x)) * Xi
                    self.coef_[0] = 0 * (self.coef_[0] + self.learning_rate * (
                            yi[i] - predicted_value) * 1 ) # b1 = b0 + l( y-h(x))
                    update = self.learning_rate * (yi[i] - predicted_value)
                    errors += int(update != 0.0)
                self.errors_.append(errors)

    '''
    Net Input is sum of weighted input signals
    '''

    def net_input(self, X):
        weighted_sum = np.dot(X, self.coef_[1:]) + self.coef_[0]
        return weighted_sum

    '''
    Activation function is fed the net input and the unit step function
    is executed to determine the output.
    '''

    def activation_function(self, X):
        weighted_sum = self.net_input(X)
        return np.where(weighted_sum >= 0.0, 1, -1)

    '''
    Prediction is made on the basis of output of activation function
    '''

    def predict(self, X):
        return self.activation_function(X)

    '''
    Model score is calculated based on comparison of
    expected value and predicted value
    '''

    def score(self, X, y):
        Xi = np.array(X)
        yi = np.array(y)
        misclassified_data_count = 0
        for i in range(Xi.shape[0]):
            output = self.predict(Xi[i])
            if (yi[i] != output):
                misclassified_data_count += 1
        total_data_count = len(X)
        self.score_ = (total_data_count - misclassified_data_count) / total_data_count
        return self.score_


