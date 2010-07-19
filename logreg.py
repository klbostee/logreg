"""
Very simple module for doing logistic regression.

Based on:
- http://blog.smellthedata.com/2009/06/python-logistic-regression-with-l2.html
- http://people.csail.mit.edu/jrennie/writing/lr.pdf
"""

from scipy.optimize.optimize import fmin_bfgs
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Data(object):
    """ Abstract base class for data objects. """

    def likelihood(self, betas, alpha=0):
        """ Likelihood of the data under the given settings of parameters. """
        
        # Data likelihood
        l = 0
        for i in range(self.n):
            l += log(sigmoid(self.y_train[i] * \
                             np.dot(betas, self.x_train[i,:])))
        
        # Prior likelihood
        for k in range(1, self.x_train.shape[1]):
            l -= (alpha / 2.0) * betas[k]**2
            
        return l


class SyntheticData(Data):
    
    def __init__(self, n, d):
        """ Create N instances of d dimensional input vectors and a 1D
        class label (-1 or 1). """
        
	self.n = n
	self.d = d

        means = .05 * np.random.randn(2, d)
        
        self.x_train = np.zeros((n, d))
        self.y_train = np.zeros(n)        
        for i in range(n):
            if np.random.random() > .5:
                y = 1
            else:
                y = 0
            self.x_train[i, :] = np.random.random(d) + means[y, :]
            self.y_train[i] = 2.0 * y - 1
        
        self.x_test = np.zeros((n, d))
        self.y_test = np.zeros(n)        
        for i in range(n):
            if np.random.randn() > .5:
                y = 1
            else:
                y = 0
            self.x_test[i, :] = np.random.random(d) + means[y, :]
            self.y_test[i] = 2.0 * y - 1


class TsvData(Data):

    def __init__(self, train_path, test_path):
        x_train, x_test = [], []
        y_train, y_test = [], []

        for path, x, y in ((train_path, x_train, y_train),
                           (test_path, x_test, y_test)):
            with open(path, "r") as file:
                for line in file:
                    numbers = [int(n) for n in line[:-1].split("\t")]
                    x.append(numbers[:-1])
                    y.append(numbers[-1])

        self.x_train = np.array(x_train)
        self.x_test = np.array(x_test)
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)

	self.n = y_train.shape[0]
	self.d = x_train.shape[1]
        

class Model(object):
    """ A simple logistic regression model with L2 regularization (zero-mean
    Gaussian priors on parameters). """

    def __init__(self, d):
	""" Create model for input data consisting of d dimensions. """

	# Initialize parameters to zero, for lack of a better choice.
        self.betas = np.zeros(d)

    def train(self, data, alpha=0):
        """ Define the gradient and hand it off to a scipy gradient-based
        optimizer. """

	# Set alpha so it can be referred to later if needed
	self.alpha = alpha

        # Define the derivative of the likelihood with respect to beta_k.
        # Need to multiply by -1 because we will be minimizing.
        dB_k = lambda B, k: (k > 0) * self.alpha * B[k] - np.sum([ \
                                    data.y_train[i] * data.x_train[i, k] * \
                                    sigmoid(-data.y_train[i] *\
                                            np.dot(B, data.x_train[i,:])) \
                                    for i in range(data.n)])
        
        # The full gradient is just an array of componentwise derivatives
        dB = lambda B: np.array([dB_k(B, k) \
                                 for k in range(data.x_train.shape[1])])
        
	# The function to be minimized
	func = lambda B: -data.likelihood(betas=B, alpha=self.alpha)

        # Optimize
        self.betas = fmin_bfgs(func, self.betas, fprime=dB)

    def predict(self, x):
        return sigmoid(np.dot(self.betas, x))

    def training_reconstruction(self, data):
        p_y1 = np.zeros(data.n)
        for i in range(data.n):
            p_y1[i] = self.predict(data.x_train[i,:])
        return p_y1

    def test_predictions(self, data):
        p_y1 = np.zeros(data.n)
        for i in range(data.n):
            p_y1[i] = self.predict(data.x_test[i,:])
        return p_y1
        
    def plot_training_reconstruction(self, data):
        plot(np.arange(data.n), .5 + .5 * data.y_train, 'bo')
        plot(np.arange(data.n), self.training_reconstruction(data), 'rx')
        ylim([-.1, 1.1])

    def plot_test_predictions(self, data):
        plot(np.arange(data.n), .5 + .5 * data.y_test, 'yo')
        plot(np.arange(data.n), self.test_predictions(data), 'rx')
        ylim([-.1, 1.1])


if __name__ == "__main__":
    from pylab import *
    import sys

    if len(sys.argv) >= 3:
        # Read data from given TSV files
        data = TsvData(sys.argv[1], sys.argv[2])
    else:
        # Create 20 dimensional data set with 25 points -- this will be
        # susceptible to overfitting.
        data = SyntheticData(25, 20)

    lr = Model(data.d)

    # Run for a variety of regularization strengths
    alphas = [0, .001, .01, .1]
    for j, a in enumerate(alphas):
        print "Initial likelihood:"
        print data.likelihood(lr.betas)
        
        # Train the model
        lr.train(data, alpha=a)
        
        # Display execution info
        print "Final betas:"
        print lr.betas
        print "Final likelihood:"
        print data.likelihood(lr.betas)
        
        # Plot the results
        subplot(len(alphas), 2, 2*j + 1)
        lr.plot_training_reconstruction(data)
        ylabel("Alpha=%s" % a)
        if j == 0:
            title("Training set reconstructions")
        
        subplot(len(alphas), 2, 2*j + 2)
        lr.plot_test_predictions(data)
        if j == 0:
            title("Test set predictions")

    show()
