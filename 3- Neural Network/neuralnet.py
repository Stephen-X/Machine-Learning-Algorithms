#!/usr/bin/env python3

"""
Project 5: Handwritten Letters Recognition with Neural Network
Author: Stephen Xie <***@andrew.cmu.edu>

This project trains a neural network with a single hidden layer to recognize
handwritten letters.

Data Assumptions:

1. The dataset is a csv file in which the first column be the indices of the
   actual letter (0: 'a', 1: 'e', 2: 'g', 3: 'i', 4: 'l', 5: 'n', 6: 'o',
   7: 'r', 8: 't', 9: 'u'), and the remaining 128 columns be the pixel values
   that can be reshaped into a 16 x 8 OCR grayscale (even though the given
   data is binary (B/W), we assume arbitrary values in range [0, 1]) image for
   each row.
2. The dataset always has the same number of features (128) and same output
   label space ({0, 1, ..., 9}) for each sample.

Notes:

1. This implementation does not have any forms of regularization (l1 / l2).
2. The implementation of Stochastic Gradient Descent (SGD) does not shuffle
   the training data, i.e. it goes through the training dataset in its original
   order during each iteration. This may not be "stochastic" but it allows
   producing deterministic results for graders.
3. Learning rate is fixed throughout the training process (set by command flag)
   for the benefit of the graders.
4. This implementation follows module-based automatic differentiation (AD)
   convention for easier gradient debugging (faster to locate the layer in
   which the gradient goes wrong) and easier model mutation in the future.
"""

import sys
import numpy as np


class NeuralNet:

    # a map between unique indices and their corresponding labels that matches
    # the dataset convention
    label_dict = {0: 'a', 1: 'e', 2: 'g', 3: 'i', 4: 'l', 5: 'n', 6: 'o',
                  7: 'r', 8: 't', 9: 'u'}

    def __init__(self, train_input, validation_input, train_out,
                 validation_out, metrics_out, num_epoch, hidden_units,
                 init_flag, learning_rate):
        t_labels, t_data = self.load(train_input)
        v_labels, v_data = self.load(validation_input)
        # number of inputs values excluding bias per sample; here it's just the
        # number of pixels per picture
        self.input_size = 16 * 8

        with open(metrics_out, mode='w') as f:
            # train model
            self.alpha, self.beta = self.train(t_labels, t_data, v_labels,
                                               v_data, num_epoch, hidden_units,
                                               init_flag, learning_rate, f)
            # print('alpha =\n', self.alpha, '\nbeta =\n', self.beta)

            # make predictions
            f.write('error(train): %.2f\n' % self._predict(t_labels, t_data,
                                                           train_out))
            f.write('error(validation): %.2f\n' % self._predict(v_labels, v_data,
                                                                validation_out))

    def load(self, input_file):
        """
        Load data into memory.
        Returns: labels list, data table
        """
        raw_data = np.genfromtxt(input_file, delimiter=',', dtype=np.int,
                                 autostrip=True)
        # in numpy, arrays slices are just views on the original array
        data = raw_data[:, 1:]
        # convert labels to one-hot representation
        # trick used here: pick the correct 1-hot representation for each label
        # from a matrix of pre-generated 1-hot representations of unique labels
        labels = np.eye(len(self.label_dict))[raw_data[:, 0]]

        # TODO: visualize dataset (comment out before submission)
        # self._visualize_dataset(raw_data[:, 0], data)

        return labels, data

    def train(self, t_labels, t_data, v_labels, v_data, num_epoch,
              hidden_units, init_flag, learning_rate, metrics_f):
        """
        Train the neural net using Stochastic Gradient Descent (SGD).
        """
        # initialize parameters
        alpha, beta = self._init_params(init_flag, hidden_units)
        # instead of checking for convergence, specify exact number of times
        # SGD loops through all of the training data.
        for e in range(num_epoch):
            for y, x in zip(t_labels, t_data):
                # convert x, y to column vectors, and add a bias term to x
                x = self._add_bias_vec(x.reshape(-1, 1))
                y = y.reshape(-1, 1)

                # forward pass
                a, z, b, y_hat, loss = self.nn_forward(x, y, alpha, beta)

                # backward pass
                djd_alpha, djd_beta = self.nn_backward(x, y, y_hat, alpha, beta, a, z, b)

                # update parameters
                alpha -= learning_rate * djd_alpha
                beta -= learning_rate * djd_beta

            # report cross entropy with the new alpha & beta
            metrics_f.write('epoch=%d crossentropy(train): %.11f\n' %
                            (e + 1, self.mean_cross_entropy(t_labels, t_data,
                             alpha, beta)))
            metrics_f.write('epoch=%d crossentropy(validation): %.11f\n' %
                            (e + 1, self.mean_cross_entropy(v_labels, v_data,
                             alpha, beta)))
            # metrics_f.write(
            #     '%d,%.11f,%.11f\n' %
            #     (e + 1,
            #      self.mean_cross_entropy(t_labels, t_data, alpha, beta),
            #      self.mean_cross_entropy(v_labels, v_data, alpha, beta)))

        return alpha, beta

    def _predict(self, labels, data, output_file):
        """
        Perform a forward pass through the neural net with the trained
        parameters on the given dataset. Output predicted labels as well as
        error metrics.
        """
        total_errors = 0
        with open(output_file, mode='w') as f:
            for y, x in zip(labels, data):
                # convert x, y to column vectors, and add a bias term to x
                x = self._add_bias_vec(x.reshape(-1, 1))
                y = y.reshape(-1, 1)
                a, z, b, y_hat, loss = self.nn_forward(x, y, self.alpha, self.beta)
                predicted = np.argmax(y_hat)
                f.write('%d\n' % predicted)
                if predicted != np.nonzero(y)[0][0]:
                    total_errors += 1

        return total_errors / len(labels)

    def _init_params(self, init_flag, hidden_units):
        """
        Initialize parameters according to the init_flag: 1 for random
        initializations (a uniform distribution from -0.1 to 0.1) and 2 for
        zero initializations. Bias term will always be initialized to zero.

        hidden_units: number of hidden units specified by user
        """
        if init_flag == 2:
            alpha, beta = np.zeros((self.input_size + 1, hidden_units)), \
                          np.zeros((hidden_units + 1, len(self.label_dict)))
        else:
            alpha, beta = np.random.uniform(-0.1, 0.1,
                                            (self.input_size + 1, hidden_units)), \
                          np.random.uniform(-0.1, 0.1,
                                            (hidden_units + 1, len(self.label_dict)))
            alpha[0], beta[0] = 0, 0  # set bias terms to 0

        return alpha, beta

    def nn_forward(self, x, y, alpha, beta):
        """
        Perform a forward pass on the neural network.
        """
        a = self._linear_forward(x, alpha)
        z = self._sigmoid_forward(a)
        b = self._linear_forward(z, beta)
        loss, y_hat = self._output_layers_forward(b, y)

        return a, z, b, y_hat, loss

    def nn_backward(self, x, y, y_hat, alpha, beta, a, z, b):
        """
        Perform a backward pass on the neural network.
        """
        djdb = self._output_layers_backward(y_hat, y)
        # assert self._test_grad(djdb, b, (self._output_layers_forward, y))
        djd_beta, djdz = self._linear_backward(djdb, beta, z)
        # assert self._test_grad(djdz, z, (self._linear_forward, beta),
        #                        (self._output_layers_forward, y))
        djda = self._sigmoid_backward(djdz, z)
        # assert self._test_grad(djda, a, self._sigmoid_forward,
        #                        (self._linear_forward, beta),
        #                        (self._output_layers_forward, y))
        djd_alpha, djdx = self._linear_backward(djda, alpha, x)
        # assert self._test_grad(djdx, x, (self._linear_forward, alpha),
        #                        self._sigmoid_forward,
        #                        (self._linear_forward, beta),
        #                        (self._output_layers_forward, y))

        return djd_alpha, djd_beta

    def mean_cross_entropy(self, labels, data, alpha, beta):
        """
        Perform a forward pass through the neural net with the given dataset
        and parameters to calculate mean cross entropy.
        """
        total_loss = 0
        for y, x in zip(labels, data):
            # convert x, y to column vectors, and add a bias term to x
            x = self._add_bias_vec(x.reshape(-1, 1))
            y = y.reshape(-1, 1)
            a, z, b, y_hat, loss = self.nn_forward(x, y, alpha, beta)
            total_loss += loss

        return total_loss / len(labels)

    # sigmoid module

    def _sigmoid_forward(self, x):
        """
        Applies forward function of the sigmoid layer to each element and adds
        a bias term to the front.
        """
        return self._add_bias_vec(self._sigmoid(x))

    def _sigmoid_backward(self, deriv, z):
        """
        Applies forward function of the sigmoid layer to each element.
        """
        # remove derivative for bias term first
        return deriv[1:] * self._d_sigmoid(z[1:])

    # linear module

    def _linear_forward(self, x, param):
        """
        Applies forward function of the linear layer to each element.
        """
        return np.dot(param.T, x)

    def _linear_backward(self, deriv, param, z):
        """
        Returns dJd\\param, dJdz.
        """
        return np.dot(z, deriv.T), np.dot(param, deriv)

    # softmax + cross entropy module

    def _output_layers_forward(self, x, y):
        """
        Applies forward function of the last two layers combined (softmax +
        cross entropy).
        """
        y_hat = self._softmax(x)
        loss = self._cross_entropy(y_hat, y)
        return loss, y_hat

    def _output_layers_backward(self, y_hat, y):
        """
        Applies backward function of the last two layers combined (softmax +
        cross entropy).
        """
        return y_hat - y

    # helper functions

    def _add_bias_vec(self, col_vec):
        """
        Returns a new column vector with a bias term 1 added to the top.
        """
        return np.concatenate(([[1]], col_vec))

    def _sigmoid(self, x):
        """
        Performs sigmoid function calculation and returns the result.
        """
        return 1 / (1 + np.exp(-x))

    def _softmax(self, x):
        """
        Performs probability calculation for multinomial logistic regression
        and returns the result.
        """
        exp = np.exp(x)
        return exp / np.sum(exp)

    def _cross_entropy(self, y_hat, y):
        """
        Calculates and returns cross entropy loss.

        y: the actual y value from dataset (either 0 or 1)
        y_hat: the predicted probabilities corresponding to y
        """
        return -np.sum(y * np.log(y_hat))

    def _d_sigmoid(self, y):
        """
        Calculates and returns derivatives of a sigmoid layer.
        """
        return y * (1 - y)

    def _test_grad(self, grad, x, *func_bundles):
        """
        A helper utility used for testing gradients computed by backpropagation.
        Returns false if the provided gradient doesn't roughly equal to the
        result calculated by finite difference approximation.

        grad: the gradient to be tested
        x: the input vector for the target function
        func_bundles: a list of tuples of functions and their additional parameters
                    (the first function should not include the input vector),
                    disassembled from the original nested function; ordered by
                    how close the current function is to the input vector x.

        Usage Example:
        Say you have a nested function sigmoid(linear(x), theta), call _test_grad
        like this: _test_grad(grad, x, linear, (sigmoid, theta))
        """
        epsilon = 1e-5
        # the parameters accepted by the nearest function defines the vector length
        # of the final gradient
        grad_ref = np.zeros((len(x), 1))
        for i in range(len(x)):
            # only 1 input is modified with epsilon at a time
            # declare type explicitly, since x could be of integer type
            y1, y2 = np.array(x, copy=True, dtype=np.float), \
                     np.array(x, copy=True, dtype=np.float)
            y1[i], y2[i] = y1[i] + epsilon, y2[i] - epsilon
            for bundle in func_bundles:
                if callable(bundle):
                    # the current bundle contains only the function
                    y1 = bundle(y1)
                    y2 = bundle(y2)
                else:
                    # must pass in additional parameters as well
                    # asterisk before a tuple unwraps it into parameters
                    y1 = bundle[0](y1, *bundle[1:])
                    if isinstance(y1, tuple):
                        y1 = y1[0]
                    y2 = bundle[0](y2, *bundle[1:])
                    if isinstance(y2, tuple):
                        y2 = y2[0]

            grad_ref[i] = (y1 - y2) / (2 * epsilon)

        # print(func_bundles)
        # print('grad', grad)
        # print('grad_ref', grad_ref)

        # test whether all gradients are a close match
        return np.all(np.absolute(grad_ref - grad) < epsilon)

    def _visualize_dataset(self, labels, data):
        """
        A test utility used to visualize a random subset of the given labels
        and corresponding 16x8 B/W pixel data. Note that the provided data must
        have at least 36 records.
        """
        import matplotlib.pyplot as plt

        fig = plt.figure()
        inds = np.random.randint(len(labels), size=36)
        for i, ind in enumerate(inds):
            # setting xticks & yticks to empty removes the ticks but not the
            # entire axis, unlike ax.axis('off'); in this way, each image still
            # keeps a 1-pixel black border around it
            ax = fig.add_subplot(6, 6, i + 1, xticks=[], yticks=[])
            pixels = data[ind].reshape(16, 8)
            ax.imshow(pixels, cmap='gray')
            ax.text(-6, 15, self.label_dict[labels[ind]], color='black',
                    fontsize=12)


if __name__ == '__main__':
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    train_out = sys.argv[3]
    validation_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = int(sys.argv[8])
    learning_rate = float(sys.argv[9])

    model = NeuralNet(train_input, validation_input, train_out, validation_out,
                      metrics_out, num_epoch, hidden_units, init_flag,
                      learning_rate)

    # model = NeuralNet('largeTrain.csv', 'largeValidation.csv', 'train_out.txt',
    #                   'validation_out.txt', 'metrics_out.txt', 2, 4, 2, 0.1)
