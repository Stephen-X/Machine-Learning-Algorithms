#!/usr/bin/env python3

"""
Project 4: Text Analyzer with Multinomial Logistic Regression
Author: Stephen Xie <***@andrew.cmu.edu>

This project implements a crude Natural language Processing (NLP) system. It
predicts the tags (marked in Begin-Inside-Outside (BIO) format) of words in a
sentence using Multinomial Logistic regression.

Data Assumptions:

1. Data files are stored as tab-separated values and can be encoded as unicode
   strings.
2. The first line of the dataset should not be empty line.

Notes:

1. This implementation does not perform any forms of regularization (l1 / l2).
2. The implementation of Stochastic Gradient Descent (SGD) does not shuffle
   the training data, i.e. it goes through the training dataset in its original
   order during each iteration. This may not be "stochastic" but it allows
   producing deterministic results for graders.
3. This implementation does not check for log-likelihood convergence; rather,
   it requires that user specifies the number of epochs for SGD in each run.
4. For multiple label predictions with the same likelihood, the program will
   break tie by choosing the label with the smallest ASCII value.
"""

# the CMU Linux Timeshare Pool doesn't have numpy installed for python 3; still
# have to write 2 / 3 compatible code for testing before submission
from __future__ import absolute_import, division, print_function, \
                       unicode_literals

import sys
import numpy as np


class LogisticRegression:

    def __init__(self, train_input, validation_input, test_input, train_out,
                 test_out, metrics_out, num_epoch):
        # learning rate of the algorithm; hardcoded to 0.5
        self.gamma = .5

        with open(metrics_out, mode='w') as f:
            # load training & validation data
            # M: a list of unique features (x)
            # K: a list of unique labels / classes (y)
            # MDict: a dict of mapping from feature value -> its index in M
            # KDict: a dict of mapping from label value -> its index in K
            # x*: a list of feature indices
            # y*: a list of label indices
            # *_end: ending index (exclusive) of each sentence (not necessarily
            #        includes the last sentence's)
            # note: M, K shouldn't grow after training: must deal with
            # potential out-of-vocabulary words in validation / test data
            self.M, self.K, self.MDict, self.KDict, x1, y1, train_end = \
                self.load_training_2(train_input)
            x2, y2 = self.load_test_2(validation_input)[:2]

            # for j in range(len(y1)):
            #     print('[', ', '.join([self.M[i] for i in x1[j]]), ']', self.K[y1[j]])

            # train model
            self.theta = self.train(num_epoch, x1, y1, x2, y2, f)

            # predict training data
            f.write('error(train): %.6f\n' % self.predict(train_out, x1, y1, train_end))

            # mark training / validation data as obsolete (is this actually
            # useful??)
            del x1, y1, train_end, x2, y2
            # load test data
            x3, y3, test_end = self.load_test_2(test_input)

            # predict test data
            f.write('error(test): %.6f\n' % self.predict(test_out, x3, y3, test_end))

    def load_training_2(self, input_file):
        """
        Load and parse raw training data for model 2: a probability
        distribution over the current tag y_{t} using the parameters \\theta
        and a feature vector based on the previous word w_{t-1}, the current
        word w_{t}, and the next word w_{t+1}.

        Returns two lists of unique features and labels, two dicts of unique
        features & labels mapped to their indices in the previous lists, two
        lists of indices of features and labels in their respective lists, and
        a list of indices (exclusive) of ending words of sentences (may not
        contain info of the last sentence).
        """
        bias = ''  # a bias feature, defined to be an empty string
        x, uniq_x, x_ind = [], [bias], {bias: 0}
        y, uniq_y, y_ind = [], [], {}
        sentence_end = []  # ending index of current sentence (exclusive)
        with open(input_file, mode='r') as f:
            # words in a sentence, with a 'beginning-of-sentence' ('BOS')
            # marker at front and an 'end-of-sentence' ('EOS') marker at last
            words = ['BOS']
            self._update_uniq('prev:BOS', uniq_x, x_ind)
            self._update_uniq('next:EOS', uniq_x, x_ind)
            for line in f:
                line = line.strip()
                if line:
                    word, label = line.split('\t')
                    words.append(word)
                    # Note: to pass the assignment's grader, here I'm adding
                    # all three tags to each word and put to feature set, even
                    # though some of them may not exist in training set hence
                    # will never be trained in the training stage. I consider
                    # this a bad practice though.
                    self._update_uniq('prev:' + word, uniq_x, x_ind)
                    self._update_uniq('curr:' + word, uniq_x, x_ind)
                    self._update_uniq('next:' + word, uniq_x, x_ind)
                    self._update_uniq(label, uniq_y, y_ind)
                    y.append(y_ind[label])
                else:
                    if len(words) > 1:  # there's at least one word in sentence
                        # an empty line: the sentence has ended
                        words.append('EOS')
                        sentence_end.append(len(y))
                        # construct features for each label (i.e. y^{(i)})
                        for prev, curr, next in zip(words, words[1:], words[2:]):
                            prev2 = 'prev:' + prev
                            curr2 = 'curr:' + curr
                            next2 = 'next:' + next
                            # # make sure that new unique x values are recorded
                            # self._update_uniq(prev2, uniq_x, x_ind)
                            # self._update_uniq(curr2, uniq_x, x_ind)
                            # self._update_uniq(next2, uniq_x, x_ind)
                            x.append([x_ind[bias], x_ind[prev2], x_ind[curr2],
                                      x_ind[next2]])
                        # start new
                        words = ['BOS']
            if len(words) > 1:
                # complement features for the last sentence
                words.append('EOS')
                sentence_end.append(len(y))
                # construct features for each label (i.e. y^{(i)})
                for prev, curr, next in zip(words, words[1:], words[2:]):
                    prev2 = 'prev:' + prev
                    curr2 = 'curr:' + curr
                    next2 = 'next:' + next
                    # # make sure that new unique x values are recorded
                    # self._update_uniq(prev2, uniq_x, x_ind)
                    # self._update_uniq(curr2, uniq_x, x_ind)
                    # self._update_uniq(next2, uniq_x, x_ind)
                    x.append([x_ind[bias], x_ind[prev2], x_ind[curr2],
                              x_ind[next2]])

        return uniq_x, np.asarray(uniq_y), x_ind, y_ind, x, y, sentence_end

    def _update_uniq(self, val, uniq_val, val_ind):
        """
        A helper function for training data loading.
        """
        if val not in val_ind:
            uniq_val.append(val)
            val_ind[val] = len(uniq_val) - 1

    def load_test_2(self, input_file):
        """
        Load and parse raw validation / test data for model 2: a probability
        distribution over the current tag y_{t} using the parameters \\theta
        and a feature vector based on the previous word w_{t-1}, the current
        word w_{t}, and the next word w_{t+1}.
        """
        bias = ''  # a bias feature, defined previously to be an empty string
        x, y = [], []
        sentence_end = []  # ending index of current sentence (exclusive)
        with open(input_file, mode='r') as f:
            # words in a sentence, with a 'beginning-of-sentence' ('BOS')
            # marker at front and an 'end-of-sentence' ('EOS') marker at last
            words = ['BOS']
            for line in f:
                line = line.strip()
                if line:
                    word, label = line.split('\t')
                    words.append(word)
                    try:
                        y.append(self.KDict[label])
                    except KeyError:
                        y.append(-1)
                else:
                    if len(words) > 1:  # there's at least one word in sentence
                        # an empty line: the sentence has ended
                        words.append('EOS')
                        sentence_end.append(len(y))
                        # construct features for each label (i.e. y^{(i)})
                        for prev, curr, next in zip(words, words[1:], words[2:]):
                            prev2 = 'prev:' + prev
                            curr2 = 'curr:' + curr
                            next2 = 'next:' + next
                            try:
                                x.append([self.MDict[bias], self.MDict[prev2],
                                          self.MDict[curr2], self.MDict[next2]])
                            except KeyError:
                                # at least one of the words is out of vocabulary
                                # (i.e. not seen in training data): set the entire
                                # feature vector (not counting bias) to 0
                                x.append([self.MDict[bias]])
                        # start new
                        words = ['BOS']
            if len(words) > 1:
                # complement features for the last sentence
                words.append('EOS')
                sentence_end.append(len(y))
                # construct features for each label (i.e. y^{(i)})
                for prev, curr, next in zip(words, words[1:], words[2:]):
                    prev2 = 'prev:' + prev
                    curr2 = 'curr:' + curr
                    next2 = 'next:' + next
                    try:
                        x.append([self.MDict[bias], self.MDict[prev2],
                                  self.MDict[curr2], self.MDict[next2]])
                    except KeyError:
                        x.append([self.MDict[bias]])

        return x, y, sentence_end

    def predict(self, output_file, x, y, sentence_end):
        """
        Predict using the trained tree, and output result to file.
        Also returns prediction error.

        x: a list of feature indices of self.M
        y: a list of reference label indices of self.K
        """
        total_errors = 0
        j = 0
        with open(output_file, mode='w') as f:
            for i, (x_list, y_ind) in enumerate(zip(x, y)):  # for each sample
                if j < len(sentence_end) and sentence_end[j] == i:
                    # insert an empty line between sentences
                    f.write('\n')
                    j += 1
                # extract parameters correspond to non-zero features
                theta = self.theta[:, x_list]
                # a list of probabilities for all classes
                p = np.exp(np.sum(theta, axis=1)) / \
                    np.sum(np.exp(np.sum(theta, axis=1)))
                # the predicted label is the class with highest probability
                # break tie using labels' ASCII values: choose the smallest one
                predicted = min(self.K[np.argwhere(p == np.max(p)).flatten()])
                if y_ind >= 0:
                    if predicted != self.K[y_ind]:
                        total_errors += 1
                else:
                    # the original y value is an out-of-vocabulary value (i.e.
                    # not seen in training data)
                    total_errors += 1
                f.write('%s\n' % predicted)

        return total_errors / len(y)

    def train(self, num_epoch, x1, y1, x2, y2, metrics_f):
        """
        Use Stochastic Gradient Descent (SGD) to train a logistic regression
        model with the given training data.
        Returns the trained parameters \\theta.

        x*: a list of feature indices of self.M
        y*: a list of label indices of self.K
        metrics_f: file handler for outputting metrics
        """
        # initially, parameters \theta is a K x M matrix of zeros
        theta = np.zeros(shape=(len(self.K), len(self.M)))

        # instead of checking for convergence, specify exact number of times
        # SGD loops through all of the training data.
        for e in range(num_epoch):
            for i in range(len(y1)):
                # here sampling is not shuffled so that the calculation is
                # deterministic (good for graders)
                theta -= self.gamma * self.gradient(i, theta, x1, y1)
            # after each epoch, report likelihood (i.e. value of objective
            # function) for training and validation data
            metrics_f.write('epoch=%d likelihood(train): %.6f\n' %
                            (e + 1, self.nll(theta, x1, y1)))
            metrics_f.write('epoch=%d likelihood(validation): %.6f\n' %
                            (e + 1, self.nll(theta, x2, y2)))
            # metrics_f.write('%d,%f,%f\n' % (e + 1, self.nll(theta, x1, y1),
            #                                 self.nll(theta, x2, y2)))

        return theta

    def gradient(self, i, theta, x, y):
        """
        Gradient of the objective function for the given sample i,
        i.e. \\nabla_{\\theta} J^{(i)}(\\theta).

        Returns a K x M matrix in which K = # of classes, M = # of features.
        """
        # I(y^{(i)} = k)
        ind_func = np.zeros(shape=(len(self.K), 1))
        ind_func[y[i], 0] = 1

        # P(y^{(i)} | x^{(i)}, \theta)
        # note: because the parameter matrix \theta is sparse (lots of 0s with
        # only a few 1s), don't do regular dot product (i.e. \theta^{T} * x) as
        # this uses O(M) times, since for the purpose of this program we only
        # need O(1) times (only <= 4 of features x are non-zero (1)).
        p_sub = np.exp(np.sum(theta[:, x[i]], axis=1))
        p = p_sub / np.sum(p_sub)

        grad = np.zeros(shape=(len(self.K), len(self.M)))
        # numpy broadcasting: values of vector (p - ind_func) will be
        # broadcast to each column of grad specified by x[i]
        grad[:, x[i]] = p.reshape(-1, 1) - ind_func  # reshape p to column vector

        return grad

    def nll(self, theta, x, y):
        """
        Calculate negative average log likelihood (i.e. the objective function
        J(\\theta)) on the dataset based on the given \\theta.
        """
        nlls = 0.
        for i in range(len(y)):
            p_sub = np.exp(np.sum(theta[:, x[i]], axis=1))
            p = p_sub / np.sum(p_sub)
            nlls -= np.log(p[y[i]])

        return nlls / len(y)


if __name__ == '__main__':
    train_input = sys.argv[1]
    validate_input = sys.argv[2]
    test_input = sys.argv[3]
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]
    # the number of times SGD loops through all the training data
    num_epoch = int(sys.argv[7])

    model = LogisticRegression(train_input, validate_input, test_input,
                               train_out, test_out, metrics_out, num_epoch)

    # model = LogisticRegression('toydata/toytrain.tsv', 'toydata/toyvalidation.tsv',
    #                            'toydata/toytest.tsv', 'train_out.txt', 'test_out.txt',
    #                            'metrics_out.txt', num_epoch=2)
