#!/usr/bin/env python3
"""
Project 7: Part-Of-Speech Tagging with Hidden Markov Models
Author: Stephen Xie <***@andrew.cmu.edu>

This program is part of the project implement a new part-of-speech tagging
system using Hidden Markov Models. It takes the parameters learned from
learnhmm.py and makes prediction on the test data using the forward-backward
algorithm.

Data Assumptions:
1. The dataset contains one sentence per line that has already been preprocessed,
   cleaned and tokenized (consider the tags in the dataset the ground truth).
   It has the following format:
   <Word0>_<Tag0> <Word1>_<Tag1> ... <WordN>_<TagN>
   where every <WordK>_<TagK> unit token is separated by white space.
"""

import sys
import numpy as np
from collections import OrderedDict

# for debugging: print the full numpy array
# np.set_printoptions(threshold=np.inf)


class HMMForwardBackward:

    def __init__(self, test_input, index_to_word, index_to_tag, hmmprior,
                 hmmemit, hmmtrans, predicted_file):
        # Read data to memory
        words, tags, words_list, tags_list, \
            pi, b, a = self.load(test_input, index_to_word, index_to_tag,
                                 hmmprior, hmmemit, hmmtrans)

        with open(predicted_file, mode='w') as f:
            total_words = 0
            total_mismatches = 0
            total_likelihood = 0.
            for sentence, sentence_tags in zip(words, tags):
                # total weight of path prefixes for each possible tag of words
                alpha = self.forward(sentence, tags_list, pi, a, b)
                # total weight of path suffixes for each possible tag of words
                beta = self.backward(sentence, tags_list, a, b)
                # print('alpha =\n', alpha)
                # print('beta =\n', beta)

                total_likelihood += self.seq_likelihood(alpha)

                # self.predict(alpha, beta, sentence, words_list, sentence_tags,
                #              tags_list, b, f)
                # output prediction for current sentence, and record mismatches
                total_words += len(sentence)
                total_mismatches += self.predict(alpha, beta, sentence,
                                                 words_list, sentence_tags,
                                                 tags_list, b, f)
            print('Test Error: %.5f%%' % (total_mismatches / total_words * 100))
            print('Average Sequence Likelihood: %f' % (total_likelihood / len(words)))

    def load(self, data_input, index_to_word, index_to_tag, hmmprior, hmmemit,
             hmmtrans):
        """
        Load data into memory.
        """
        words_dict = OrderedDict()  # unique words and their corresponding indices
        tags_dict = OrderedDict()  # unique tags and their corresponding indices
        words = []  # a list of lists, each sub-list contains the words as indices in words_dict of a sentence
        tags = []  # a list of lists of tags as indices in tags_dict corresponding to words

        # load indices
        with open(index_to_word, mode='r') as f:
            curr_index = 0
            for line in f:
                line = line.strip()
                if line:
                    words_dict[line] = curr_index
                    curr_index += 1
        with open(index_to_tag, mode='r') as f:
            curr_index = 0
            for line in f:
                line = line.strip()
                if line:
                    tags_dict[line] = curr_index
                    curr_index += 1

        # load input data
        with open(data_input, mode='r') as f:
            for line in f:
                line = line.strip()
                if line:
                    sentence_words, sentence_tags = [], []
                    word_tags = line.split()
                    try:
                        for pair in word_tags:
                            split = pair.split('_')
                            sentence_words.append(words_dict[split[0]])
                            sentence_tags.append(tags_dict[split[1]])
                        # Note: np.array() won't covert list of lists with varying
                        # lengths into (asymmetric) numpy matrix; do it manually
                        words.append(np.array(sentence_words))
                        tags.append(np.array(sentence_tags))
                    except KeyError:
                        # At least one of the words / tags is not in word / tag
                        # dict; skip the entire sentence
                        print('Error at line "%s": word or tag not in dict.' % line)
                        continue

        # load parameters trained by learnhmm.py
        hmm_prior = np.genfromtxt(hmmprior, delimiter=' ', dtype=np.float,
                                  autostrip=True)
        hmm_emission = np.genfromtxt(hmmemit, delimiter=' ', dtype=np.float,
                                     autostrip=True)
        hmm_transition = np.genfromtxt(hmmtrans, delimiter=' ', dtype=np.float,
                                       autostrip=True)

        return words, tags, \
               np.array(list(words_dict.keys())), \
               np.array(list(tags_dict.keys())), \
               hmm_prior, hmm_emission, hmm_transition

    def forward(self, sentence, tags_list, pi, a, b):
        """
        Forward pass that computes total weight of path prefixes for each
        possible tag of words in a given sentences.
        """
        alpha = np.zeros((len(sentence), len(tags_list)))
        for t, word in enumerate(sentence):
            if t == 0:
                # initialize for the first word in a sentence
                alpha[t] = pi * b[:, word]
            else:
                alpha[t] = b[:, word] * np.dot(alpha[t - 1], a)

        return alpha

    def backward(self, sentence, tags_list, a, b):
        """
        Backward pass that computes total weight of path suffixes for each
        possible tag of words in a given sentences.
        """
        beta = np.zeros((len(sentence), len(tags_list)))
        # initialize for the last word in a sentence
        beta[len(sentence) - 1] = 1  # all states could be ending states
        for t, word in self.rev_enumerate(sentence):
            if t > 0:
                beta[t - 1] = np.dot(b[:, word] * beta[t], a.T)

        return beta

    def predict(self, alpha, beta, sentence, words_list, sentence_tags,
                tags_list, emission, output_f):
        """
        Computes probabilities based on total path prefix & suffix weights from
        forward & backward pass, then output prediction for the current
        sentence using minimum Bayes risk predictor.

        Returns: number of mismatches in the current sentence
        """
        # approximate probabilities of each tag for each word
        # likelihoods = alpha * beta
        # likelihoods = alpha * beta / np.sum(alpha[-1])
        # Note: here we compute log likelihoods to avoid potential underflow
        likelihoods = np.log(alpha) + np.log(beta)
        # Get the index of tag with the largest likelihood for each word
        predicted = np.argmax(likelihoods, axis=1)
        # output result
        output_f.write('%s\n' % ' '.join(['%s_%s' % (w, t)
                            for (w, t) in zip(words_list[sentence],
                                              tags_list[predicted])]))
        return np.sum(predicted != sentence_tags)

    def seq_likelihood(self, alpha):
        """
        Computes the log likelihood of an input sequence.
        """
        return np.log(np.sum(alpha[-1]))

    def rev_enumerate(self, arr):
        """
        A helper generator for performing reversed enumeration on a given array
        while providing original indices without creating a copy of the array,
        i.e. this does reversed(list(enumerate(arr))) without creating a list
        for enumerate().
        """
        for i in reversed(range(len(arr))):
            yield i, arr[i]


if __name__ == '__main__':
    test_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    predicted_file = sys.argv[7]

    model = HMMForwardBackward(test_input, index_to_word, index_to_tag,
                               hmmprior, hmmemit, hmmtrans, predicted_file)

    # model = HMMForwardBackward('toydata/toytest.txt',
    #                            'toydata/toy_index_to_word.txt',
    #                            'toydata/toy_index_to_tag.txt',
    #                            'hmmprior.txt', 'hmmemit.txt', 'hmmtrans.txt',
    #                            'predictedtest.txt')
    # model = HMMForwardBackward('fulldata/testwords.txt',
    #                            'fulldata/index_to_word.txt',
    #                            'fulldata/index_to_tag.txt',
    #                            'hmmprior.txt', 'hmmemit.txt', 'hmmtrans.txt',
    #                            'predictedtest.txt')
