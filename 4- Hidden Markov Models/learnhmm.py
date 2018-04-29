#!/usr/bin/env python3
"""
Project 7: Part-Of-Speech Tagging with Hidden Markov Models
Author: Stephen Xie <***@andrew.cmu.edu>

This program is part of the project implement a new part-of-speech tagging
system using Hidden Markov Models. It learns the hiddem Markov model parameters
required to apply the forward backward algorithm via supervised learning.

Data Assumptions:
1. The dataset contains one sentence per line that has already been preprocessed,
   cleaned and tokenized (consider the tags in the dataset the ground truth).
   It has the following format:
   <Word0>_<Tag0> <Word1>_<Tag1> ... <WordN>_<TagN>
   where every <WordK>_<TagK> unit token is separated by white space.

Notes:
1. A pseudocount +1 is added to each count during MLE calculation of the
   parameters.
"""

import sys
import numpy as np
from itertools import zip_longest
from scipy.sparse import coo_matrix
from collections import Counter, OrderedDict

# for debugging: print the full numpy array
# np.set_printoptions(threshold=np.inf)


class HMMParams:

    def __init__(self, train_input, index_to_word, index_to_tag, hmmprior,
                 hmmemit, hmmtrans):
        # Read data to memory
        words, tags, words_list, tags_list = self.load(train_input,
                                                       index_to_word,
                                                       index_to_tag)

        # Compute and output parameters for HMM
        pi = self.prior_probs(tags, tags_list)  # initialization probabilities (estimated prior)
        self.output(hmmprior, pi)
        a = self.transition_probs(tags, tags_list)  # transition probabilities
        self.output(hmmtrans, a)
        b = self.emission_probs(tags, tags_list, words, words_list)  # emission probabilities
        self.output(hmmemit, b)

    def load(self, data_input, index_to_word, index_to_tag):
        """
        Load data into memory.
        Note that this function assumes that index_to_word and index_to_tag
        files contain all the unique words and tags in data_input. It'll skip
        any sentences in data_input that contains words or tags not in
        index_to_word or index_to_tags.

        Returns: the word indices collection as a sparse matrix where empty
                 cells to the right will be padded -1, the corresponding tag
                 indices matrix with -1 padding, and ordered lists of unique
                 words and tags.
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
            total_words = 0  # TODO: debug
            for line in f:
                line = line.strip()
                if line:
                    sentence_words, sentence_tags = [], []
                    word_tags = line.split()
                    total_words += len(word_tags)  # TODO: debug
                    try:
                        for pair in word_tags:
                            split = pair.split('_')
                            sentence_words.append(words_dict[split[0]])
                            sentence_tags.append(tags_dict[split[1]])
                        words.append(sentence_words)
                        tags.append(sentence_tags)
                    except KeyError:
                        # At least one of the words / tags is not in word / tag
                        # dict; skip the entire sentence
                        print('Error at line "%s": word or tag not in dict.' % line)
                        continue

        # Convert words and tags to matrices
        # The trick used to convert list of lists of varying lengths to a
        # numpy matrix: use itertools.zip_longest to zip all rows as separate
        # lists (items in the same column will be zipped together) with missing
        # values filled with -1; after converting this iterator to numpy matrix,
        # the result will be a transpose of the objective matrix.
        # Ref: https://stackoverflow.com/a/38619333
        words_mtx = np.array(list(zip_longest(*words, fillvalue=-1))).T
        tags_mtx = np.array(list(zip_longest(*tags, fillvalue=-1))).T
        # print('words_mtx:\n', words_mtx, end='\n\n')
        # print('tags_mtx:\n', tags_mtx, end='\n\n')
        print('%% Fillers: %.5f%%' % (np.sum(words_mtx == -1) / total_words * 100))  # TODO: debug

        # Convert to scipy sparse matrix to save space (since the matrices
        # could have lots of -1s)
        # Note: coo_matrix is a scipy sparse matrix in COOrdinate format;
        # it allows very fast convertion (linear time) to csr_matrix
        # (row-slicing optimized sparse matrix) or csc_matrix (column-slicing
        # optimized sparse matrix) later depending on computation requirement
        return coo_matrix(words_mtx), coo_matrix(tags_mtx), \
            list(words_dict.keys()), list(tags_dict.keys())

    def prior_probs(self, tags, tags_list):
        """
        This computes the initialization probabilities (estimated prior) of
        each state being associated with the first word of a sentence. Here we
        model the probabilities using a multinomial distribution.

        Returns an array of initialization probabilities corresponding to each
        state type, i.e. P(y_1 = j), j \\in {unique_tags}.
        """
        # First convert to column-slicing optimized sparse matrix, then get
        # the first column of tags and convert to normal numpy array
        first_tags = tags.tocsc()[:, 0].toarray()
        # Count the number of times each unique tag is associated with the
        # first word of a sentence
        tags_set, cnts = np.unique(first_tags, return_counts=True)
        # tags_set, cnts = self.count_unique(tags[:, 0])
        # expand counts to include all unique tags
        counts = np.zeros(len(tags_list), dtype=int)
        counts[tags_set] = cnts
        counts += 1  # Add 1 to each count to make a pseudocount

        # Note that np.unique will sort the unique value output, hence
        # counts will be in order (remember tags_set is a set of consecutive
        # indices of unique tags starting from 0)
        likelihoods = counts / np.sum(counts)

        return likelihoods

    def transition_probs(self, tags, tags_list):
        """
        This computes the transition probabilities between adjacent states.
        Here we model the probabilities using a multinomial distribution.

        Returns a matrix of transition probabilities, i.e. P(y_t = k | y_{t-1} = j),
        k, j \\in {unique_tags}.
        """
        # Convert tags back to normal (dense) numpy matrix, and add 1 to all
        # tag indices to eliminate -1 (useful for the computation trick later)
        tags = tags.toarray() + 1
        # A matrix C in which C_j^k records the number of times state s_j is
        # followed by state s_k; index 0 for the original padded -1s
        count_mtx = np.zeros((len(tags_list) + 1, len(tags_list) + 1), dtype=int)
        # Do the counting for all sentences in tags
        # Note: this trick works fast only if there aren't many -1s in tags,
        # otherwise it might be best to just traverse through each sentence and
        # count manually; in hindsight, this is probably not a good idea, as
        # number of filler values is over 4 times the number of total words in
        # the training data (see console output).
        # Note 2: the following code:
        # count_mtx[tags[:, :-1], tags[:, 1:]] += 1
        # won't work if there're duplicate indices; will only add once.
        # Use numpy.add.at instead:
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.at.html
        np.add.at(count_mtx, [tags[:, :-1], tags[:, 1:]], 1)
        # Remove counts for -1s, and add 1 to each count to make a pseudocount
        count_mtx = count_mtx[1:, 1:] + 1

        # sum of each row, reshaped as column vector
        row_sum = np.sum(count_mtx, axis=1)[:, None]
        likelihoods = count_mtx / row_sum

        return likelihoods

    def emission_probs(self, tags, tags_list, words, words_list):
        """
        This computes the emission probabilities of each state being
        associated with a certain word. Here we model the probabilities using a
        multinomial distribution.

        Returns a matrix of emission probabilities, i.e. P(x_t = k | y_t = j),
        k \\in {unique_words}, j \\in {unique_tags}.
        """
        # Same trick as self.transition_probs()
        tags = tags.toarray() + 1
        words = words.toarray() + 1
        # A matrix C in which C_j^k records the number of times state s_j is
        # associated with word w_k; index 0 for the original padded -1s
        count_mtx = np.zeros((len(tags_list) + 1, len(words_list) + 1), dtype=int)
        np.add.at(count_mtx, [tags, words], 1)
        count_mtx = count_mtx[1:, 1:] + 1

        row_sum = np.sum(count_mtx, axis=1)[:, None]
        likelihoods = count_mtx / row_sum

        return likelihoods

    def output(self, output_file, params):
        """
        Output given parameters to the specified location.
        """
        with open(output_file, mode='w') as f:
            for line in params:
                if isinstance(line, np.ndarray):
                    f.write('%s\n' % ' '.join(['%.20e' % d for d in line]))
                else:
                    # just a single number
                    f.write('%.20e\n' % line)

    def count_unique(self, data):
        """
        A helper function for counting unique elements in data, as numpy 1.7.1
        doesn't support np.unique(data, return_counts=True).
        """
        counter = Counter(data)
        return np.array(list(counter.keys())), np.array(list(counter.values()))


if __name__ == '__main__':
    train_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]

    model = HMMParams(train_input, index_to_word, index_to_tag, hmmprior,
                      hmmemit, hmmtrans)

    # model = HMMParams('toydata/toytrain.txt', 'toydata/toy_index_to_word.txt',
    #                   'toydata/toy_index_to_tag.txt', 'hmmprior.txt',
    #                   'hmmemit.txt', 'hmmtrans.txt')
    # model = HMMParams('fulldata/trainwords.txt', 'fulldata/index_to_word.txt',
    #                   'fulldata/index_to_tag.txt', 'hmmprior.txt',
    #                   'hmmemit.txt', 'hmmtrans.txt')
