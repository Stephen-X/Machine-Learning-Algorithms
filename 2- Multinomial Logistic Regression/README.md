# Text Analyzer with Multinomial Logistic Regression
**Author:** Stephen Tse \<***@cmu.edu\>

This project implements a crude Natural language Processing (NLP) system. It predicts the tags (marked in Begin-Inside-Outside (BIO) format) of words in a sentence using Multinomial Logistic regression. The model for this NLP system is a probability distribution over the current tag y_{t} using the parameters \\theta and a feature vector based on the previous word w_{t-1}, the current word w_{t}, and the next word w_{t+1}.


## Data Assumptions

1. Data files are stored as tab-separated values and can be encoded as unicode
   strings.

2. The first line of the dataset should not be empty line.


## Implementation Notes

1. This implementation does not perform any forms of regularization (l1 / l2).

2. The implementation of Stochastic Gradient Descent (SGD) does not shuffle the training data, i.e. it goes through the training dataset in its original order during each iteration. This may not be "stochastic" but it allows producing deterministic results for graders.

3. This implementation does not check for log-likelihood convergence; rather, it requires that user specifies the number of epochs for SGD in each run.

4. For multiple label predictions with the same likelihood, the program will break tie by choosing the label with the smallest ASCII value.


## Usage

```bash
python tagger.py <train input> <validation input> <test input> <train out> <test out> <metrics out> <num epoch>
```
1. `<train input>`: path to the training input .tsv file
2. `<validation input>`: path to the validation input .tsv file
3. `<test input>`: path to the test input .tsv file
4. `<train out>`: path to output .labels file to which the prediction on the training data should be written
5. `<test out>`: path to output .labels file to which the prediction on the test data should be written
6. `<metrics out>`: path of the output .txt file to which metrics such as train and test error should be written
7. `<num epoch>`: integer specifying the number of times SGD loops through all of the training data (e.g., if `<num epoch>` equals 5, then each training example will be used in SGD 5 times).


### Example

Input:

```bash
python tagger.py largedata/train.tsv largedata/validation.tsv largedata/test.tsv train_out.labels test_out.labels metrics_out.txt 10
```

3 files are generated in the root directory: train_out.labels, test_out.labels, and metrics_out.txt. The metrics output will report the objective function values (i.e. negative average log likelihood) for both the training dataset and the validation dataset in each epoch as well as final model prediction error rates for both the training dataset and the test dataset.


### A Note on Sample Dataset

The sample dataset included are from the Airline Travel Information System (ATIS) dataset. Each data set consists of attributes
(words) and labels (airline flight information tags in Begin-Inside-Outside (BIO) format). The attributes and tags are separated into sequences (i.e. phrases) with a blank line between each sequence. See [here](http://deeplearning.net/tutorial/rnnslu.html) for more information.


## Language & Dependencies

**Language:** Python 2.7 / 3.6

**Dependency Requirements:** `numpy` (version 1.7.1)
