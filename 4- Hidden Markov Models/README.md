# Part-Of-Speech Tagging with Hidden Markov Models
**Author:** Stephen Tse \<***@cmu.edu\>

This project implements a [part-of-speech tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging) system for natural language processing (NLP) using Hidden Markov Models (HMM). It first trains the model using supervised learning method for HMM (i.e. Maximum Likelihood Estimation), then makes prediction with the [forward-backward algorithm](https://en.wikipedia.org/wiki/Forward–backward_algorithm).


## Data Assumptions

1. The dataset contains one sentence per line that has already been preprocessed, cleaned and tokenized (consider the tags in the dataset the ground truth). It has the following format:
    ```
    <Word0>_<Tag0> <Word1>_<Tag1> ... <WordN>_<TagN>
    ```
    where every `<WordK>_<TagK>` unit token is separated by white space.


## Implementation Notes

1. This implementation does not have any forms of regularization (l1 / l2).

2. In `learnhmm.py`, I padded word and tag samples with filler values at the end to create big matrices for coding convenience (matrix operation is usually more concise and closer to mathematical expressions!). In hindsight, this is probably not a good idea, as I checked that the number of filler values is over 4 times the number of total words in the training data (in other words, the max. sentence length difference is quite large in the dataset), and it did cause the program to run a bit slower than the reference solution developed by teaching assistants that didn't use much vectorization.

3. Although the provided dataset is carefully sanitized, in practice one should address the common issue of floating point number underflow when implementing HMM. I changed the likelihood multiplication in the `predict` method of `forwardbackward.py` to log likelihood computation, as this is the most obvious place that could cause underflow. Nonetheless, I might still miss something as the overall algorithm implementation is not tested against real life scenarios.

4. A [pseudocount](https://en.wikipedia.org/wiki/Additive_smoothing) +1 is added to each count during MLE calculation of the parameters in `learnhmm.py`.

## Usage

First use `learnhmm.py` to train for HMM parameters, then use `forwardbackward.py` to make predictions.

```bash
python learnhmm.py <train input> <index to word> <index to tag> <hmmprior> <hmmemit> <hmmtrans>
```
1. `<train input>`: path to the training input .txt file
2. `<index to word>`: path to the .txt that specifies the dictionary mapping from words to indices. The tags are ordered by index, with the first word having index of 1, the second word having index of 2, etc.
3. `<index to tag>`: path to the .txt that specifies the dictionary mapping from tags to indices. The tags are ordered by index, with the first tag having index of 1, the second tag having index of 2, etc.
4. `<hmmprior>`: path to output .txt file to which the estimated prior (\\pi) will be written.
5. `<hmmemit>`: path to output .txt file to which the emission probabilities (A) will be written.
6. `<hmmtrans>`: path to output .txt file to which the transition probabilities (B) will be written.

```bash
python forwardbackward.py <test input> <index to word> <index to tag> <hmmprior> <hmmemit> <hmmtrans> <predicted file>
```
1. `<test input>`: path to the test input .txt file that will be evaluated by the forward backward algorithm
2. `<index to word>`: path to the .txt that specifies the dictionary mapping from words to indices. The tags are ordered by index, with the first word having index of 1, the second word having index of 2, etc. This is the same file as was described for `learnhmm.py`.
3. `<index to tag>`: path to the .txt that specifies the dictionary mapping from tags to indices. The tags are ordered by index, with the first tag having index of 1, the second tag having index of 2, etc. This is the same file as was described for `learnhmm.py`.
4. `<hmmprior>`: path to input .txt file which contains the estimated prior (\\pi).
5. `<hmmemit>`: path to input .txt file which contains the emission probabilities (A).
6. `<hmmtrans>`: path to input .txt file which contains transition probabilities (B).
7. `<predicted file>`: path to the output .txt file to which the predicted tags will be written.


### Example

```bash
python learnhmm.py fulldata/trainwords.txt fulldata/index_to_word.txt fulldata/index_to_tag.txt hmmprior.txt hmmemit.txt hmmtrans.txt

python forwardbackward.py fulldata/testwords.txt fulldata/index_to_word.txt fulldata/index_to_tag.txt hmmprior.txt hmmemit.txt hmmtrans.txt predicted.txt
```


## Language & Dependencies

**Language:** Python 3.6

**Dependency Requirements:** `numpy` (tested with 1.14.1), `scipy` (tested with 1.0.1)
