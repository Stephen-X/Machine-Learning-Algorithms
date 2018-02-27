# Decision Tree Learner with ID3 Algorithm
**Author:** Stephen Tse \<***@cmu.edu\>

A decision tree learner that uses mutural information to decide splits.

## Data Assumptions

1. For n columns in csv, column 1 ~ (n-1) are feature values, column n is label values.

2. All values are categorical.

## Usage

```bash
python decisionTree.py <train_input> <test_input> <max_depth> <train_out> <test_out> <metrics_out>
```
1. `<train_input>`: path to the training input .csv file
2. `<test_input>`: path to the test input .csv file
3. `<max_depth>`: maximum depth to which the tree should be built
4. `<train_out>`: path of output .labels file to which the predictions on the training data should
be written
5. `<test_out>`: path of output .labels file to which the predictions on the test data should be
written
6. `<metrics_out>`: path of the output .txt file to which metrics such as train and test error should
be written

### Example

Input:

```bash
python decisionTree.py politicians_train.csv politicians_test.csv 3 train_out.txt test_out.txt metrics.txt
```

Terminal Output:

```bash
[83 democrat / 66 republican]
| Superfund_right_to_sue = n: [55 democrat / 2 republican]
| | Export_south_africa = n: [0 democrat / 1 republican]
| | Export_south_africa = y: [55 democrat / 1 republican]
| | | Immigration = n: [46 democrat / 0 republican]
| | | Immigration = y: [9 democrat / 1 republican]
| Superfund_right_to_sue = y: [28 democrat / 64 republican]
| | Aid_to_nicaraguan_contras = n: [13 democrat / 58 republican]
| | | Export_south_africa = n: [0 democrat / 20 republican]
| | | Export_south_africa = y: [13 democrat / 38 republican]
| | Aid_to_nicaraguan_contras = y: [15 democrat / 6 republican]
| | | Mx_missile = n: [12 democrat / 0 republican]
| | | Mx_missile = y: [3 democrat / 6 republican]
```

3 files are generated in the root directory: train_out.txt, test_out.txt, and metrics.txt.

### Note

The sample dataset included are the 1984 United Stated Congressional Voting Records, classified as Republican or Democrat.


## Language & Dependencies

**Language:** Python 2.7 / 3.6

**Dependency Restriction:** only `numpy` (version 1.7.1), `scipy` and Python's built-in libraries (unfortunately `pandas` is not supported in grading VMs)
