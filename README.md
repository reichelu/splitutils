# <a name="contents">Machine learning data partition tools</a>

- [Contents](#contents)
    - [Author](#author)
    - [Purpose](#purpose)
    - [Installation](#installation)
    - [Synopsis](#synopsis)
        - [optimize_traintest_split()](#otts)
        - [optimize_traindevtest_split()](#otdts)
        - [binning()](#binning)
    - [Usage](#usage)
        - [Example 1: Split dummy data into training and test partitions](#example1)
        - [Example 2: split dummy data into training, development, and test partitions](#example2)
        - [Example 3: split dummy data into training, development, and test partitions, the target and several stratification variables being numeric](#example3)
    - [Algorithm](#algorithm)
    - [How to interpret the returned info dict](#interpretation)

## <a name="author">Author</a>

Uwe Reichel, audEERING GmbH, Gilching, Germany

## <a name="purpose">Purpose</a>

* machine learning data splitting tool that allows for:
  * group-disjunct splits (e.g. different speakers in train, dev, and test partition)
  * stratification on multiple target and grouping variables (e.g. emotion, gender, language)

## <a name="installation">Installation</a>

### From PyPI

* set up a virtual environment `venv_splitutils`, activate it, and install `splitutils`. For Linux this works e.g. as follows:

```bash
$ virtualenv --python="/usr/bin/python3" venv_splitutils
$ source venv_splitutils/bin/activate
(venv_splitutils) $ pip install splitutils
```

### From GitHub

* project URL: https://github.com/reichelu/splitutils
* set up a virtual environment venv_splitutils, activate it, and install requirements. For Linux this works e.g. as follows:

```bash
$ git clone git@github.com:reichelu/spliutils.git
$ cd splitutils/
$ virtualenv --python="/usr/bin/python3" venv_splitutils
$ source venv_splitutils/bin/activate
$ (venv_splitutils) $ pip install -r requirements.txt
```

## <a name="synopsis">Synopsis</a>

### <a name="otts">optimize_traintest_split()</a>

```python
def optimize_traintest_split(X, y, split_on, stratify_on, weight=None,
                             test_size=.1, k=30, seed=42):

    ''' optimize group-disjunct split into training and test set which is guided by:
    - disjunct split of values in SPLIT_ON
    - stratification by all keys in STRATIFY_ON (targets and groupings)
    - test set proportion in X should be close to test_size (which is the test
      proportion in set(split_on))

    Parameters:
    X: (np.array or pd.DataFrame) of features
    y: (np.array) of targets of length N
      if type(y[0]) in ["str", "int"]: y is assumed to be categorical, so that it is
      additionally tested that all partitions cover all classes.
      Else y is assumed to be numeric and no coverage test is done.
    split_on: (np.array) list of length N with grouping variable (e.g. speaker IDs),
      on which the group-disjunct split is to be performed. Must be categorical.
    stratify_on: (dict) Dict-keys are variable names (targets and/or further groupings)
      the split should be stratified on (groupings could e.g. be sex, age class, etc).
      Dict-Values are np.array-s of length N that contain the variable values. All
      variables must be categorical.
    weight: (dict) weight for each variable in stratify_on. Defines their amount of
      contribution to the optimization score. Uniform weighting by default. Additional
      key: "size_diff" defines how test size diff should be weighted.
    test_size: (float) test proportion in set(split_on), e.g. 10% of speakers to be
      held-out
    k: (int) number of different splits to be tried out
    seed: (int) random seed

    Returns:
    train_i: (np.array) train set indices in X
    test_i: (np.array) test set indices in X
    info: (dict) detail information about reference and achieved prob distributions
        "size_testset_in_spliton": intended test_size
        "size_testset_in_X": optimized test proportion in X
        "p_ref_{c}": reference class distribution calculated from stratify_on[c]
        "p_test_{c}": test set class distribution calculated from stratify_on[c][test_i]
    '''
```

### <a name="otdts">optimize_traindevtest_split()</a>


```python
def optimize_traindevtest_split(X, y, split_on, stratify_on, weight=None,
                                dev_size=.1, test_size=.1, k=30, seed=42):

    ''' optimize group-disjunct split into training, dev, and test set, which is
    guided by:
    - disjunct split of values in SPLIT_ON
    - stratification by all keys in STRATIFY_ON (targets and groupings)
    - test set proportion in X should be close to test_size (which is the test
      proportion in set(split_on))

    Parameters:
    X: (np.array or pd.DataFrame) of features
    y: (np.array) of targets of length N
      if type(y[0]) in ["str", "int"]: y is assumed to be categorical, so
         that it is additionally tested that all partitions cover all classes.
         Else y is assumed to be numeric and no coverage test is done.
    split_on: (np.array) list of length N with grouping variable (e.g. speaker IDs),
      on which the group-disjunct split is to be performed. Must be categorical.
    stratify_on: (dict) Dict-keys are variable names (targets and/or further groupings)
      the split should be stratified on (groupings could e.g. be sex, age class, etc).
      Dict-Values are np.array-s of length N that contain the variable values. All
      variables must be categorical.
    weight: (dict) weight for each variable in stratify_on. Defines their amount of
      contribution to the optimization score. Uniform weighting by default. Additional
      key: "size_diff" defines how the corresponding size differences should be weighted.
    dev_size: (float) proportion in set(split_on) for dev set, e.g. 10% of speakers
      to be held-out
    test_size: (float) test proportion in set(split_on) for test set
    k: (int) number of different splits to be tried out
    seed: (int) random seed

    Returns:
    train_i: (np.array) train set indices in X
    dev_i: (np.array) dev set indices in X
    test_i: (np.array) test set indices in X
    info: (dict) detail information about reference and achieved prob distributions
        "dev_size_in_spliton": intended grouping dev_size
        "dev_size_in_X": optimized dev proportion of observations in X
        "test_size_in_spliton": intended grouping test_size
        "test_size_in_X": optimized test proportion of observations in X
        "p_ref_{c}": reference class distribution calculated from stratify_on[c]
        "p_dev_{c}": dev set class distribution calculated from stratify_on[c][dev_i]
        "p_test_{c}": test set class distribution calculated from stratify_on[c][test_i]
    '''
```

### <a name="binning">binning()</a>

```python

def binning(x, nbins=2, lower_boundaries=None, seed=42):

    '''
    bins numeric data.

    If X is one-dimensional:
        binning is done either intrinsically into nbins classes
        based on an equidistant percentile split, or extrinsically
        by using the lower_boundaries values.
    If X is two-dimensional
        binning is done by kmeans clustering into nbins clusters

    Parameters:
    x: (list, np.array) with numeric data.
    nbins: (int) number of bins
    lower_boundaries: (list) of lower bin boundaries.
      If y is 1-dim and lower_boundaries is provided, nbins will be ignored
      and y is binned extrinsically. The first value of lower_boundaries
      is always corrected not to be higher than min(y).
    seed: (int) random seed for kmeans

    Returns:
    c: (np.array) integers as bin IDs
    '''
```

## <a name="usage">Usage</a>

### <a name="example1">Example 1: Split dummy data into training and test partitions</a>

* see `scripts/run_traintest_split.py`
* partitions are:
    * disjunct on categorical "split_var"
    * stratified on categorical "target", "strat_var1", "strat_var2"
    * each contain all levels of "target"

```python
import numpy as np
import os
import pandas as pd
import sys

# add this line if you have cloned the code from github to PROJECT_DIR
# sys.path.append(PROJECT_DIR)

from splitutils import optimize_traindevtest_split

# set seed
seed = 42
np.random.seed(seed)

# DUMMY DATA
# size
n = 100

# feature array
data = np.random.rand(100, 20)

# target variable
target = np.random.choice(["A", "B"], size=n, replace=True)

# array with variable on which to do a disjunct split
split_var = np.random.choice(["D", "E", "F", "G", "H", "I", "J", "K"],
                             size=n, replace=True)

# dict of variables to stratify on. Key names are arbitrary.
stratif_vars = {
    "target": target,
    "strat_var1": np.random.choice(["L", "M"], size=n, replace=True),
    "strat_var2": np.random.choice(["N", "O"], size=n, replace=True)
}

# ARGUMENTS
# weight importance of all stratification variables in stratify_in
# as well as of "size_diff", which punishes the deviation of intended
# and received partition sizes.
# Key names must match the names in stratif_vars.
weights = {
    "target": 2,
    "strat_var1": 1,
    "strat_var2": 1,
    "size_diff": 1
}

# test partition proportion (from 0 to 1)
test_size = .2

# number of disjunct splits to be tried out in brute force optimization
k = 30

# FIND OPTIMAL SPLIT
train_i, test_i, info = optimize_traintest_split(
    X=data,
    y=target,
    split_on=split_var,
    stratify_on=stratif_vars,
    weight=weights,
    test_size=test_size,
    k=k,
    seed=seed
)

# SOME OUTPUT
print("test levels of split_var:", sorted(set(split_var[test_i])))
print("goodness of split:", info)
```

### <a name="example2">Example 2: Split dummy data into training, development, and test partitions</a>

* see `scripts/run_traindevtest_split.py`
* Partitions are
    * disjunct on categorical "split_var"
    * stratified on categorical "target", "strat_var1", "strat_var2"
    * each contain all levels of "target"

```python
import numpy as np
import os
import pandas as pd
import sys

# add this line if you have cloned the code from github to PROJECT_DIR
# sys.path.append(PROJECT_DIR)

from splitutils import optimize_traindevtest_split

# set seed
seed = 42
np.random.seed(seed)

# DUMMY DATA
# size
n = 100

# feature array
data = np.random.rand(100, 20)

# target variable
target = np.random.choice(["A", "B"], size=n, replace=True)

# array with variable on which to do a disjunct split
split_var = np.random.choice(["D", "E", "F", "G", "H", "I", "J", "K"],
                             size=n, replace=True)

# dict of variables to stratify on. Key names are arbitrary.
stratif_vars = {
    "target": target,
    "strat_var1": np.random.choice(["F", "G"], size=n, replace=True),
    "strat_var2": np.random.choice(["H", "I"], size=n, replace=True)
}

# ARGUMENTS
# weight importance of all stratification variables in stratify_in
# as well as of "size_diff", which punishes the deviation of intended
# and received partition sizes.
# Key names must match the names in stratif_vars.
weights = {
    "target": 2,
    "strat_var1": 1,
    "strat_var2": 1,
    "size_diff": 1
}

# dev and test partition proportion (from 0 to 1)
dev_size = .1
test_size = .1

# number of disjunct splits to be tried out in brute force optimization
k = 30

# FIND OPTIMAL SPLIT
train_i, dev_i, test_i, info = optimize_traindevtest_split(
    X=data,
    y=target,
    split_on=split_var,
    stratify_on=stratif_vars,
    weight=weights,
    dev_size=dev_size,
    test_size=test_size,
    k=k,
    seed=seed
)

# SOME OUTPUT
print("test levels of split_var:", sorted(set(split_var[test_i])))
print("goodness of split:", info)
```

### <a name="example3">Example 3: Split dummy data into training, development, and test partitions, the target and several stratification variables being numeric</a>

* see `scripts/run_traindevtest_split_with_binning.py`
* Partitions are
    * disjunct on categorical "split_var"
    * stratified on numeric "target", and on 3 other numeric stratification variables 

```python
import numpy as np
import os
import pandas as pd
import sys

# add this line if you have cloned the code from github to PROJECT_DIR
# sys.path.append(PROJECT_DIR)

from splitutils import (
    binning,
    optimize_traindevtest_split
)

"""
example script how to split dummy data into training, development,
and test partitions that are
* disjunct on categorical "split_var"
* stratified on numeric "target", and on 3 other numeric stratification
  variables
"""

# set seed
seed = 42
np.random.seed(seed)

# DUMMY DATA
# size
n = 100

# features
data = np.random.rand(n, 20)

# numeric target variable
num_target = np.random.rand(n)

# array with variable on which to do a disjunct split
split_var = np.random.choice(["D", "E", "F", "G", "H", "I", "J", "K"],
                             size=n, replace=True)

# further numeric variables to stratify on
num_strat_vars = np.random.rand(n, 3)

# intrinsically bin target into 3 bins by equidistant
# percentile boundaries
binned_target = binning(num_target, nbins=3)

# ... alternatively, a variable can be extrinsically binned by
# specifying lower boundaries:
# binned_target = binning(num_target, lower_boundaries=[0, 0.33, 0.66])

# bin other stratification variables into a single variable with 6 bins
# (2-dim input is binned by StandardScaling and KMeans clustering)
binned_strat_var = binning(num_strat_vars, nbins=6)

# ... alternatively, each stratification variable could be binned
# individually - intrinsically or extrinsically the same way as num_target
# strat_var1 = binning(num_strat_vars[:,0], nbins=...) etc.

# now add the obtained categorical variable to stratification dict
stratif_vars = {
    "target": binned_target,
    "strat_var": binned_strat_var
}

# ARGUMENTS
# weight importance of all stratification variables in stratify_in
# as well as of "size_diff", which punishes the deviation of intended
# and received partition sizes
weights = {
    "target": 2,
    "strat_var": 1,
    "size_diff": 1
}

# dev and test partition proportion (from 0 to 1)
dev_size = .1
test_size = .1

# number of disjunct splits to be tried out in brute force optimization
k = 30

# FIND OPTIMAL SPLIT
train_i, dev_i, test_i, info = optimize_traindevtest_split(
    X=data,
    y=num_target,
    split_on=split_var,
    stratify_on=stratif_vars,
    weight=weights,
    dev_size=dev_size,
    test_size=test_size,
    k=k,
    seed=seed
)

# SOME OUTPUT
print("test levels of split_var:", sorted(set(split_var[test_i])))
print("goodness of split:", info)
```

## <a name="algorithm">Algorithm</a>

* find optimal train, dev, and test set split based on:
    * disjunct split of a categorical grouping variable *G* (e.g. speaker)
    * optimized joint stratification on an arbitrary amount of categorical target and grouping variables (e.g. emotion, gender, ...)
    * close match of partition proportions in *G* and underlying dataset *X*
* brute-force optimization on *k* disjunct splits of *G*
* **score to be minimzed for train/test set split:**

```
(sum_v[w(v) * irad(v)] + w(d) * d) / (sum_v[w(v)] + w(d))

v: variables to be stratified on
w(v): their weight
irad(v): information radius between reference and test set distribution of factor levels in v
d: absolute difference between test proportions of X and G, i.e. between the proportion of test
   samples and the proportion of groups (e.g. speakers) that go into the test set
w(d): its weight
```

* **score to be minimzed for train / dev / test set split:**

```
(sum_v[w(v) * max_irad(v)] + w(d) * max_d) / (sum_v[w(v)] + w(d))

v: variables to be stratified on
w(v): their weight
max_irad(v): maximum information radius of reference distribution of classes in v and
             - dev set distribution,
             - test set distribution
max_d: maximum of absolute difference between proportions of X and G (see above) calculated for
       the dev and test set
w(d): its weight
```

## <a name="interpretation">How to interpret the returned `info` dict</a>

* let's look at [Example 2](#example2) above. There `info` becomes:

```python
{
  'score': 0.030828359568603338,
  'size_devset_in_spliton': 0.1,
  'size_devset_in_X': 0.14,
  'size_testset_in_spliton': 0.1,
  'size_testset_in_X': 0.13,
  'p_target_ref': {'B': 0.49, 'A': 0.51},
  'p_target_dev': {'A': 0.5, 'B': 0.5},
  'p_target_test': {'A': 0.5384615384615384, 'B': 0.46153846153846156},
  'p_strat_var1_ref': {'G': 0.56, 'F': 0.44},
  'p_strat_var1_dev': {'G': 0.5714285714285714, 'F': 0.42857142857142855},
  'p_strat_var1_test': {'F': 0.5384615384615384, 'G': 0.46153846153846156},
  'p_strat_var2_ref': {'I': 0.48, 'H': 0.52},
  'p_strat_var2_dev': {'I': 0.5, 'H': 0.5},
  'p_strat_var2_test': {'I': 0.46153846153846156, 'H': 0.5384615384615384}
}
```

* **Explanations**
    * **score:** see above, **score to be minimzed for train / dev / test set split:**
    * **size_devset_in_spliton:** proportion of to-be-split-on variable levels in development set
    * **size_devset_in_X:** proportion of rows in X in development set
    * **size_testset_in_spliton:** proportion of to-be-split-on variable levels in test set
    * **size_testset_in_X:** proportion of rows in X in test set
    * **p_target_ref:** reference target class distribution over all data
    * **p_target_dev:** target class distribution in development set
    * **p_target_test:** target class distribution in test set
    * **p_strat_var1_ref:** first stratification variable's reference distribution over all data
    * **p_strat_var1_dev:** first stratification variable's class distribution in development set
    * **p_strat_var1_test:** first stratification variable's class distribution in test set
    * **p_strat_var2_ref:** second stratification variable's reference distribution over all data
    * **p_strat_var2_dev:** second stratification variable's class distribution in development set
    * **p_strat_var2_test:** second stratification variable's class distribution in test set
* **Remarks**
    * for `splitutils.optimize_traintest_split()` no development set results are reported
    * all `*_strat_var*` keys: key names derived from key names in `stratify_on` argument


