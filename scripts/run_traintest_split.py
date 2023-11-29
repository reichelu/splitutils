import numpy as np
import os
import pandas as pd
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from splitutils import optimize_traintest_split

"""
example script how to split dummy data into training and
test partitions that are
* disjunct on categorical "split_var"
* stratified on categorical "target", "strat_var1", "strat_var2"
* each contain all levels of "target"
"""

# set seed
seed = 42
np.random.seed(seed)

# DUMMY DATA
# size
n = 100

# features
data = np.random.rand(n, 20)

# target variable
target = np.random.choice(["A", "B"], size=n, replace=True)

# array with variable on which to do a disjunct split
split_var = np.random.choice(["D", "E", "F", "G", "H", "I", "J", "K"],
                             size=n, replace=True)

# dict of variables to stratify on
stratif_vars = {
    "target": target,
    "strat_var1": np.random.choice(["L", "M"], size=n, replace=True),
    "strat_var2": np.random.choice(["N", "O"], size=n, replace=True)
}

# ARGUMENTS
# weight importance of all stratification variables in stratify_in
# as well as of "size_diff", which punishes the deviation of intended
# and received partition sizes
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
