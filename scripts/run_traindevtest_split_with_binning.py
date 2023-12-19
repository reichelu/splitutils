import numpy as np
import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

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
# binned_target = binning(target, lower_boundaries: [0.33, 0.66])

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
