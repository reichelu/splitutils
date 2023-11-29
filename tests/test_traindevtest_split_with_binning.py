import numpy as np
import os
import pandas as pd
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from splitutils import (
    binning,
    optimize_traindevtest_split
)

def test_traindevtest_split_with_binning():
    seed = 42
    np.random.seed(seed)
    n = 100
    data = np.random.rand(n, 20)
    num_target = np.random.rand(n)
    split_var = np.random.choice(["D", "E", "F", "G", "H", "I", "J", "K"],
                                 size=n, replace=True)
    num_strat_vars = np.random.rand(n, 3)
    binned_target = binning(num_target, nbins=3)
    binned_strat_var = binning(num_strat_vars, nbins=6)
    stratif_vars = {
        "target": binned_target,
        "strat_var": binned_strat_var
    }
    weights = {
        "target": 2,
        "strat_var": 1,
        "size_diff": 1
    }
    dev_size = .1
    test_size = .1
    k = 30

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

    reference = {
        'score': 0.07371591554676618, 'size_devset_in_spliton': 0.1, 'size_devset_in_X': 0.15, 'size_testset_in_spliton': 0.1, 'size_testset_in_X': 0.12, 'p_target_ref': {0: 0.34, 2: 0.34, 1: 0.32}, 'p_target_dev': {0: 0.26666666666666666, 2: 0.3333333333333333, 1: 0.4}, 'p_target_test': {0: 0.4166666666666667, 2: 0.3333333333333333, 1: 0.25}, 'p_strat_var_ref': {4: 0.2, 2: 0.19, 3: 0.18, 0: 0.11, 5: 0.13, 1: 0.19}, 'p_strat_var_dev': {2: 0.2, 4: 0.2, 3: 0.26666666666666666, 5: 0.06666666666666667, 0: 0.13333333333333333, 1: 0.13333333333333333}, 'p_strat_var_test': {2: 0.25, 4: 0.25, 5: 0.16666666666666666, 1: 0.16666666666666666, 0: 0.08333333333333333, 3: 0.08333333333333333}
    }

    # list of mismatching keys
    mismatches = []
    for key in info:
        if type(info[key]) is not dict:
            if np.round(info[key], 4) != np.round(reference[key], 4):
                mismatches.append(key)
        else:
            for subkey in info[key]:
                if np.round(info[key][subkey], 4) != \
                   np.round(reference[key][subkey], 4):
                    mismatches.append(f"{key}.{subkey}")

    if len(mismatches) > 0:
        print("mismatches:", mismatches)
        return False

    print("ok!")
    return True


if __name__ == "__main__":
    test_traindevtest_split_with_binning()
