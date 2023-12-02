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
        'score': 0.13115023887323182, 'size_devset_in_spliton': 0.1, 'size_devset_in_X': 0.12, 'size_testset_in_spliton': 0.1, 'size_testset_in_X': 0.18, 'p_target_ref': {0: 0.34, 2: 0.34, 1: 0.32}, 'p_target_dev': {0: 0.4166666666666667, 2: 0.4166666666666667, 1: 0.16666666666666666}, 'p_target_test': {2: 0.2777777777777778, 1: 0.3333333333333333, 0: 0.3888888888888889}, 'p_strat_var_ref': {3: 0.25, 1: 0.13, 0: 0.16, 4: 0.12, 2: 0.25, 5: 0.09}, 'p_strat_var_dev': {3: 0.3333333333333333, 2: 0.08333333333333333, 0: 0.16666666666666666, 5: 0.08333333333333333, 4: 0.08333333333333333, 1: 0.25}, 'p_strat_var_test': {0: 0.16666666666666666, 2: 0.3333333333333333, 4: 0.16666666666666666, 3: 0.2222222222222222, 5: 0.05555555555555555, 1: 0.05555555555555555}
    }

    for key in info:
        if type(info[key]) is not dict:
            assert np.round(info[key], 4) == np.round(reference[key], 4), \
                f"test fails for {key}"
            
        else:
            for subkey in info[key]:
                assert np.round(info[key][subkey], 4) == np.round(reference[key][subkey], 4), \
                    f"test fails for {key}.{subkey}"

if __name__ == "__main__":
    test_traindevtest_split_with_binning()
