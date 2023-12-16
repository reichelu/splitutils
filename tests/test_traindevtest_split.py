import numpy as np
import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from splitutils import optimize_traindevtest_split

# tests traindevtest split


def test_traindevtest_split():

    seed = 42
    np.random.seed(seed)
    n = 100

    data = np.random.rand(n, 20)

    target = np.random.choice(["A", "B"], size=n, replace=True)
    split_var = np.random.choice(["D", "E", "F", "G", "H", "I", "J", "K"],
                                 size=n, replace=True)

    stratif_vars = {
        "target": target,
        "strat_var1": np.random.choice(["F", "G"], size=n, replace=True),
        "strat_var2": np.random.choice(["H", "I"], size=n, replace=True)
    }

    weights = {
        "target": 2,
        "strat_var1": 1,
        "strat_var2": 1,
        "size_diff": 1
    }

    dev_size = .1
    test_size = .1
    k = 30

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

    reference = {
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

    for key in info:
        if type(info[key]) is not dict:
            assert np.round(info[key], 4) == np.round(reference[key], 4), \
                f"test fails for {key}"
            
        else:
            for subkey in info[key]:
                assert np.round(info[key][subkey], 4) == np.round(reference[key][subkey], 4), \
                    f"test fails for {key}.{subkey}"
            
