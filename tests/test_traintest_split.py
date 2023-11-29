import numpy as np
import os
import pandas as pd
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from splitutils import optimize_traintest_split

# tests traintest split


def test_traintest_split():
    seed = 42
    np.random.seed(seed)
    n = 100
    data = np.random.rand(n, 20)

    target = np.random.choice(["A", "B"], size=n, replace=True)

    split_var = np.random.choice(["D", "E", "F", "G", "H", "I", "J", "K"],
                                 size=n, replace=True)

    stratif_vars = {
        "target": target,
        "strat_var1": np.random.choice(["L", "M"], size=n, replace=True),
        "strat_var2": np.random.choice(["N", "O"], size=n, replace=True)
    }

    weights = {
        "target": 2,
        "strat_var1": 1,
        "strat_var2": 1,
        "size_diff": 1
    }

    test_size = .2
    k = 30

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

    reference = {
        'score': 0.022505470292586156,
        'size_testset_in_spliton': 0.2,
        'size_testset_in_X': 0.27,
        'p_target_ref': {'B': 0.49, 'A': 0.51},
        'p_target_test': {'A': 0.5185185185185185, 'B': 0.48148148148148145},
        'p_strat_var1_ref': {'M': 0.56, 'L': 0.44},
        'p_strat_var1_test': {'M': 0.5185185185185185, 'L': 0.48148148148148145},
        'p_strat_var2_ref': {'O': 0.48, 'N': 0.52},
        'p_strat_var2_test': {'O': 0.48148148148148145, 'N': 0.5185185185185185}
    }

    # list of mismatching keys
    mismatches = []
    for key in info:
        if type(info[key]) is not dict:
            if np.round(info[key], 4) != np.round(reference[key], 4):
                mismatches.append(key)
        else:
            for subkey in info[key]:
                if np.round(info[key][subkey], 4) != np.round(reference[key][subkey], 4):
                    mismatches.append(f"{key}.{subkey}")

    if len(mismatches) > 0:
        print("mismatches:", mismatches)
        return False

    print("ok!")
    return True


if __name__ == "__main__":
    test_traintest_split()
