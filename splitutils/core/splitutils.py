from collections import Counter
import copy
import numpy as np
import pandas as pd
import scipy.spatial as ssp
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import sys
from typing import Any, Tuple, Union


def optimize_traindevtest_split(
        X: Union[np.array, pd.DataFrame],
        y: Union[np.array, list],
        split_on: Union[np.array, list],
        stratify_on: dict,
        weight: dict = None,
        dev_size: float = .1,
        test_size: float = .1,
        testset_not_smaller: bool = False,
        k: int = 30,
        seed: int = 42
) -> Tuple[np.array, np.array, np.array, dict]:
    """Optimize group-disjunct split into training, dev, and test set, which is guided by:
        - disjunct split of values in SPLIT_ON
        - stratification by all keys in STRATIFY_ON (targets and groupings)
        - test set proportion in X should be close to test_size (which is the test
          proportion in set(split_on))

    Score to be minimized: (sum_v[w(v) * max_irad(v)] + w(d) * max_d) / (sum_v[w(v)] + w(d))
        (v: variables to be stratified on
        w(v): their weight
        max_irad(v): maximum information radius of reference distribution of classes in v and
                 - dev set distribution,
                 - test set distribution
        N(v): number of stratification variables
        max_d: maximum of absolute difference between dev and test sizes of X and set(split_on)
        w(d): its weight

    Args:
        X: feature array or table
        y: targets of length N
            if type(y[0]) in ["str", "int"]: y is assumed to be categorical,
            so that it is additionally tested that all partitions cover all classes.
            Else y is assumed to be numeric and no coverage test is done.
        split_on: list of length N with grouping variable (e.g. speaker IDs),
            on which the group-disjunct split is to be performed. Must be categorical.
        stratify_on: dict with variable names (targets and/or further groupings)
            the split should be stratified on (groupings could e.g. be sex, age class, etc).
            Dict-Values are np.array-s of length N that contain the variable values. All
            variables must be categorical.
        weight: weight for each variable in stratify_on. Defines their amount of
            contribution to the optimization score. Uniform weighting by default. Additional
            key: "size_diff" defines how the corresponding size differences should be weighted.
        dev_size: devset proportion in set(split_on) for dev set, e.g. 10% of speakers
            to be held-out
        test_size: test set proportion in set(split_on) for test set
        testset_not_smaller: if True, and if test_size >= dev_size, it is ensured, that
            the resulting test set is not smaller than the dev set
        k: number of different splits to be tried out
        seed: random seed

    Returns:
        train set indices in X
        dev set indices in X
        test set indices in X
        dict with detail information about reference and achieved prob distributions
            "dev_size_in_spliton": intended grouping dev_size
            "dev_size_in_X": optimized dev proportion of observations in X
            "test_size_in_spliton": intended grouping test_size
            "test_size_in_X": optimized test proportion of observations in X
            "p_ref_{c}": reference class distribution calculated from stratify_on[c]
            "p_dev_{c}": dev set class distribution calculated from stratify_on[c][dev_i]
            "p_test_{c}": test set class distribution calculated from stratify_on[c][test_i]

    """
    # data size
    N = len(y)

    # size checks
    assert size_check(stratify_on, N), \
        f"all stratify_on arrays must have length {N}"
    assert size_check(split_on, N), \
        f"split_on array must have length {N}"

    # categorical target: get number of classes for coverage test
    if is_categorical(y[0]):
        nc = len(set(y))
    else:
        nc = None

    # adjusted dev_size after having split off the test set
    dev_size_adj = (dev_size * N) / (N - test_size * N)

    # split all into train/dev vs test
    gss_o = GroupShuffleSplit(n_splits=k, test_size=test_size,
                              random_state=seed)

    # split train/dev into train vs dev
    gss_i = GroupShuffleSplit(n_splits=k, test_size=dev_size_adj,
                              random_state=seed)

    # set weight defaults
    if weight is None:
        weight = {}
    for c in stratify_on.keys():
        if c not in weight:
            weight[c] = 1
    if "size_diff" not in weight:
        weight["size_diff"] = 1

    # stratification reference distributions calculated on stratify_on
    p_ref = {}
    for c in stratify_on:
        p_ref[c] = class_prob(stratify_on[c])

    # best train/dev/test indices in X; best associated score
    train_i, dev_i, test_i, best_sco = None, None, None, np.inf

    # full target coverage in all partitions
    full_target_coverage = False

    # for appropriate subsetting
    xtype = type(X)

    # brute-force optimization of SPLIT_ON split
    #    outer loop *_o: splitting into train/dev and test
    #    inner loop *_i: spltting into train and dev
    for tri_o, tei_o in gss_o.split(X, y, split_on):

        # current train/dev partition
        if xtype == pd.DataFrame:
            X_i = X.iloc[tri_o]
        else:
            X_i = X[tri_o]
        y_i = y[tri_o]
        split_on_i = split_on[tri_o]

        for tri_i, tei_i in gss_i.split(X_i, y_i, split_on_i):

            # all classes maintained in all partitions?
            if nc:
                nc_train = len(set(y[tri_o[tri_i]]))
                nc_dev = len(set(y[tri_o[tei_i]]))
                nc_test = len(set(y[tei_o]))
                if min(nc_train, nc_dev, nc_test) < nc:
                    continue

            full_target_coverage = True

            sco = calc_split_score(test_i=tei_o,
                                   stratify_on=stratify_on,
                                   weight=weight, p_ref=p_ref,
                                   N=N, test_size=test_size,
                                   dev_i=tri_o[tei_i],
                                   dev_size=dev_size_adj)

            if sco < best_sco:
                best_sco = sco
                test_i = tei_o
                train_i = tri_o[tri_i]
                dev_i = tri_o[tei_i]

    if test_i is None:
        sys.exit(exit_message(full_target_coverage, "dev and test"))

    # evtl. swap dev and test indices
    if test_size >= dev_size and testset_not_smaller:
        if len(dev_i) > len(test_i):
            b = test_i
            test_i = dev_i
            dev_i = b
        
    # matching info
    info = {"score": best_sco,
            "size_devset_in_spliton": dev_size,
            "size_devset_in_X": np.round(len(dev_i) / N, 2),
            "size_testset_in_spliton": test_size,
            "size_testset_in_X": np.round(len(test_i) / N, 2)}

    for c in p_ref:
        info[f"p_{c}_ref"] = p_ref[c]
        info[f"p_{c}_dev"] = class_prob(stratify_on[c][dev_i])
        info[f"p_{c}_test"] = class_prob(stratify_on[c][test_i])
        
    return train_i, dev_i, test_i, info


def optimize_traintest_split(
        X: Union[np.array, pd.DataFrame],
        y: Union[np.array, list],
        split_on: Union[np.array, list],
        stratify_on: dict,
        weight: dict = None,
        test_size: float = .1,
        k: int = 30,
        seed: int = 42
) -> Tuple[np.array, np.array, dict]:
    """Optimize group-disjunct split which is guided by:

        - disjunct split of values in SPLIT_ON
        - stratification by all keys in STRATIFY_ON (targets and groupings)
        - test set proportion in X should be close to test_size (which is the test
          proportion in set(split_on))

    Score to be minimized: (sum_v[w(v) * irad(v)] + w(d) * d) / (sum_v[w(v)] + w(d))
        (v: variables to be stratified on
        w(v): their weight
        irad(v): information radius between reference distribution of classes in v
        and test set distribution
        N(v): number of stratification variables
        d: absolute difference between test sizes of X and set(split_on)
        w(d): its weight

    Args:
        X: features array or dataframe
        y: array of targets of length N
            if type(y[0]) in ["str", "int"]: y is assumed to be categorical, so that it is
            additionally tested that all partitions cover all classes. Else y is assumed to
            be numeric and no coverage test is done.
        split_on: list of length N with grouping variable (e.g. speaker IDs),
            on which the group-disjunct split is to be performed. Must be categorical.
        stratify_on: dict with variable names (targets and/or further groupings)
            the split should be stratified on (groupings could e.g. be sex, age class, etc).
            Dict-Values are np.array-s of length N that contain the variable values.
            All variables must be categorical.
        weight: weight for each variable in stratify_on. Defines their amount of
            contribution to the optimization score. Uniform weighting by default. Additional
            key: "size_diff" defines how test size diff should be weighted.
        test_size: test proportion in set(split_on), e.g. 10% of speakers to be held-out
        k: number of different splits to be tried out
        seed: random seed

    Returns:
        train set indices in X
        test set indices in X
        dict with detail information about reference and achieved prob distributions
            "size_testset_in_spliton": intended test_size
            "size_testset_in_X": optimized test proportion in X
            "p_ref_{c}": reference class distribution calculated from stratify_on[c]
            "p_test_{c}": test set class distribution calculated from stratify_on[c][test_i]

    """
    gss = GroupShuffleSplit(n_splits=k, test_size=test_size,
                            random_state=seed)

    # data size
    N = len(y)

    # size checks
    assert size_check(stratify_on, N), \
        f"all stratify_on arrays must have length {N}"
    assert size_check(split_on, N), \
        f"split_on array must have length {N}"

    # set weight defaults
    if weight is None:
        weight = {}
    for c in stratify_on.keys():
        if c not in weight:
            weight[c] = 1
    if "size_diff" not in weight:
        weight["size_diff"] = 1

    # stratification reference distributions calculated on stratify_on
    p_ref = {}
    for c in stratify_on:
        p_ref[c] = class_prob(stratify_on[c])

    # best train and test indices in X; best associated score
    train_i, test_i, best_sco = None, None, np.inf

    # full target coverage in all partitions
    full_target_coverage = False

    # categorical target: number of classes for coverage test
    if is_categorical(y[0]):
        nc = len(set(y))
    else:
        nc = None

    # brute-force optimization of SPLIT_ON split
    for tri, tei in gss.split(X, y, split_on):

        # all classes maintained in all partitions?
        if nc:
            nc_train = len(set(y[tri]))
            nc_test = len(set(y[tei]))
            if min(nc_train, nc_test) < nc:
                continue

        full_target_coverage = True

        sco = calc_split_score(tei, stratify_on, weight, p_ref, N, test_size)
        if sco < best_sco:
            train_i, test_i, best_sco = tri, tei, sco

    if test_i is None:
        sys.exit(exit_message(full_target_coverage))

    # matching info
    info = {"score": best_sco,
            "size_testset_in_spliton": test_size,
            "size_testset_in_X": np.round(len(test_i) / N, 2)}

    for c in p_ref:
        info[f"p_{c}_ref"] = p_ref[c]
        info[f"p_{c}_test"] = class_prob(stratify_on[c][test_i])

    return train_i, test_i, info


def binning(x: Union[list, np.array],
            nbins: int = 2,
            lower_boundaries: Union[list, np.array, dict] = None,
            seed: int = 42) -> np.array:
    """Bins numeric data.

        If X is one-dimensional:
        binning is done either intrinsically into nbins classes
        based on an equidistant percentile split, or extrinsically
        by using the lower_boundaries values.
        If X is two-dimensional
        binning is done by kmeans clustering into nbins clusters

    Args:
        x: array with numeric data.
        nbins: number of bins
        lower_boundaries: (list of lower bin boundaries.
            If y is 1-dim and lower_boundaries is provided, nbins will be ignored
            and y is binned extrinsically. The first value of lower_boundaries
            is always corrected not to be higher than min(y).
        seed: random seed for kmeans

    Returns:
        array of integers as bin IDs 
    """

    assert ((nbins is not None) or (lower_boundaries is not None)), \
        "One of nbins or lower_boundaries must be set."

    x = np.array(x, dtype=float)

    assert ((nbins is not None) or x.ndim == 1), \
        "For 2-dimensional data input nbins must be set for KMeans clustering."

    if x.ndim == 1:
        # 1-dim array
        if lower_boundaries is None:
            # intrinsic binning by equidistant percentiles
            prct = np.linspace(0, 100, nbins+1)
            lower_boundaries = np.percentile(x, prct)
            lower_boundaries = lower_boundaries[0:nbins]
        else:
            # extrinsic binning
            # make sure that entire range of x is covered
            lower_boundaries[0] = min(lower_boundaries[0], np.min(x))

        # binned array
        c = np.zeros(len(x), dtype=int)
        for i in range(1, len(lower_boundaries)):
            c[x >= lower_boundaries[i]] = i

    else:
        # 2-dim array
        # centering+scaling
        sca = StandardScaler()
        xs = sca.fit_transform(x)
        # clustering
        mod = KMeans(n_clusters=nbins, init='k-means++', max_iter=300, tol=0.0001,
                     random_state=seed, algorithm='lloyd', n_init=1)
        mod.fit(xs)
        c = mod.predict(xs)

    return c


def calc_split_score(
        test_i: np.array,
        stratify_on: dict,
        weight: dict,
        p_ref: dict,
        N: int,
        test_size: float,
        dev_i: np.array = None,
        dev_size: float = None
) -> float:
    """Calculate split score based on class distribution IRADs and
        differences in partition sizes of groups vs observations; smaller is better.
        If dev_i and dev_size are not provided, the score is calculated for the train/test
        split only. If they are provided the score is calculated for the train/dev/test split

    Args:
        test_i: array of test set indices
        stratify_on: dict of variable names (targets and/or further groupings)
            the split should be stratified on (groupings could e.g. be sex, age class, etc).
            Dict-Values are np.array-s of length N that contain the variable values.
        weight: weight for each variable in stratify_on. Additional
            key: "size_diff" that weights the grouping vs observation level test set size difference
        p_ref: reference class distributions for all variables in stratify_on
        N: size of underlying data set
        test_size: test proportion in value set of variable, the disjunct grouping
            has been carried out
        dev_i: array of dev indices
        dev_size: dev proportion in value set of variable, the disjunct grouping
            has been carried out (this value should have been adjusted after splitting off the
            test set)

    Returns:
        split score

    """
    if dev_i is None:
        do_dev = False
    else:
        do_dev = True

    # dev and test set class distributions
    p_test, p_dev = {}, {}
    for c in p_ref:
        p_test[c] = class_prob(stratify_on[c][test_i])
        if do_dev:
            p_dev[c] = class_prob(stratify_on[c][dev_i])

    # score
    sco, wgt = 0, 0

    # IRADs (if p_test[c] or p_dec[c] do not contain
    # all classes in p_ref[c], return INF)
    for c in p_ref:
        irad, full_coverage = calc_irad(p_ref[c], p_test[c])
        if not full_coverage:
            return np.inf
        if do_dev:
            irad_dev, full_coverage = calc_irad(p_ref[c], p_dev[c])
            if not full_coverage:
                return np.inf
            irad = max(irad, irad_dev)

        sco += (weight[c] * irad)
        wgt += weight[c]

    # partition size difference groups vs observations
    size_diff = np.abs(len(test_i) / N - test_size)
    if do_dev:
        size_diff_dev = np.abs(len(dev_i) / N - dev_size)
        size_diff = max(size_diff, size_diff_dev)

    sco += (weight["size_diff"] * size_diff)
    wgt += weight["size_diff"]

    sco /= wgt

    return sco


def calc_irad(
        p1: dict,
        p2: dict
) -> Tuple[float, bool]:
    """Calculate information radius of prob dicts p1 and p2.
    
    Args:
        p1, p2: dicts of probabilities

    Returns:
        information radius
        full coverage boolean which is True if all elements
        in p1 occur in p2 and vice versa

    """

    p, q = [], []
    full_coverage = True

    for u in sorted(p1.keys()):

        if u not in p2:
            full_coverage = False
            a = 0.0
        else:
            a = p2[u]

        p.append(p1[u])
        q.append(a)

    if full_coverage:
        if len(p2.keys()) > len(p1.keys()):
            full_coverage = False

    irad = ssp.distance.jensenshannon(p, q)

    return irad, full_coverage


def size_check(
        d=Union[list, np.array, dict, pd.DataFrame],
        n=int
) -> bool:
    """Size check for d. If list, np.array it's length is tested.
        If dict, the the length of all its values (which are lists) is
        tested. If pd.DataFrame, shape[0] is tested.

    Args:
        d: input data
        n: reference length

    Returns:
        True if size requirement is met, else False

    """
    if type(d) in [list, np.array] and len(d) != n:
        return False
    elif type(d) is dict:
        for v in d:
            if len(d[v]) != n:
                return False
    elif d.shape[0] != n:
        return False

    return True


def class_prob(
        y: Union[np.array, list]
) -> dict:
    """Returns class probabilities in y.

    Args:
        y: array of classes

    Returns:
        dict assigning to each class in Y its maximum likelihood

    """
    p = {}
    N = len(y)
    c = Counter(y)
    for x in c:
        p[x] = c[x] / N

    return p


def is_categorical(x: Any) -> bool:
    """Returns True if type of x is in str or int*, else False."""
    if type(x) in [str, int, np.int16, np.int32, np.int64,
                   np.uint8, np.uint16, np.uint32]:
        return True

    return False


def exit_message(
        full_target_coverage: bool,
        infx: str = "test"
) -> str:
    """Exit messages."""
    if not full_target_coverage:
        return "not all partitions contain all target classes. What you can do:\n" \
            "(1) increase your dev and/or test partition, or\n" \
            "(2) reduce the amount of target classes by merging some of them."
    return f"\n:-o No {infx} set split found. Reason is, that for at least one of the\n" \
        f"stratification variables not all its values can make it into the {infx} set.\n" \
        f"This happens e.g. if the {infx} set size is chosen too small or\n" \
        "if the (multidimensional) distribution of the stratification\n" \
        "variables is sparse. What you can do:\n" \
        "(1) remove a variable from this stratification, or\n" \
        "(2) merge classes within a variable to increase the per class probabilities, or\n" \
        f"(3) increase the {infx} set size, or\n" \
        "(4) increase the number of different splits (if it was small, say < 10, before), or\n" \
        "(5) in case your target is numeric and you have added a binned target array to the\n" \
        "    stratification variables: reduce the number of bins.\n" \
        "Good luck!\n"
