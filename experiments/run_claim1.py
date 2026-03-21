"""
Experiment suite for β-SVARM downstream task evaluation.

Experiments (no ground truth needed):
  Exp 1: Point Removal  — per-dataset line plots
  Exp 2: Point Addition — per-dataset line plots
  Exp 3: Noisy Label Detection — per-dataset line plots
  Exp 4: Runtime bar chart on Adult dataset

All figures saved to results/.
"""

import os
import sys
import time
import warnings
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.beta_svarm import BetaSVARM
from src.baselines import (
    run_tmc_shapley, run_beta_shapley,
    run_banzhaf, run_loo, run_random,
)
from src.datasets import (
    load_adult, load_covertype, load_phoneme,
    load_creditcard, load_2dplanes, load_mnist_pca,
    load_fashion_mnist_pca, load_gaussian,
)
from src.utils import (
    plot_point_removal, plot_point_addition,
    plot_noisy_detection, plot_runtime,
)

warnings.filterwarnings(
    "ignore",
    message=".*samples of at least 2 classes.*",
    category=RuntimeWarning,
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
N_TRAIN = 200
N_VAL = 200
N_TEST = 500
BUDGET = 5000
SEED = 42
REMOVAL_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ADDITION_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
NOISE_RATE = 0.10
DETECTION_FRACTIONS = [0.01, 0.05, 0.10, 0.15, 0.20]

DATASETS = {
    'Adult':         partial(load_adult,            n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, seed=SEED),
    'Covertype':    partial(load_covertype,         n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, seed=SEED),
    'Phoneme':      partial(load_phoneme,           n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, seed=SEED),
    'CreditCard':   partial(load_creditcard,         n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, seed=SEED),
    '2dplanes':     partial(load_2dplanes,           n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, seed=SEED),
    'MNIST':        partial(load_mnist_pca,         n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, seed=SEED),
    'FashionMNIST': partial(load_fashion_mnist_pca, n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, seed=SEED),
    'Gaussian':     partial(load_gaussian,           n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, seed=SEED),
}

METHODS = [
    ('β-SVARM (Ours)', 'beta_svarm'),
    ('TMC-Shapley',    'tmc'),
    ('Beta Shapley',   'beta_shapley'),
    ('Data Banzhaf',   'banzhaf'),
    ('LOO',            'loo'),
    ('Random',         'random'),
]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _train_eval(X_tr, y_tr, X_te, y_te):
    """Train LogisticRegression and return test accuracy."""
    if len(np.unique(y_tr)) < 2:
        return float('nan')
    try:
        model = LogisticRegression(max_iter=500, random_state=SEED)
        model.fit(X_tr, y_tr)
        return accuracy_score(y_te, model.predict(X_te))
    except Exception:
        return float('nan')


def _flip_labels(y, rate, seed):
    """Randomly flip `rate` fraction of labels (within each class)."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y, dtype=int)
    idx_0 = np.flatnonzero(y == 0)
    idx_1 = np.flatnonzero(y == 1)
    n_flip = int(len(y) * rate)

    all_idx = np.concatenate([idx_0, idx_1])
    rng.shuffle(all_idx)
    flip_idx = all_idx[:n_flip]
    y_noisy = y.copy()
    y_noisy[flip_idx] = 1 - y_noisy[flip_idx]
    return y_noisy, set(flip_idx)


# ---------------------------------------------------------------------------
# Valuation runners
# ---------------------------------------------------------------------------
def _run_beta_svarm(X_tr, y_tr, X_val, y_val, budget, **kwargs):
    svarm = BetaSVARM(alpha=16, beta_param=1, adaptive=False, random_state=SEED)
    vals, _ = svarm.fit(X_tr, y_tr, X_val, y_val, budget=budget)
    return vals


def _run_tmc(X_tr, y_tr, X_val, y_val, budget, **kwargs):
    vals, _ = run_tmc_shapley(X_tr, y_tr, X_val, y_val,
                              max_updates=budget, n_jobs=1, seed=SEED)
    return vals


def _run_beta_shapley(X_tr, y_tr, X_val, y_val, budget, **kwargs):
    vals, _ = run_beta_shapley(X_tr, y_tr, X_val, y_val,
                               alpha=1, beta=16, max_updates=budget, n_jobs=1, seed=SEED)
    return vals


def _run_banzhaf(X_tr, y_tr, X_val, y_val, budget, **kwargs):
    vals, _ = run_banzhaf(X_tr, y_tr, X_val, y_val,
                          max_updates=budget, n_jobs=1, seed=SEED)
    return vals


def _run_loo(X_tr, y_tr, X_val, y_val, budget, **kwargs):
    vals, _ = run_loo(X_tr, y_tr, X_val, y_val, seed=SEED)
    return vals


def _run_random(X_tr, y_tr, X_val, y_val, budget, n_train, **kwargs):
    vals, _ = run_random(n_train, seed=SEED)
    return vals


VAL_RUNNERS = {
    'beta_svarm':  _run_beta_svarm,
    'tmc':         _run_tmc,
    'beta_shapley': _run_beta_shapley,
    'banzhaf':     _run_banzhaf,
    'loo':         _run_loo,
    'random':      _run_random,
}


# ---------------------------------------------------------------------------
# Experiment 1: Point Removal
# ---------------------------------------------------------------------------
def run_point_removal(dataset_name, load_fn):
    """Point removal: remove high-value points first, track test accuracy."""
    print(f'\n=== Point Removal: {dataset_name} ===')
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_fn()
    n = len(X_tr)
    budget = BUDGET

    results = {}

    for method_name, method_key in METHODS:
        print(f'  {method_name}...')
        if method_key == 'random':
            vals = _run_random(X_tr, y_tr, X_val, y_val, budget, n)
        else:
            vals = VAL_RUNNERS[method_key](X_tr, y_tr, X_val, y_val, budget)

        # Sort by value descending → remove HIGH-value points first
        order = np.argsort(vals)[::-1]
        accs = []
        for frac in REMOVAL_FRACTIONS:
            n_remove = int(n * frac)
            n_keep = n - n_remove
            if n_keep == 0:
                accs.append(float('nan'))
                continue
            keep_idx = order[n_remove:]  # skip the top n_remove highest-value points
            acc = _train_eval(X_tr[keep_idx], y_tr[keep_idx], X_te, y_te)
            accs.append(acc)
        results[method_name] = accs
        print(f'    {[f"{a:.3f}" for a in accs]}')

    plot_point_removal(REMOVAL_FRACTIONS, results, dataset_name)
    return results


# ---------------------------------------------------------------------------
# Experiment 2: Point Addition
# ---------------------------------------------------------------------------
def run_point_addition(dataset_name, load_fn):
    """Point addition: add high-value points first, track test accuracy."""
    print(f'\n=== Point Addition: {dataset_name} ===')
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_fn()
    n = len(X_tr)
    budget = BUDGET

    results = {}

    for method_name, method_key in METHODS:
        print(f'  {method_name}...')
        if method_key == 'random':
            vals = _run_random(X_tr, y_tr, X_val, y_val, budget, n)
        else:
            vals = VAL_RUNNERS[method_key](X_tr, y_tr, X_val, y_val, budget)

        # Sort by value descending → add highest-value points first
        order = np.argsort(vals)[::-1]
        accs = []
        for frac in ADDITION_FRACTIONS:
            n_add = int(n * frac)
            if n_add == 0:
                # Use most common class as baseline
                mc = np.bincount(y_tr).argmax()
                acc = np.mean(y_te == mc)
                accs.append(acc)
                continue
            add_idx = order[:n_add]
            acc = _train_eval(X_tr[add_idx], y_tr[add_idx], X_te, y_te)
            accs.append(acc)
        results[method_name] = accs
        print(f'    {[f"{a:.3f}" for a in accs]}')

    plot_point_addition(ADDITION_FRACTIONS, results, dataset_name)
    return results


# ---------------------------------------------------------------------------
# Experiment 3: Noisy Label Detection
# ---------------------------------------------------------------------------
def run_noisy_detection(dataset_name, load_fn):
    """Noisy label detection: check if low-value points are the noisy ones."""
    print(f'\n=== Noisy Label Detection: {dataset_name} ===')
    X_tr, y_tr_orig, X_val, y_val, X_te, y_te = load_fn()
    n = len(X_tr)
    budget = BUDGET

    # Flip labels
    y_tr_noisy, noisy_idx = _flip_labels(y_tr_orig, NOISE_RATE, SEED)
    print(f'  Flipped {len(noisy_idx)}/{n} labels ({len(noisy_idx)/n:.1%})')

    results = {}

    for method_name, method_key in METHODS:
        print(f'  {method_name}...')
        if method_key == 'random':
            vals = _run_random(X_tr, y_tr_orig, X_val, y_val, budget, n)
        else:
            # Use noisy labels for valuation
            vals = VAL_RUNNERS[method_key](X_tr, y_tr_noisy, X_val, y_val, budget)

        # Sort by value ascending → lowest value = most suspicious
        order = np.argsort(vals)
        detection_rates = []
        total_noisy = len(noisy_idx)

        for frac in DETECTION_FRACTIONS:
            n_check = max(int(n * frac), 1)
            checked = set(order[:n_check])
            found = len(checked & noisy_idx)
            rate = found / total_noisy if total_noisy > 0 else 0.0
            detection_rates.append(rate)
        results[method_name] = detection_rates
        print(f'    {[f"{r:.2f}" for r in detection_rates]}')

    plot_noisy_detection(DETECTION_FRACTIONS, results, dataset_name)
    return results


# ---------------------------------------------------------------------------
# Experiment 4: Runtime comparison
# ---------------------------------------------------------------------------
def run_runtime_comparison():
    """Wall-clock time comparison on Adult dataset at fixed budget."""
    print('\n=== Runtime Comparison: Adult ===')
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_adult(
        n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, seed=SEED)
    budget = BUDGET

    times = {}
    methods_to_time = [
        ('β-SVARM',       'beta_svarm'),
        ('TMC-Shapley',   'tmc'),
        ('Beta Shapley',  'beta_shapley'),
        ('Data Banzhaf',  'banzhaf'),
        ('LOO',           'loo'),
    ]

    for method_name, method_key in methods_to_time:
        print(f'  {method_name}...')
        t0 = time.time()
        if method_key == 'random':
            _run_random(X_tr, y_tr, X_val, y_val, budget, len(X_tr))
        else:
            VAL_RUNNERS[method_key](X_tr, y_tr, X_val, y_val, budget)
        elapsed = time.time() - t0
        times[method_name] = elapsed
        print(f'    {elapsed:.1f}s')

    plot_runtime(times, 'Adult', budget)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('=' * 60)
    print('β-SVARM Downstream Task Experiments')
    print(f'  n_train={N_TRAIN}, budget={BUDGET}, seed={SEED}')
    print('=' * 60)

    # Exp 1: Point Removal
    print('\n' + '#' * 60)
    print('# Experiment 1: Point Removal')
    print('#' * 60)
    for ds_name, load_fn in DATASETS.items():
        run_point_removal(ds_name, load_fn)

    # Exp 2: Point Addition
    print('\n' + '#' * 60)
    print('# Experiment 2: Point Addition')
    print('#' * 60)
    for ds_name, load_fn in DATASETS.items():
        run_point_addition(ds_name, load_fn)

    # Exp 3: Noisy Label Detection
    print('\n' + '#' * 60)
    print('# Experiment 3: Noisy Label Detection')
    print('#' * 60)
    for ds_name, load_fn in DATASETS.items():
        run_noisy_detection(ds_name, load_fn)

    # Exp 4: Runtime
    print('\n' + '#' * 60)
    print('# Experiment 4: Runtime Comparison')
    print('#' * 60)
    run_runtime_comparison()

    print('\n' + '=' * 60)
    print('All results saved to results/')
    print('=' * 60)
