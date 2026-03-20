"""
Toy verification: n=6 players, compare β-SVARM against brute-force exact values.
Run this FIRST to make sure the algorithm is correct before running full experiments.
Expected runtime: ~30 seconds.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from src.beta_svarm import BetaSVARM
from src.datasets import load_gaussian


def _eval_utility(indices, X_train, y_train, X_val, y_val):
    """Simple utility: train logistic regression, return accuracy."""
    if len(indices) == 0:
        most_common = np.bincount(y_val.astype(int)).argmax()
        return np.mean(y_val == most_common)
    X_sub, y_sub = X_train[indices], y_train[indices]
    if len(np.unique(y_sub)) < 2:
        return np.mean(y_val == y_sub[0])
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_sub, y_sub)
    return model.score(X_val, y_val)


def exact_shapley(X_train, y_train, X_val, y_val):
    """Brute-force exact Shapley values for small n."""
    n = len(X_train)
    from math import factorial

    values = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for size in range(n):
            for S in combinations(others, size):
                S_list = list(S)
                v_with = _eval_utility(
                    np.array(S_list + [i]), X_train, y_train, X_val, y_val)
                v_without = _eval_utility(
                    np.array(S_list) if S_list else np.array([], dtype=int),
                    X_train, y_train, X_val, y_val)
                weight = factorial(size) * factorial(n - 1 - size) / factorial(n)
                values[i] += weight * (v_with - v_without)
    return values


if __name__ == '__main__':
    print('Toy verification test (n=6)')
    print('-' * 40)

    # Small dataset
    X_tr, y_tr, X_val, y_val, _, _ = load_gaussian(n_train=6, n_val=100, seed=42)
    n = len(X_tr)

    # Exact Shapley
    print('Computing exact Shapley values (brute force)...')
    exact = exact_shapley(X_tr, y_tr, X_val, y_val)
    print(f'  Exact: {exact}')

    # β-SVARM with Shapley weights (α=1, β=1)
    print('Running β-SVARM (α=1, β=1, budget=3000)...')
    svarm = BetaSVARM(alpha=1, beta_param=1, adaptive=False, random_state=42)
    estimated, meta = svarm.fit(X_tr, y_tr, X_val, y_val, budget=3000)
    print(f'  β-SVARM: {estimated}')

    # Compare
    mse = np.mean((estimated - exact) ** 2)
    rank_corr = np.corrcoef(exact, estimated)[0, 1]
    print(f'\n  MSE: {mse:.6f}')
    print(f'  Pearson correlation: {rank_corr:.4f}')

    if rank_corr > 0.8:
        print('\n  ✓ PASS — β-SVARM values correlate well with exact Shapley.')
    else:
        print('\n  ✗ FAIL — correlation too low, check implementation.')

    # Also test reweighting
    print('\nTesting reweight (Beta(16,1) from same run)...')
    beta16_values = svarm.reweight(meta, n, alpha_new=16, beta_new=1)
    print(f'  Beta(16,1): {beta16_values}')
    print(f'  Rank preserved: {np.corrcoef(exact, beta16_values)[0,1]:.4f}')

    print('\nDone!')
