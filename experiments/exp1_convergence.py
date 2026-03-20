"""
Experiment 1.1: Convergence curves.

Compares β-SVARM against baselines at different budgets.
Run via run_claim1.py or standalone:
    python experiments/exp1_convergence.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.beta_svarm import BetaSVARM
from src.baselines import run_tmc_shapley, run_beta_shapley, run_banzhaf
from src.datasets import load_gaussian, load_adult, load_mnist_pca
from src.utils import plot_convergence


def run_convergence(X_tr, y_tr, X_val, y_val, dataset_name,
                    checkpoints, gt_budget=3000):
    """Run convergence experiment for a single dataset."""
    print(f'\n--- Convergence: {dataset_name} ---')

    print('  Computing ground truth (TMC-Shapley, large budget)...')
    true_values, _ = run_tmc_shapley(X_tr, y_tr, X_val, y_val,
                                      max_updates=gt_budget, n_jobs=4)

    results = {name: [] for name in
               ['β-SVARM', 'TMC-Shapley', 'Beta Shapley', 'Data Banzhaf']}

    for budget in checkpoints:
        print(f'  T = {budget}...')

        vals, _ = BetaSVARM(alpha=16, beta_param=1, random_state=42).fit(
            X_tr, y_tr, X_val, y_val, budget=budget)
        results['β-SVARM'].append(np.mean((vals - true_values) ** 2))

        vals, _ = run_tmc_shapley(X_tr, y_tr, X_val, y_val,
                                   max_updates=budget, n_jobs=4)
        results['TMC-Shapley'].append(np.mean((vals - true_values) ** 2))

        vals, _ = run_beta_shapley(X_tr, y_tr, X_val, y_val,
                                    alpha=1, beta=16, max_updates=budget, n_jobs=4)
        results['Beta Shapley'].append(np.mean((vals - true_values) ** 2))

        vals, _ = run_banzhaf(X_tr, y_tr, X_val, y_val,
                               max_updates=budget, n_jobs=4)
        results['Data Banzhaf'].append(np.mean((vals - true_values) ** 2))

        for name in results:
            print(f'    {name}: MSE = {results[name][-1]:.6f}')

    plot_convergence(checkpoints, results, dataset_name)
    return results


if __name__ == '__main__':
    print('Loading Gaussian dataset...')
    Xg, yg, Xgv, ygv, _, _ = load_gaussian(n_train=15, n_val=200, seed=42)
    run_convergence(Xg, yg, Xgv, ygv, 'Gaussian',
                    checkpoints=[50, 100, 200, 500, 1000, 2000, 3000],
                    gt_budget=10000)

    print('\nLoading Adult dataset...')
    Xa, ya, Xav, yav, _, _ = load_adult(n_train=200, n_val=200, seed=42)
    run_convergence(Xa, ya, Xav, yav, 'Adult',
                    checkpoints=[500, 1000, 2000, 3000, 5000, 8000, 10000],
                    gt_budget=30000)

    print('\nLoading MNIST dataset...')
    Xm, ym, Xmv, ymv, _, _ = load_mnist_pca(n_train=200, n_val=200, seed=42)
    run_convergence(Xm, ym, Xmv, ymv, 'MNIST',
                    checkpoints=[500, 1000, 2000, 3000, 5000, 8000, 10000],
                    gt_budget=30000)

    print('\nDone!')
