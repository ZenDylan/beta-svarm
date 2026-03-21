"""
Master script: run ALL Claim 1 experiments.

Usage (on ASUS RTX 4060 machine):
    conda activate beta-svarm
    python experiments/run_claim1.py

Expected runtime: 1-3 hours depending on dataset sizes.
Outputs saved in results/ directory.
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.beta_svarm import BetaSVARM
from src.baselines import run_tmc_shapley, run_beta_shapley, run_banzhaf
from src.datasets import (
    load_2dplanes,
    load_adult,
    load_covertype,
    load_creditcard,
    load_fashion_mnist_pca,
    load_gaussian,
    load_mnist_pca,
    load_phoneme,
)
from src.utils import plot_convergence, plot_runtime, plot_multisemivalue


FORMAL_DATASETS = [
    ('Adult', load_adult),
    ('2dplanes', load_2dplanes),
    ('Covertype', load_covertype),
    ('Phoneme', load_phoneme),
    ('CreditCard', load_creditcard),
    ('MNIST', load_mnist_pca),
    ('FashionMNIST', load_fashion_mnist_pca),
]

CONVERGENCE_CHECKPOINTS = [500, 1000, 2000, 3000, 5000, 8000, 10000]
CONVERGENCE_GT_BUDGET = 30000
RUNTIME_BUDGET = 5000
MULTISEMIVALUE_BUDGET = 10000


def run_convergence(X_tr, y_tr, X_val, y_val, dataset_name,
                    checkpoints, gt_budget=3000):
    """Experiment 1.1: Convergence curves."""
    print(f'\n--- Convergence: {dataset_name} ---')

    print('  Computing ground truth (TMC-Shapley, large budget)...')
    true_values, _ = run_tmc_shapley(X_tr, y_tr, X_val, y_val,
                                     max_updates=gt_budget, n_jobs=1)

    results = {name: [] for name in
               ['β-SVARM', 'TMC-Shapley', 'Beta Shapley', 'Data Banzhaf']}

    for budget in checkpoints:
        print(f'  T = {budget}...')

        vals, _ = BetaSVARM(alpha=16, beta_param=1, random_state=42).fit(
            X_tr, y_tr, X_val, y_val, budget=budget)
        results['β-SVARM'].append(np.mean((vals - true_values) ** 2))

        vals, _ = run_tmc_shapley(X_tr, y_tr, X_val, y_val,
                                  max_updates=budget, n_jobs=1)
        results['TMC-Shapley'].append(np.mean((vals - true_values) ** 2))

        vals, _ = run_beta_shapley(X_tr, y_tr, X_val, y_val,
                                   alpha=1, beta=16, max_updates=budget, n_jobs=1)
        results['Beta Shapley'].append(np.mean((vals - true_values) ** 2))

        vals, _ = run_banzhaf(X_tr, y_tr, X_val, y_val,
                              max_updates=budget, n_jobs=1)
        results['Data Banzhaf'].append(np.mean((vals - true_values) ** 2))

        for name in results:
            print(f'    {name}: MSE = {results[name][-1]:.6f}')

    plot_convergence(checkpoints, results, dataset_name)
    return results


def run_runtime(X_tr, y_tr, X_val, y_val, dataset_name, budget=5000):
    """Experiment 1.2: Runtime comparison."""
    print(f'\n--- Runtime: {dataset_name} ---')

    times = {}

    t0 = time.time()
    BetaSVARM(alpha=16, beta_param=1, random_state=42).fit(
        X_tr, y_tr, X_val, y_val, budget=budget)
    times['β-SVARM'] = time.time() - t0

    t0 = time.time()
    run_tmc_shapley(X_tr, y_tr, X_val, y_val, max_updates=budget, n_jobs=1)
    times['TMC-Shapley'] = time.time() - t0

    t0 = time.time()
    run_beta_shapley(X_tr, y_tr, X_val, y_val, alpha=1, beta=16,
                     max_updates=budget, n_jobs=1)
    times['Beta Shapley'] = time.time() - t0

    t0 = time.time()
    run_banzhaf(X_tr, y_tr, X_val, y_val, max_updates=budget, n_jobs=1)
    times['Data Banzhaf'] = time.time() - t0

    for name, runtime in times.items():
        print(f'  {name}: {runtime:.2f}s')

    plot_runtime(times, dataset_name, budget)
    return times


def run_multisemivalue(X_tr, y_tr, X_val, y_val, dataset_name, budget=10000):
    """Experiment 1.3: Multiple semivalues from one run."""
    print(f'\n--- Multi-semivalue: {dataset_name} ---')

    svarm = BetaSVARM(alpha=16, beta_param=1, adaptive=True, random_state=42)
    values_beta16, meta = svarm.fit(X_tr, y_tr, X_val, y_val, budget=budget)
    n = len(X_tr)

    semivalues = {
        'Shapley': svarm.reweight(meta, n, 1, 1),
        'Beta(4,1)': svarm.reweight(meta, n, 4, 1),
        'Beta(16,1)': values_beta16,
        'Banzhaf': svarm.banzhaf_reweight(meta, n),
    }

    print('  Spearman correlations:')
    from scipy.stats import spearmanr
    for s1 in semivalues:
        for s2 in semivalues:
            if s1 < s2:
                rho, _ = spearmanr(semivalues[s1], semivalues[s2])
                print(f'    {s1} vs {s2}: ρ = {rho:.3f}')

    plot_multisemivalue(semivalues, dataset_name)


if __name__ == '__main__':
    print('=' * 60)
    print('β-SVARM Experiment Suite — Claim 1')
    print('=' * 60)

    print('\nLoading datasets...')

    print('  Gaussian synthetic (n=15 for exact ground truth)...')
    gaussian_data = load_gaussian(n_train=15, n_val=200, seed=42)

    formal_data = {}
    for dataset_name, loader in FORMAL_DATASETS:
        print(f'  {dataset_name} (n_train=200)...')
        formal_data[dataset_name] = loader(n_train=200, n_val=200, n_test=500, seed=42)

    print('\n' + '=' * 60)
    print('EXPERIMENT 1.1: Convergence curves')
    print('=' * 60)

    Xg, yg, Xgv, ygv, _, _ = gaussian_data
    run_convergence(Xg, yg, Xgv, ygv, 'Gaussian',
                    checkpoints=[50, 100, 200, 500, 1000, 2000, 3000],
                    gt_budget=10000)

    for dataset_name, _ in FORMAL_DATASETS:
        X_tr, y_tr, X_val, y_val, _, _ = formal_data[dataset_name]
        run_convergence(X_tr, y_tr, X_val, y_val, dataset_name,
                        checkpoints=CONVERGENCE_CHECKPOINTS,
                        gt_budget=CONVERGENCE_GT_BUDGET)

    print('\n' + '=' * 60)
    print('EXPERIMENT 1.2: Runtime comparison')
    print('=' * 60)

    for dataset_name, _ in FORMAL_DATASETS:
        X_tr, y_tr, X_val, y_val, _, _ = formal_data[dataset_name]
        run_runtime(X_tr, y_tr, X_val, y_val, dataset_name, budget=RUNTIME_BUDGET)

    print('\n' + '=' * 60)
    print('EXPERIMENT 1.3: Multi-semivalue output')
    print('=' * 60)

    for dataset_name, _ in FORMAL_DATASETS:
        X_tr, y_tr, X_val, y_val, _, _ = formal_data[dataset_name]
        run_multisemivalue(X_tr, y_tr, X_val, y_val, dataset_name,
                           budget=MULTISEMIVALUE_BUDGET)

    print('\n' + '=' * 60)
    print('ALL EXPERIMENTS COMPLETE!')
    print('Check results/ directory for output figures.')
    print('Gaussian is used only for exact convergence validation.')
    print('Formal datasets: Adult, 2dplanes, Covertype, Phoneme, CreditCard, MNIST, FashionMNIST.')
    print('=' * 60)
