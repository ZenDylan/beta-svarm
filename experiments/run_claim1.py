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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from sklearn.linear_model import LogisticRegression

from src.beta_svarm import BetaSVARM
from src.baselines import (run_tmc_shapley, run_beta_shapley, run_banzhaf,
                            run_loo, run_random)
from src.datasets import load_adult, load_mnist_pca, load_gaussian
from src.utils import plot_convergence, plot_runtime, plot_multisemivalue


def run_convergence(X_tr, y_tr, X_val, y_val, dataset_name,
                    checkpoints, gt_budget=3000):
    """Experiment 1.1: Convergence curves."""
    print(f'\n--- Convergence: {dataset_name} ---')

    # Ground truth: large-budget TMC-Shapley via pyDVL
    print('  Computing ground truth (TMC-Shapley, large budget)...')
    true_values, _ = run_tmc_shapley(X_tr, y_tr, X_val, y_val,
                                      max_updates=gt_budget, n_jobs=4)

    results = {name: [] for name in
               ['β-SVARM', 'TMC-Shapley', 'Beta Shapley', 'Data Banzhaf']}

    for budget in checkpoints:
        print(f'  T = {budget}...')

        # β-SVARM (ours)
        vals, _ = BetaSVARM(alpha=16, beta_param=1, random_state=42).fit(
            X_tr, y_tr, X_val, y_val, budget=budget)
        results['β-SVARM'].append(np.mean((vals - true_values) ** 2))

        # TMC-Shapley via pyDVL
        vals, _ = run_tmc_shapley(X_tr, y_tr, X_val, y_val,
                                   max_updates=budget, n_jobs=4)
        results['TMC-Shapley'].append(np.mean((vals - true_values) ** 2))

        # Beta Shapley via pyDVL
        vals, _ = run_beta_shapley(X_tr, y_tr, X_val, y_val,
                                    alpha=1, beta=16, max_updates=budget, n_jobs=4)
        results['Beta Shapley'].append(np.mean((vals - true_values) ** 2))

        # Data Banzhaf via pyDVL
        vals, _ = run_banzhaf(X_tr, y_tr, X_val, y_val,
                               max_updates=budget, n_jobs=4)
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

    for name, t in times.items():
        print(f'  {name}: {t:.2f}s')

    plot_runtime(times, dataset_name, budget)
    return times


def run_multisemivalue(X_tr, y_tr, X_val, y_val, dataset_name, budget=10000):
    """Experiment 1.3: Multiple semivalues from one run."""
    print(f'\n--- Multi-semivalue: {dataset_name} ---')

    svarm = BetaSVARM(alpha=16, beta_param=1, adaptive=True, random_state=42)
    values_beta16, meta = svarm.fit(X_tr, y_tr, X_val, y_val, budget=budget)
    n = len(X_tr)

    semivalues = {
        'Shapley':    svarm.reweight(meta, n, 1, 1),
        'Beta(4,1)':  svarm.reweight(meta, n, 4, 1),
        'Beta(16,1)': values_beta16,
        'Banzhaf':    svarm.banzhaf_reweight(meta, n),
    }

    print('  Spearman correlations:')
    from scipy.stats import spearmanr
    for s1 in semivalues:
        for s2 in semivalues:
            if s1 < s2:
                rho, _ = spearmanr(semivalues[s1], semivalues[s2])
                print(f'    {s1} vs {s2}: ρ = {rho:.3f}')

    plot_multisemivalue(semivalues, dataset_name)


# ====================================================================
if __name__ == '__main__':
    print('=' * 60)
    print('β-SVARM Experiment Suite — Claim 1')
    print('=' * 60)

    # ----- Load datasets -----
    print('\nLoading datasets...')

    print('  Gaussian synthetic (n=15 for exact ground truth)...')
    Xg, yg, Xgv, ygv, _, _ = load_gaussian(n_train=15, n_val=200, seed=42)

    print('  Adult (n=200)...')
    Xa, ya, Xav, yav, _, _ = load_adult(n_train=200, n_val=200, seed=42)

    print('  MNIST-PCA32 (n=200)...')
    Xm, ym, Xmv, ymv, _, _ = load_mnist_pca(n_train=200, n_val=200, seed=42)

    # ----- Experiment 1.1: Convergence -----
    print('\n' + '=' * 60)
    print('EXPERIMENT 1.1: Convergence curves')
    print('=' * 60)

    run_convergence(Xg, yg, Xgv, ygv, 'Gaussian',
                    checkpoints=[50, 100, 200, 500, 1000, 2000, 3000],
                    gt_budget=10000)

    run_convergence(Xa, ya, Xav, yav, 'Adult',
                    checkpoints=[500, 1000, 2000, 3000, 5000, 8000, 10000],
                    gt_budget=30000)

    run_convergence(Xm, ym, Xmv, ymv, 'MNIST',
                    checkpoints=[500, 1000, 2000, 3000, 5000, 8000, 10000],
                    gt_budget=30000)

    # ----- Experiment 1.2: Runtime -----
    print('\n' + '=' * 60)
    print('EXPERIMENT 1.2: Runtime comparison')
    print('=' * 60)

    run_runtime(Xa, ya, Xav, yav, 'Adult', budget=5000)

    # ----- Experiment 1.3: Multi-semivalue -----
    print('\n' + '=' * 60)
    print('EXPERIMENT 1.3: Multi-semivalue output')
    print('=' * 60)

    run_multisemivalue(Xa, ya, Xav, yav, 'Adult', budget=10000)

    # ----- Done -----
    print('\n' + '=' * 60)
    print('ALL EXPERIMENTS COMPLETE!')
    print('Check results/ directory for output figures:')
    print('  convergence_Gaussian.png')
    print('  convergence_Adult.png')
    print('  convergence_MNIST.png')
    print('  runtime_Adult.png')
    print('  multisemivalue_Adult.png')
    print('=' * 60)
