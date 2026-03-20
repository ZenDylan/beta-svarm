"""
Experiment 1.2: Runtime comparison.

Compares wall-clock time of β-SVARM vs baselines at a fixed budget.
Run via run_claim1.py or standalone:
    python experiments/exp2_runtime.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from src.beta_svarm import BetaSVARM
from src.baselines import run_tmc_shapley, run_beta_shapley, run_banzhaf
from src.datasets import load_adult
from src.utils import plot_runtime


def run_runtime(X_tr, y_tr, X_val, y_val, dataset_name, budget=5000):
    """Run runtime comparison experiment."""
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


if __name__ == '__main__':
    print('Loading Adult dataset...')
    Xa, ya, Xav, yav, _, _ = load_adult(n_train=200, n_val=200, seed=42)
    run_runtime(Xa, ya, Xav, yav, 'Adult', budget=5000)
    print('\nDone!')
