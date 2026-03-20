"""
Experiment 1.3: Multi-semivalue output from one β-SVARM run.

Demonstrates that β-SVARM can produce Shapley, Beta(4,1), Beta(16,1),
and Banzhaf values from a single run by reweighting stratum estimates.

Run via run_claim1.py or standalone:
    python experiments/exp3_multisemivalue.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.stats import spearmanr
from src.beta_svarm import BetaSVARM
from src.datasets import load_adult
from src.utils import plot_multisemivalue


def run_multisemivalue(X_tr, y_tr, X_val, y_val, dataset_name, budget=10000):
    """Run multi-semivalue experiment."""
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
    for s1 in semivalues:
        for s2 in semivalues:
            if s1 < s2:
                rho, _ = spearmanr(semivalues[s1], semivalues[s2])
                print(f'    {s1} vs {s2}: ρ = {rho:.3f}')

    plot_multisemivalue(semivalues, dataset_name)


if __name__ == '__main__':
    print('Loading Adult dataset...')
    Xa, ya, Xav, yav, _, _ = load_adult(n_train=200, n_val=200, seed=42)
    run_multisemivalue(Xa, ya, Xav, yav, 'Adult', budget=10000)
    print('\nDone!')
