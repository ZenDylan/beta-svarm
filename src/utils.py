"""Plotting utilities for β-SVARM experiments."""

import matplotlib.pyplot as plt
import numpy as np
import os

# Consistent style
COLORS = {
    'β-SVARM': '#D85A30',
    'TMC-Shapley': '#534AB7',
    'Beta Shapley': '#1D9E75',
    'MC-Beta-Shapley': '#1D9E75',
    'Data Banzhaf': '#378ADD',
    'LOO': '#888780',
    'Random': '#B4B2A9',
}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_convergence(budget_checkpoints, results_dict, dataset_name):
    """
    Plot MSE vs budget for all methods.

    Parameters
    ----------
    budget_checkpoints : list of int
    results_dict : dict {method_name: list of MSE values}
    dataset_name : str
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, mses in results_dict.items():
        color = COLORS.get(name, '#000000')
        ax.plot(budget_checkpoints[:len(mses)], mses, 'o-',
                label=name, color=color, linewidth=2, markersize=5)

    ax.set_xlabel('Number of utility evaluations (T)', fontsize=13)
    ax.set_ylabel('Mean Squared Error', fontsize=13)
    ax.set_title(f'Convergence — {dataset_name}', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, f'convergence_{dataset_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path}')


def plot_runtime(times_dict, dataset_name, budget):
    """Plot bar chart of wall-clock times."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(times_dict.keys())
    vals = [times_dict[n] for n in names]
    colors = [COLORS.get(n, '#888780') for n in names]

    bars = ax.bar(names, vals, color=colors, width=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('Wall-clock time (seconds)', fontsize=13)
    ax.set_title(f'Runtime at T={budget} — {dataset_name}', fontsize=14)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, f'runtime_{dataset_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path}')


def plot_multisemivalue(semivalues_dict, dataset_name):
    """Plot 2x2 scatter of different semivalues from one β-SVARM run."""
    from scipy.stats import spearmanr

    pairs = [
        ('Shapley', 'Beta(16,1)'),
        ('Shapley', 'Banzhaf'),
        ('Beta(4,1)', 'Beta(16,1)'),
        ('Beta(4,1)', 'Banzhaf'),
    ]
    scatter_colors = ['#534AB7', '#D85A30', '#1D9E75', '#378ADD']

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for idx, (s1, s2) in enumerate(pairs):
        ax = axes[idx // 2][idx % 2]
        v1 = semivalues_dict[s1]
        v2 = semivalues_dict[s2]
        ax.scatter(v1, v2, alpha=0.5, s=15, color=scatter_colors[idx])
        rho, _ = spearmanr(v1, v2)
        ax.set_xlabel(s1, fontsize=11)
        ax.set_ylabel(s2, fontsize=11)
        ax.set_title(f'{s1} vs {s2}  (ρ = {rho:.3f})', fontsize=12)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Multi-semivalue from ONE β-SVARM run — {dataset_name}',
                 fontsize=14, y=1.01)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, f'multisemivalue_{dataset_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path}')
