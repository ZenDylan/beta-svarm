"""Plotting utilities for β-SVARM experiments — publication-quality figures."""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ---- Publication-style matplotlib defaults ----
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ---- Color palette (colorblind-friendly, inspired by β-SVARM paper) ----
PAPER_COLORS = {
    'β-SVARM':      '#D85A30',   # burnt orange
    'TMC-Shapley':  '#534AB7',   # indigo
    'Beta Shapley': '#1D9E75',   # teal
    'Data Banzhaf': '#378ADD',   # sky blue
    'LOO':          '#888780',   # grey
    'Random':       '#B4B2A9',   # light grey
    'Shapley':      '#534AB7',   # same as TMC-Shapley (purple)
    'Beta(16,1)':   '#D85A30',   # same as β-SVARM (orange)
    'Beta(4,1)':    '#1D9E75',   # same as Beta Shapley (teal)
}

# ---- Marker and linestyle cycle (different per method for B&W compatibility) ----
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h']
LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.']
# Default color cycler (overridden by explicit PAPERS_COLORS)
COLORS_CYCLE = [
    '#D85A30', '#534AB7', '#1D9E75', '#378ADD',
    '#888780', '#B4B2A9', '#7FBB00', '#984EA3',
]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def _get_style(name, idx):
    """Return (color, marker, linestyle) for a method name."""
    color = PAPER_COLORS.get(name, COLORS_CYCLE[idx % len(COLORS_CYCLE)])
    marker = MARKERS[idx % len(MARKERS)]
    linestyle = LINESTYLES[idx % len(LINESTYLES)]
    return color, marker, linestyle


# =============================================================================
# Convergence plot: MSE vs budget (log-scale)
# =============================================================================
def plot_convergence(budget_checkpoints, results_dict, dataset_name):
    """
    Plot MSE vs budget (log-scale) for all methods.

    Parameters
    ----------
    budget_checkpoints : list of int
    results_dict : dict {method_name: list of MSE values}
    dataset_name : str
    """
    n_methods = len(results_dict)
    fig, ax = plt.subplots(figsize=(4, 3))  # single-column: 4×3 inch

    for idx, (name, mses) in enumerate(results_dict.items()):
        color, marker, linestyle = _get_style(name, idx)
        x = budget_checkpoints[:len(mses)]
        ax.plot(x, mses,
                color=color, marker=marker, linestyle=linestyle,
                linewidth=1.5, markersize=5, label=name)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of utility evaluations (T)')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title(f'Convergence — {dataset_name}')
    ax.grid(True, which='both', alpha=0.3, linewidth=0.5)
    ax.legend(loc='upper right', framealpha=0.9)
    # Add minor ticks
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs='auto', numticks=12))
    ax.tick_params(axis='both', which='major', length=4)
    ax.tick_params(axis='both', which='minor', length=2)

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f'convergence_{dataset_name}.pdf')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f'  Saved {path}')


# =============================================================================
# Runtime bar chart
# =============================================================================
def plot_runtime(times_dict, dataset_name, budget):
    """
    Bar chart of wall-clock times at a fixed budget.

    Parameters
    ----------
    times_dict : dict {method_name: seconds}
    dataset_name : str
    budget : int
    """
    names = list(times_dict.keys())
    vals = [times_dict[n] for n in names]
    colors = [PAPER_COLORS.get(n, COLORS_CYCLE[i % len(COLORS_CYCLE)]) for i, n in enumerate(names)]

    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(names, vals, color=colors, width=0.6, edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('Wall-clock time (seconds)')
    ax.set_title(f'Runtime at T={budget} — {dataset_name}')
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Rotate x labels to avoid overlap
    plt.xticks(rotation=15, ha='right')

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f'runtime_{dataset_name}.pdf')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f'  Saved {path}')


# =============================================================================
# Multi-semivalue scatter (2×2)
# =============================================================================
def plot_multisemivalue(semivalues_dict, dataset_name):
    """
    2×2 scatter matrix comparing different semivalues from one β-SVARM run.

    Parameters
    ----------
    semivalues_dict : dict {semivalue_name: np.array of values}
    dataset_name : str
    """
    from scipy.stats import spearmanr

    pairs = [
        ('Shapley', 'Beta(16,1)'),
        ('Shapley', 'Banzhaf'),
        ('Beta(4,1)', 'Beta(16,1)'),
        ('Beta(4,1)', 'Banzhaf'),
    ]
    # Colors per pair (not per method — each subplot has 2 methods)
    pair_colors = ['#534AB7', '#D85A30', '#1D9E75', '#378ADD']

    fig, axes = plt.subplots(2, 2, figsize=(7, 6))  # double-column: 7×6

    for idx, (s1, s2) in enumerate(pairs):
        ax = axes[idx // 2][idx % 2]
        v1 = semivalues_dict[s1]
        v2 = semivalues_dict[s2]
        color = pair_colors[idx]

        ax.scatter(v1, v2, alpha=0.6, s=20, color=color, edgecolors='white', linewidth=0.3)
        rho, _ = spearmanr(v1, v2)

        ax.set_xlabel(s1, fontsize=12)
        ax.set_ylabel(s2, fontsize=12)
        ax.set_title(f'ρ = {rho:.3f}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        # Equal aspect so diagonal is meaningful
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')

    fig.suptitle(f'Multi-semivalue from ONE β-SVARM run — {dataset_name}',
                 fontsize=13, y=1.01)
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, f'multisemivalue_{dataset_name}.pdf')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f'  Saved {path}')
