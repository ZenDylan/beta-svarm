"""Plotting utilities for β-SVARM experiments — publication-quality figures."""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ---- Publication-style matplotlib defaults (DataShapley-inspired) ----
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 13,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# ---- Color palette (colorblind-friendly, DataShapley-inspired) ----
PAPER_COLORS = {
    'β-SVARM (Ours)': '#D85A30',   # burnt orange
    'β-SVARM':        '#D85A30',
    'TMC-Shapley':    '#534AB7',   # indigo
    'Beta Shapley':   '#1D9E75',   # teal
    'Data Banzhaf':   '#378ADD',   # sky blue
    'LOO':            '#888780',   # grey
    'Random':         '#B4B2A9',   # light grey
    'Shapley':        '#534AB7',
    'Beta(16,1)':    '#D85A30',
    'Beta(4,1)':     '#1D9E75',
}

# Distinct markers and linestyles per method (B&W friendly)
_MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h']
_LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.']
_COLOR_CYCLE = [
    '#D85A30', '#534AB7', '#1D9E75', '#378ADD',
    '#888780', '#B4B2A9', '#7FBB00', '#984EA3',
]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def _style(name, idx):
    color = PAPER_COLORS.get(name, _COLOR_CYCLE[idx % len(_COLOR_CYCLE)])
    marker = _MARKERS[idx % len(_MARKERS)]
    linestyle = _LINESTYLES[idx % len(_LINESTYLES)]
    return color, marker, linestyle


def _save(fig, name):
    """Save figure as both PDF (publication) and PNG (preview)."""
    pdf_path = os.path.join(RESULTS_DIR, f'{name}.pdf')
    png_path = os.path.join(RESULTS_DIR, f'{name}.png')
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f'  Saved {pdf_path}  +  {png_path}')


# =============================================================================
# Convergence plot (MSE vs T)
# =============================================================================
def plot_convergence(budget_checkpoints, results_dict, dataset_name):
    """Log-scale MSE vs number of utility evaluations T."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (name, mses) in enumerate(results_dict.items()):
        color, marker, ls = _style(name, idx)
        x = budget_checkpoints[:len(mses)]
        ax.plot(x, mses, color=color, marker=marker, linestyle=ls,
                linewidth=2.5, markersize=7, label=name)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of utility evaluations (T)')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title(f'Convergence — {dataset_name}')
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs='auto', numticks=12))
    ax.legend(loc='upper right', framealpha=0.9,
              bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    _save(fig, f'convergence_{dataset_name}')


# =============================================================================
# Runtime bar chart
# =============================================================================
def plot_runtime(times_dict, dataset_name, budget):
    """Bar chart of wall-clock times."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(times_dict.keys())
    vals = [times_dict[n] for n in names]
    colors = [PAPER_COLORS.get(n, _COLOR_CYCLE[i % len(_COLOR_CYCLE)]) for i, n in enumerate(names)]

    bars = ax.bar(names, vals, color=colors, width=0.6,
                  edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=12)

    ax.set_ylabel('Wall-clock time (seconds)')
    ax.set_title(f'Runtime at T={budget} — {dataset_name}')
    plt.xticks(rotation=15, ha='right')
    fig.tight_layout()
    _save(fig, f'runtime_{dataset_name}')


# =============================================================================
# Multi-semivalue scatter (2×2)
# =============================================================================
def plot_multisemivalue(semivalues_dict, dataset_name):
    """2×2 Spearman correlation scatter between semivalues."""
    from scipy.stats import spearmanr

    pairs = [
        ('Shapley', 'Beta(16,1)'),
        ('Shapley', 'Banzhaf'),
        ('Beta(4,1)', 'Beta(16,1)'),
        ('Beta(4,1)', 'Banzhaf'),
    ]
    pair_colors = ['#534AB7', '#D85A30', '#1D9E75', '#378ADD']

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for idx, (s1, s2) in enumerate(pairs):
        ax = axes[idx // 2][idx % 2]
        v1, v2 = semivalues_dict[s1], semivalues_dict[s2]
        ax.scatter(v1, v2, alpha=0.6, s=25, color=pair_colors[idx],
                   edgecolors='white', linewidth=0.5)
        rho, _ = spearmanr(v1, v2)
        ax.set_xlabel(s1, fontsize=14)
        ax.set_ylabel(s2, fontsize=14)
        ax.set_title(f'ρ = {rho:.3f}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')

    fig.suptitle(f'Multi-semivalue from ONE β-SVARM run — {dataset_name}',
                 fontsize=16, y=1.01)
    fig.tight_layout()
    _save(fig, f'multisemivalue_{dataset_name}')


# =============================================================================
# Point Removal line plot
# =============================================================================
def plot_point_removal(fracs, results_dict, dataset_name):
    """
    x = fraction of training data removed (high-value points removed first),
    y = test accuracy.
    Good methods drop faster (removing valuable points hurts more).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (name, accs) in enumerate(results_dict.items()):
        color, marker, ls = _style(name, idx)
        ax.plot(fracs, accs, color=color, marker=marker, linestyle=ls,
                linewidth=2.5, markersize=7, label=name)

    ax.set_xlabel('Fraction of training data removed (%)')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title(f'Point Removal — {dataset_name}')
    ax.set_xlim(0, max(fracs))
    ax.legend(loc='upper right', framealpha=0.9,
              bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    _save(fig, f'point_removal_{dataset_name}')


# =============================================================================
# Point Addition line plot
# =============================================================================
def plot_point_addition(fracs, results_dict, dataset_name):
    """
    x = fraction of training data added (high-value points added first),
    y = test accuracy.
    Good methods rise faster (adding valuable points helps more).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (name, accs) in enumerate(results_dict.items()):
        color, marker, ls = _style(name, idx)
        ax.plot(fracs, accs, color=color, marker=marker, linestyle=ls,
                linewidth=2.5, markersize=7, label=name)

    ax.set_xlabel('Fraction of training data added (%)')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title(f'Point Addition — {dataset_name}')
    ax.set_xlim(0, max(fracs))
    ax.legend(loc='lower right', framealpha=0.9,
              bbox_to_anchor=(1.0, 0.0))
    fig.tight_layout()
    _save(fig, f'point_addition_{dataset_name}')


# =============================================================================
# Noisy Label Detection line plot
# =============================================================================
def plot_noisy_detection(fracs, results_dict, dataset_name):
    """
    x = fraction of data checked (low-value points first),
    y = noise detection rate.
    Good methods rise faster (noisy labels get low values).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (name, rates) in enumerate(results_dict.items()):
        color, marker, ls = _style(name, idx)
        ax.plot(fracs, rates, color=color, marker=marker, linestyle=ls,
                linewidth=2.5, markersize=7, label=name)

    # Random baseline: diagonal
    ax.plot(fracs, fracs, color='gray', linestyle=':', linewidth=2, label='Random')

    ax.set_xlabel('Fraction of data checked (%)')
    ax.set_ylabel('Detected noise ratio')
    ax.set_title(f'Noisy Label Detection — {dataset_name}')
    ax.set_xlim(0, max(fracs))
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', framealpha=0.9,
              bbox_to_anchor=(0.0, 1.0))
    fig.tight_layout()
    _save(fig, f'noisy_detection_{dataset_name}')
