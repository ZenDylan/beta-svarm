"""
Baseline data valuation methods — all via pyDVL.
We only implement β-SVARM ourselves; everything else uses the library.

pyDVL v0.10+ API:
  Dataset → SupervisedScorer → ModelUtility → Valuation.fit(data)
"""

import os
os.environ['JOBLIB_TEMP_FOLDER'] = 'C:/tmp'

import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import parallel_config

from pydvl.valuation import (
    Dataset,
    ModelUtility,
    SupervisedScorer,
    # Valuation methods
    TMCShapleyValuation,
    BetaShapleyValuation,
    BanzhafValuation,
    LOOValuation,
    DataOOBValuation,
    # Samplers
    PermutationSampler,
    UniformSampler,
    MSRSampler,
    # Stopping criteria
    MaxUpdates,
    MinUpdates,
)


def _make_utility(model, X_train, y_train, X_val, y_val):
    """Create pyDVL Dataset + ModelUtility from raw arrays."""
    train = Dataset(X_train, y_train)
    test = Dataset(X_val, y_val)
    scorer = SupervisedScorer("accuracy", test, default=0.0)
    utility = ModelUtility(model, scorer)
    return train, utility


def run_tmc_shapley(X_train, y_train, X_val, y_val, max_updates=1000,
                    model=None, n_jobs=1, seed=42):
    """TMC-Shapley (Ghorbani & Zou, ICML 2019) via pyDVL."""
    if model is None:
        model = LogisticRegression(max_iter=500, random_state=seed)
    train, utility = _make_utility(model, X_train, y_train, X_val, y_val)

    valuation = TMCShapleyValuation(
        utility=utility,
        is_done=MaxUpdates(max_updates),
        seed=seed,
    )
    with parallel_config(n_jobs=1):
        valuation.fit(train)
    result = valuation.result
    # result.values is indexed by training point index
    return result.values, result


def run_beta_shapley(X_train, y_train, X_val, y_val, alpha=1, beta=16,
                     max_updates=1000, model=None, n_jobs=1, seed=42):
    """Beta Shapley (Kwon & Zou, AISTATS 2022) via pyDVL."""
    if model is None:
        model = LogisticRegression(max_iter=500, random_state=seed)
    train, utility = _make_utility(model, X_train, y_train, X_val, y_val)

    sampler = PermutationSampler(seed=seed)
    valuation = BetaShapleyValuation(
        utility=utility,
        sampler=sampler,
        is_done=MaxUpdates(max_updates),
        alpha=alpha,
        beta=beta,
    )
    with parallel_config(n_jobs=1):
        valuation.fit(train)
    result = valuation.result
    return result.values, result


def run_banzhaf(X_train, y_train, X_val, y_val, max_updates=1000,
                model=None, n_jobs=1, seed=42):
    """Data Banzhaf with MSR (Wang & Jia, AISTATS 2023) via pyDVL."""
    if model is None:
        model = LogisticRegression(max_iter=500, random_state=seed)
    train, utility = _make_utility(model, X_train, y_train, X_val, y_val)

    sampler = MSRSampler(seed=seed)
    valuation = BanzhafValuation(
        utility=utility,
        sampler=sampler,
        is_done=MaxUpdates(max_updates),
    )
    with parallel_config(n_jobs=1):
        valuation.fit(train)
    result = valuation.result
    return result.values, result


def run_loo(X_train, y_train, X_val, y_val, model=None, n_jobs=1, seed=42):
    """Leave-One-Out valuation via pyDVL."""
    if model is None:
        model = LogisticRegression(max_iter=500, random_state=seed)
    train, utility = _make_utility(model, X_train, y_train, X_val, y_val)

    valuation = LOOValuation(utility=utility, progress=False)
    with parallel_config(n_jobs=1):
        valuation.fit(train)
    result = valuation.result
    return result.values, result


def run_random(n_train, seed=42):
    """Random baseline — just random values."""
    rng = np.random.RandomState(seed)
    return rng.randn(n_train), None
