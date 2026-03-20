"""
Dataset loaders. All datasets auto-download via sklearn — no manual download needed.
"""

import numpy as np
from sklearn.datasets import fetch_openml, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


def load_adult(n_train=200, n_val=200, n_test=500, seed=42):
    """Adult income dataset (binary classification)."""
    data = fetch_openml('adult', version=2, as_frame=False, parser='auto')
    X, y = data.data, data.target
    y = (y == '>50K').astype(int) if y.dtype == object else y.astype(int)
    imp = SimpleImputer(strategy='most_frequent')
    X = imp.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    rng = np.random.RandomState(seed)
    total = n_train + n_val + n_test
    idx = rng.choice(len(X), size=min(total, len(X)), replace=False)
    X, y = X[idx], y[idx]

    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, train_size=n_train, random_state=seed, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_rest, y_rest, train_size=n_val, random_state=seed, stratify=y_rest)
    return X_tr, y_tr, X_val, y_val, X_te, y_te


def load_mnist_pca(n_train=200, n_val=200, n_test=500, n_components=32, seed=42):
    """MNIST reduced to 32D via PCA, binarized (0-4 vs 5-9)."""
    data = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = data.data, data.target.astype(int)
    y = (y >= 5).astype(int)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=seed)
    X = pca.fit_transform(X)

    rng = np.random.RandomState(seed)
    total = n_train + n_val + n_test
    idx = rng.choice(len(X), size=min(total, len(X)), replace=False)
    X, y = X[idx], y[idx]

    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, train_size=n_train, random_state=seed, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_rest, y_rest, train_size=n_val, random_state=seed, stratify=y_rest)
    return X_tr, y_tr, X_val, y_val, X_te, y_te


def load_gaussian(n_train=200, n_val=200, n_test=500, n_features=20, seed=42):
    """Synthetic Gaussian classification data."""
    X, y = make_classification(
        n_samples=n_train + n_val + n_test,
        n_features=n_features, n_informative=10,
        n_redundant=5, n_classes=2, random_state=seed)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, train_size=n_train, random_state=seed, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_rest, y_rest, train_size=n_val, random_state=seed, stratify=y_rest)
    return X_tr, y_tr, X_val, y_val, X_te, y_te
