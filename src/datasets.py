"""
Dataset loaders. All datasets auto-download via sklearn — no manual download needed.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, make_classification
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def _binary_from_positive_label(y, positive_label):
    """Convert a label array to {0, 1} using the given positive label."""
    return (np.asarray(y).astype(str) == str(positive_label)).astype(int)


def _binary_from_two_classes(y):
    """Convert a binary target with arbitrary labels to {0, 1}."""
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f'Expected a binary target, got {len(classes)} classes.')
    return (y == classes[1]).astype(int)


def _select_balanced_subset(X, y, total, seed):
    """Sample a balanced binary subset of the requested size."""
    if total > len(y):
        raise ValueError(f'Requested {total} samples, but only {len(y)} are available.')

    y = np.asarray(y)
    class0_idx = np.flatnonzero(y == 0)
    class1_idx = np.flatnonzero(y == 1)
    n_class1 = total // 2
    n_class0 = total - n_class1

    if len(class0_idx) < n_class0 or len(class1_idx) < n_class1:
        raise ValueError('Not enough samples to build a balanced subset of the requested size.')

    rng = np.random.RandomState(seed)
    indices = np.concatenate([
        rng.choice(class0_idx, size=n_class0, replace=False),
        rng.choice(class1_idx, size=n_class1, replace=False),
    ])
    rng.shuffle(indices)

    if hasattr(X, 'iloc'):
        X = X.iloc[indices].reset_index(drop=True)
    else:
        X = X[indices]
    y = y[indices]
    return X, y


def _sample_and_split(X, y, n_train, n_val, n_test, seed, balanced_subset=False):
    """Sample the requested total size and split into train/val/test."""
    total = n_train + n_val + n_test
    if total > len(y):
        raise ValueError(f'Requested {total} samples, but only {len(y)} are available.')

    if balanced_subset:
        X_sel, y_sel = _select_balanced_subset(X, y, total, seed)
    elif total < len(y):
        X_sel, _, y_sel, _ = train_test_split(
            X, y, train_size=total, random_state=seed, stratify=y)
    else:
        X_sel, y_sel = X, np.asarray(y)

    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X_sel, y_sel, train_size=n_train, random_state=seed, stratify=y_sel)
    X_val, X_te, y_val, y_te = train_test_split(
        X_rest, y_rest, train_size=n_val, random_state=seed, stratify=y_rest)
    return X_tr, y_tr, X_val, y_val, X_te, y_te


def _scale_numeric_splits(X_tr, X_val, X_te):
    """Fit StandardScaler on train and apply to all splits."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te = scaler.transform(X_te)
    return X_tr, X_val, X_te


def _load_numeric_openml(name, target_transform, n_train=200, n_val=200,
                        n_test=500, seed=42, balanced_subset=False, version='active'):
    """Load an all-numeric OpenML dataset and return scaled train/val/test splits."""
    data = fetch_openml(name, version=version, as_frame=False, parser='auto')
    X = np.asarray(data.data, dtype=np.float64)
    y = target_transform(data.target)
    X_tr, y_tr, X_val, y_val, X_te, y_te = _sample_and_split(
        X, y, n_train, n_val, n_test, seed, balanced_subset=balanced_subset)
    X_tr, X_val, X_te = _scale_numeric_splits(X_tr, X_val, X_te)
    return X_tr, y_tr, X_val, y_val, X_te, y_te


def load_adult(n_train=200, n_val=200, n_test=500, seed=42):
    """Adult income dataset (binary classification)."""
    data = fetch_openml('adult', version=2, as_frame=True, parser='auto')
    X = data.data
    y = _binary_from_positive_label(data.target, '>50K')

    X_tr_raw, y_tr, X_val_raw, y_val, X_te_raw, y_te = _sample_and_split(
        X, y, n_train, n_val, n_test, seed)

    numeric_cols = X_tr_raw.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X_tr_raw.columns if col not in numeric_cols]

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
        ]), numeric_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ]), categorical_cols),
    ])

    X_tr = preprocessor.fit_transform(X_tr_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_te = preprocessor.transform(X_te_raw)
    X_tr, X_val, X_te = _scale_numeric_splits(X_tr, X_val, X_te)
    return X_tr, y_tr, X_val, y_val, X_te, y_te


def load_2dplanes(n_train=200, n_val=200, n_test=500, seed=42):
    """2dplanes dataset binarized by the median regression target."""
    def target_transform(y):
        y = np.asarray(y, dtype=np.float64)
        return (y >= np.median(y)).astype(int)

    return _load_numeric_openml(
        '2dplanes', target_transform, n_train=n_train, n_val=n_val,
        n_test=n_test, seed=seed, version=1)


def load_covertype(n_train=200, n_val=200, n_test=500, seed=42):
    """Covertype dataset (class 1 vs rest)."""
    return _load_numeric_openml(
        'covertype',
        lambda y: _binary_from_positive_label(y, '1'),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=seed,
        version=3,
    )


def load_phoneme(n_train=200, n_val=200, n_test=500, seed=42):
    """Phoneme dataset (binary classification)."""
    return _load_numeric_openml(
        'phoneme', _binary_from_two_classes, n_train=n_train, n_val=n_val,
        n_test=n_test, seed=seed, version=1)


def load_creditcard(n_train=200, n_val=200, n_test=500, seed=42):
    """Credit card fraud dataset with a balanced subset due to extreme imbalance."""
    return _load_numeric_openml(
        'creditcard',
        lambda y: _binary_from_positive_label(y, '1'),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=seed,
        balanced_subset=True,
    )


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


def load_fashion_mnist_pca(n_train=200, n_val=200, n_test=500,
                           n_components=32, seed=42):
    """Fashion-MNIST reduced to 32D via PCA, binarized (0-4 vs 5-9)."""
    data = fetch_openml('Fashion-MNIST', as_frame=False, parser='auto')
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
