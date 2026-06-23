"""
Shared feature engineering for Playground S6E6
==============================================
Single source of truth. Import from all training/ensemble scripts.
Eliminates ~200 lines of duplicated code across 4+ files.

Usage:
    from code.features.shared import add_basic_features, add_deotte_features, add_target_encoding

Design decisions:
    - Feature names use Deotte convention (_, _-a-b, _TE_ci)
    - Target encoding uses cv=5 internally but is still done BEFORE model CV
      (fold-safe TE is a TODO — see EXPERIMENTS.md direction A)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, TargetEncoder


# ============================================================
# 1. Basic features
# ============================================================

def add_color_indices(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Add 10 color indices: pairwise band differences + color curvature."""
    pairs = [('u', 'g'), ('g', 'r'), ('r', 'i'), ('i', 'z'),
             ('u', 'r'), ('g', 'i'), ('r', 'z')]
    for df in [train, test]:
        for a, b in pairs:
            df[f'{a}_{b}'] = (df[a] - df[b]).astype('float32')
        df['u_z'] = (df['u'] - df['z']).astype('float32')
        df['g_z'] = (df['g'] - df['z']).astype('float32')
        df['color_curv'] = (df['u_g'] - df['g_r']).astype('float32')
    return train, test


def add_alpha_cyclic(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Cyclic encoding for right ascension (alpha in degrees)."""
    for df in [train, test]:
        rad = np.deg2rad(df['alpha'])
        df['alpha_sin'] = np.sin(rad).astype('float32')
        df['alpha_cos'] = np.cos(rad).astype('float32')
    return train, test


def add_categorical_enc(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Label-encode spectral_type and galaxy_population."""
    for col in ['spectral_type', 'galaxy_population']:
        le = LabelEncoder()
        train[col + '_enc'] = le.fit_transform(train[col])
        test[col + '_enc'] = le.transform(test[col])
    return train, test


# ============================================================
# 2. Deotte features (from Chris Deotte's S6E6 notebook)
# ============================================================

def add_deotte_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Add 21 engineered features from Deotte's pipeline.

    Includes: g/redshift ratio, i/redshift ratio, mag stats,
    log1p redshift, floor-binned categories, delta quantile bins.
    """
    for df in [train, test]:
        # Redshift ratios (clip to avoid division-by-zero explosion)
        df['_g_div_redshift'] = (df['g'] / (df['redshift'] + 1e-6)).clip(-10, 10).astype('float32')
        df['_i_div_redshift'] = (df['i'] / (df['redshift'] + 1e-6)).clip(-10, 10).astype('float32')

        # Magnitude statistics
        mags = df[['u', 'g', 'r', 'i', 'z']].astype('float32')
        df['_mag_mean'] = mags.mean(axis=1).astype('float32')
        df['_mag_range'] = (mags.max(axis=1) - mags.min(axis=1)).astype('float32')

        # Log-redshift
        df['_log1p_redshift'] = np.log1p(df['redshift'].clip(lower=0) + 1e-4).astype('float32')

    # Floor-binned categorical features
    for df in [train, test]:
        for col in ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']:
            df[f'{col}_cat_'] = np.floor(df[col]).astype('int32').astype('category')

    # Delta quantile binning (100 and 500 bins)
    for df in [train, test]:
        for n_bins in [100, 500]:
            kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal',
                                  strategy='quantile', subsample=None)
            vals = kb.fit_transform(df[['delta']]).ravel().astype('int32')
            df[f'delta_{n_bins}_bin_'] = pd.Series(vals, index=df.index).astype('category')

    return train, test


# ============================================================
# 3. Target encoding (NOT fold-safe — to be fixed in direction A)
# ============================================================

def add_target_encoding(train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray,
                        random_state: int = 42) -> tuple:
    """Add target encoding for (alpha,delta) and (u,z) combos.

    Uses internal cv=5 in TargetEncoder but is applied BEFORE model CV splits.
    This gives slightly optimistic OOF estimates (~+0.001).
    Fix coming in direction A (fold-safe TE re-computed inside each CV fold).
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    for df in [train, test]:
        df['_ca'] = df['alpha_cat_'].astype(str) + '|' + df['delta_cat_'].astype(str)
        df['_cz'] = df['u_cat_'].astype(str) + '|' + df['z_cat_'].astype(str)

    for combo in ['_ca', '_cz']:
        for ci in range(3):
            col_name = f'{combo}_TE_{ci}'
            yb = (y_enc == ci).astype(int)

            te = TargetEncoder(cv=5, smooth='auto', random_state=random_state)
            train[col_name] = te.fit_transform(train[[combo]], yb).ravel()

            te2 = TargetEncoder(smooth='auto', random_state=random_state)
            te2.fit(train[[combo]], yb)
            test[col_name] = te2.transform(test[[combo]]).ravel()

    train.drop(['_ca', '_cz'], axis=1, inplace=True)
    test.drop(['_ca', '_cz'], axis=1, inplace=True)
    return train, test


# ============================================================
# 4. Pipeline: run all features at once
# ============================================================

def engineer_all(train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray,
                 include_deotte: bool = True, random_state: int = 42) -> tuple:
    """Run complete feature engineering pipeline.

    Args:
        train, test: raw DataFrames
        y: training labels (for target encoding)
        include_deotte: set False to skip Deotte features (for A/B tests)
        random_state: seed for target encoding

    Returns:
        (train_df, test_df, feature_list)
    """
    train, test = add_color_indices(train, test)
    train, test = add_alpha_cyclic(train, test)
    train, test = add_categorical_enc(train, test)

    if include_deotte:
        train, test = add_deotte_features(train, test)
        train, test = add_target_encoding(train, test, y, random_state)

    # Build ordered feature list
    base = ['u', 'g', 'r', 'i', 'z', 'redshift', 'alpha', 'delta',
            'u_g', 'g_r', 'r_i', 'i_z', 'u_r', 'g_i', 'r_z',
            'color_curv', 'u_z', 'g_z', 'alpha_sin', 'alpha_cos',
            'spectral_type_enc', 'galaxy_population_enc']

    new_cols = [c for c in train.columns
                if c.startswith('_') or '_TE_' in c or '_cat_' in c or '_bin_' in c]
    feature_list = base + new_cols
    feature_list = [c for c in feature_list if c in train.columns]
    feature_list = list(dict.fromkeys(feature_list))  # dedupe, keep order

    return train, test, feature_list
