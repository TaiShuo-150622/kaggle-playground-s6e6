"""
Shared feature engineering for Playground S6E6
==============================================
V2: Ported from community CatBoost notebook (cat-v3, CV 0.96897).

Feature groups:
  1. Basic: color indices, alpha cyclic, categorical enc
  2. Deotte: g/redshift ratio, mag stats, log1p, floor bins, delta bins, TE
  3. Flux: magnitude→linear flux conversion + stats + ratios
  4. Color advanced: absolute diffs, ratios, curvature, statistics
  5. Redshift: boolean bins, band interactions, nonlinear transforms
  6. Sky coordinates: cartesian from spherical
  7. Categorical expanded: round bins, mod bins, frac bins, decimal bins
  8. Hash combos: pairwise + triplet interactions of categorical features
  9. Spectral proxies: color temperature, UV excess, red sequence score

Total: ~200+ features (vs ~43 before)
"""

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, TargetEncoder


# ============================================================
# 1. Basic features
# ============================================================

def add_color_indices(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """10 color indices + absolute differences."""
    BANDS = ['u', 'g', 'r', 'i', 'z']
    for df in [train, test]:
        for a, b in combinations(BANDS, 2):
            df[f'{a}_{b}'] = (df[a] - df[b]).astype('float32')
            df[f'{a}_{b}_abs'] = df[f'{a}_{b}'].abs().astype('float32')
        df['u_z'] = df['u_z'].astype('float32') if 'u_z' in df.columns else None
        df['g_z'] = df['g_z'].astype('float32') if 'g_z' in df.columns else None
        df['color_curv'] = (df['u_g'] - df['g_r']).astype('float32')
    # Ensure u_z, g_z exist
    if 'u_z' not in train.columns:
        train['u_z'] = (train['u'] - train['z']).astype('float32')
        test['u_z'] = (test['u'] - test['z']).astype('float32')
    if 'g_z' not in train.columns:
        train['g_z'] = (train['g'] - train['z']).astype('float32')
        test['g_z'] = (test['g'] - test['z']).astype('float32')
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
        train[col + '_enc'] = le.fit_transform(train[col].astype(str))
        test[col + '_enc'] = le.transform(test[col].astype(str))
    return train, test


# ============================================================
# 2. Deotte features
# ============================================================

def add_deotte_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """21 engineered features from Deotte's pipeline."""
    for df in [train, test]:
        df['_g_div_redshift'] = (df['g'] / (df['redshift'] + 1e-6)).clip(-10, 10).astype('float32')
        df['_i_div_redshift'] = (df['i'] / (df['redshift'] + 1e-6)).clip(-10, 10).astype('float32')
        mags = df[['u', 'g', 'r', 'i', 'z']].astype('float32')
        df['_mag_mean'] = mags.mean(axis=1).astype('float32')
        df['_mag_range'] = (mags.max(axis=1) - mags.min(axis=1)).astype('float32')
        df['_log1p_redshift'] = np.log1p(df['redshift'].clip(lower=0) + 1e-4).astype('float32')
    for df in [train, test]:
        for col in ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']:
            df[f'{col}_cat_'] = np.floor(df[col]).astype('int32').astype('category')
    for df in [train, test]:
        for n_bins in [100, 500]:
            kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None)
            df[f'delta_{n_bins}_bin_'] = pd.Series(
                kb.fit_transform(df[['delta']]).ravel().astype('int32'),
                index=df.index
            ).astype('category')
    return train, test


def add_target_encoding(train: pd.DataFrame, test: pd.DataFrame, y, random_state: int = 42) -> tuple:
    """Target encoding for (alpha,delta) and (u,z) combos. NOT fold-safe."""
    le = LabelEncoder(); y_enc = le.fit_transform(y)
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
# 3. Flux features (magnitude → linear flux)
# ============================================================

def add_flux_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Convert astronomical magnitudes to linear flux units.

    flux = 10^(-0.4 * mag) — standard Pogson's ratio.
    This reveals physical brightness relationships that magnitudes hide.
    """
    BANDS = ['u', 'g', 'r', 'i', 'z']
    for df in [train, test]:
        fluxes = []
        for b in BANDS:
            clipped = np.clip(df[b].values, -30, 30).astype(np.float32)
            flux = np.power(10.0, -0.4 * clipped).astype(np.float32)
            df[f'flux_{b}'] = flux
            df[f'log_flux_{b}'] = np.log1p(flux).astype(np.float32)
            fluxes.append(flux)
        fv = np.stack(fluxes, axis=1)
        df['flux_mean'] = np.nanmean(fv, axis=1).astype(np.float32)
        df['flux_std'] = np.nanstd(fv, axis=1).astype(np.float32)
        df['flux_range'] = (np.nanmax(fv, axis=1) - np.nanmin(fv, axis=1)).astype(np.float32)
        for a, b in [('u', 'g'), ('g', 'r'), ('r', 'i'), ('i', 'z'), ('u', 'r'), ('r', 'z')]:
            color = np.clip(df[a].values - df[b].values, -20, 20).astype(np.float32)
            df[f'flux_ratio_{a}_{b}'] = np.exp(-0.921034 * color).astype(np.float32)
    return train, test


# ============================================================
# 4. Advanced color + magnitude features
# ============================================================

def add_advanced_color_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Color statistics, ratios, curvature, mag statistics, slope."""
    BANDS = ['u', 'g', 'r', 'i', 'z']

    for df in [train, test]:
        # Magnitude statistics (extended)
        mag_vals = df[list(BANDS)].values.astype(np.float32)
        df['mag_mean'] = np.nanmean(mag_vals, axis=1).astype(np.float32)
        df['mag_std'] = np.nanstd(mag_vals, axis=1).astype(np.float32)
        df['mag_min'] = np.nanmin(mag_vals, axis=1).astype(np.float32)
        df['mag_max'] = np.nanmax(mag_vals, axis=1).astype(np.float32)
        df['mag_range'] = (df['mag_max'] - df['mag_min']).astype(np.float32)
        df['mag_argmin'] = np.nanargmin(mag_vals, axis=1).astype('int32')
        df['mag_argmax'] = np.nanargmax(mag_vals, axis=1).astype('int32')

        # Spectral slope (linear regression of mags vs band index)
        x = np.arange(len(BANDS), dtype=np.float32)
        x_center = x - x.mean()
        centered = mag_vals - np.nanmean(mag_vals, axis=1)[:, None]
        df['mag_slope'] = (centered.dot(x_center) / np.sum(x_center ** 2)).astype(np.float32)

        # Color curvatures (3 types)
        df['mag_curvature'] = (df['u'] - 2 * df['r'] + df['z']).astype('float32')
        df['blue_curvature'] = (df['u'] - 2 * df['g'] + df['r']).astype('float32')
        df['red_curvature'] = (df['r'] - 2 * df['i'] + df['z']).astype('float32')

        # Color ratios
        for a_name, b_name in [('u_g', 'g_r'), ('g_r', 'r_i'), ('r_i', 'i_z')]:
            a = df[a_name].values.astype(np.float32)
            b = df[b_name].values.astype(np.float32)
            df[f'{a_name}_{b_name}_ratio'] = np.where(
                np.abs(b) > 1e-8, a / b, 0.0
            ).astype('float32')

        # Color statistics
        color_cols = ['u_g', 'g_r', 'r_i', 'i_z']
        color_mat = df[color_cols].values.astype(np.float32)
        df['color_mean'] = np.nanmean(color_mat, axis=1).astype(np.float32)
        df['color_std'] = np.nanstd(color_mat, axis=1).astype(np.float32)
        df['color_abs_sum'] = np.nansum(np.abs(color_mat), axis=1).astype(np.float32)

        # Color temperature proxy (B-V → Teff approximation)
        gr = np.clip(df['g_r'].values.astype(np.float32), -5, 5)
        df['color_temp_gr_proxy'] = (4600.0 * ((1.0 / (0.92 * gr + 1.7)) +
                                                (1.0 / (0.92 * gr + 0.62)))).astype(np.float32)
        df['uv_excess_proxy'] = (df['u_g'] - (0.75 * df['g_r'] + 0.18)).astype('float32')

        # Red sequence score
        ur = np.clip(df['u_r'].values.astype(np.float32), -5, 8)
        df['red_sequence_score_proxy'] = (ur - 2.2).astype('float32')

    return train, test


# ============================================================
# 5. Redshift features
# ============================================================

def add_redshift_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Extended redshift features: boolean bins, interactions, transforms."""
    BANDS = ['u', 'g', 'r', 'i', 'z']

    for df in [train, test]:
        z = df['redshift'].values.astype(np.float32)
        z_abs = np.abs(z)

        df['redshift_abs'] = z_abs.astype('float32')
        df['redshift_log1p_abs'] = np.log1p(z_abs).astype('float32')
        df['redshift_sq'] = (z ** 2).astype('float32')
        df['redshift_cbrt'] = np.cbrt(z).astype('float32')

        # Boolean bins
        df['redshift_is_neg'] = (z < 0).astype('int32')
        df['redshift_lt_002'] = (z < 0.02).astype('int32')
        df['redshift_gt_07'] = (z > 0.7).astype('int32')

        # Band / redshift ratios
        for b in ['g', 'i', 'z']:
            df[f'{b}_over_redshift'] = np.where(
                z_abs > 1e-8, df[b].values.astype(np.float32) / z_abs, 0.0
            ).astype('float32')

        # z/g ratios
        g_abs = np.abs(df['g'].values.astype(np.float32))
        df['z_over_g'] = np.where(g_abs > 1e-8, z / g_abs, 0.0).astype('float32')
        df['z2_over_g2'] = np.where(g_abs > 1e-8, (z ** 2) / (g_abs ** 2), 0.0).astype('float32')
        log_z = np.log1p(z_abs)
        log_g = np.log1p(g_abs)
        df['log_z_over_log_g'] = np.where(log_g > 1e-8, log_z / log_g, 0.0).astype('float32')
        sqrt_z = np.sqrt(z_abs)
        sqrt_g = np.sqrt(g_abs)
        df['sqrt_z_over_sqrt_g'] = np.where(sqrt_g > 1e-8, sqrt_z / sqrt_g, 0.0).astype('float32')

        # Redshift × band interactions
        for b in BANDS:
            df[f'redshift_x_{b}'] = (df['redshift'] * df[b]).astype('float32')

    return train, test


# ============================================================
# 6. Sky coordinates (spherical → cartesian)
# ============================================================

def add_sky_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Convert (alpha, delta) spherical to cartesian sky coordinates."""
    for df in [train, test]:
        alpha_rad = np.deg2rad(df['alpha'].values.astype(np.float32))
        delta_rad = np.deg2rad(df['delta'].values.astype(np.float32))

        df['delta_sin'] = np.sin(delta_rad).astype('float32')
        df['delta_cos'] = np.cos(delta_rad).astype('float32')
        df['sky_x'] = (np.cos(delta_rad) * np.cos(alpha_rad)).astype('float32')
        df['sky_y'] = (np.cos(delta_rad) * np.sin(alpha_rad)).astype('float32')
        df['sky_z'] = np.sin(delta_rad).astype('float32')

    return train, test


# ============================================================
# 7. Expanded categorical features (round/mod/frac/decimal bins)
# ============================================================

def _safe_quantile_bin(df, col, n_bins, name):
    """Quantile binning, handling edge cases."""
    vals = df[col].values.astype(np.float32)
    valid = np.isfinite(vals)
    if valid.sum() <= 1:
        df[name] = -1
        return
    try:
        df[name] = pd.qcut(df[col], n_bins, labels=False, duplicates='drop').fillna(-1).astype('int32')
    except Exception:
        df[name] = -1


def add_expanded_categoricals(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Round, mod, frac, decimal bins — ported from cat-v3 GPU code."""
    RAW_NUM = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']
    BANDS = ['u', 'g', 'r', 'i', 'z']

    # Quantile bin specs
    q_specs = {}
    for c in RAW_NUM:
        q_specs[c] = [32, 100, 500]
    for c in ['u_g', 'g_r', 'r_i', 'i_z', 'u_r', 'r_z', 'mag_range', 'mag_std']:
        if c in train.columns:
            q_specs[c] = [64]

    all_data = pd.concat([train, test], ignore_index=True)
    n_train = len(train)

    for c, bins_list in q_specs.items():
        if c not in all_data.columns:
            continue
        for n_bins in bins_list:
            name = f'{c}_q{n_bins}_cat'
            _safe_quantile_bin(all_data, c, n_bins, name)

    # Round bins (round to N decimals)
    round_specs = {
        'alpha': 1, 'delta': 1, 'u': 2, 'g': 2, 'r': 2, 'i': 2, 'z': 2,
        'redshift': 4, 'u_g': 3, 'g_r': 3, 'r_i': 3, 'i_z': 3,
        'u_r': 3, 'r_z': 3, 'mag_range': 3,
    }
    for c, dec in round_specs.items():
        if c in all_data.columns:
            name = f'{c}_round{dec}_cat'
            all_data[name] = np.rint(all_data[c].values * (10 ** dec)).astype('int32')

    # Mod and frac bins for raw numerics
    for c in RAW_NUM:
        if c not in all_data.columns:
            continue
        vals = np.abs(all_data[c].values.astype(np.float32))
        vals = np.nan_to_num(vals, nan=0.0)
        int_part = np.floor(vals).astype(np.int64)
        frac_part = vals - np.floor(vals)

        all_data[f'{c}_mod10_cat'] = (int_part % 10).astype('int32')
        all_data[f'{c}_mod100_cat'] = (int_part % 100).astype('int32')
        all_data[f'{c}_frac20_cat'] = np.floor(frac_part * 20).astype('int32')
        all_data[f'{c}_decimal1000_cat'] = np.floor(frac_part * 1000).astype('int32')

    # Split back
    train = all_data.iloc[:n_train].copy()
    test = all_data.iloc[n_train:].copy()

    return train, test


# ============================================================
# 8. Hash combo features
# ============================================================

def _hash_combo(df, cols, name):
    """Simple integer hash of categorical column combinations."""
    combined = df[cols[0]].astype(str).fillna('NA')
    for c in cols[1:]:
        combined = combined + '_' + df[c].astype(str).fillna('NA')
    df[name] = pd.factorize(combined)[0].astype('int32')


def add_hash_combos(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Pairwise and triplet feature interactions via hashing.

    Uses factorize() instead of prime-based hash (equivalent, simpler).
    """
    all_data = pd.concat([train, test], ignore_index=True)
    n_train = len(train)

    # Identify categorical bases for combos
    cat_candidates = [c for c in all_data.columns
                      if any(c.endswith(s) for s in ['_cat', '_cat_', '_bin_'])
                      or c in ['spectral_type_enc', 'galaxy_population_enc']]
    cat_candidates = list(dict.fromkeys(cat_candidates))  # dedupe

    # Manual combos (from cat-v3)
    manual_pairs = [
        ('alpha_cat_', 'delta_cat_'),
        ('u_cat_', 'z_cat_'),
    ]
    # Add quantile-based pairs if available
    extra_pairs = [
        ('alpha_q100_cat', 'delta_q100_cat'),
        ('u_q100_cat', 'z_q100_cat'),
    ]
    for a, b in manual_pairs + extra_pairs:
        if a in all_data.columns and b in all_data.columns:
            _hash_combo(all_data, [a, b], f'COMBO_{a}__{b}')

    # Combinatorial pairs from top category bases
    bases = [c for c in cat_candidates[:20] if c in all_data.columns][:10]
    for a, b in combinations(bases, 2):
        _hash_combo(all_data, [a, b], f'PAIR_{a}__{b}')

    # Triplet combos (group of 3)
    for i in range(0, min(len(bases), 9) - 2, 3):
        trio = bases[i:i + 3]
        if len(trio) == 3:
            _hash_combo(all_data, trio, 'TRIO_' + '__'.join(trio))

    train = all_data.iloc[:n_train].copy()
    test = all_data.iloc[n_train:].copy()
    return train, test


# ============================================================
# 9. Spectral/population proxy features
# ============================================================

def add_spectral_proxies(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Recalculate spectral_type and galaxy_population from photometry."""
    for df in [train, test]:
        rg = df['r'].values - df['g'].values
        df['spectral_type_calc'] = np.select(
            [rg <= -1.0, rg <= -0.5, rg <= 0.0],
            [0, 1, 2], default=3
        ).astype('int32')

        ur = df['u'].values - df['r'].values
        df['galaxy_population_calc'] = np.where(ur <= 2.2, 0, 1).astype('int32')

        # Interaction
        if 'spectral_type_enc' in df.columns:
            df['spectral_x_pop'] = (
                df['spectral_type_enc'].fillna(-1).astype('int32') * 10 +
                df['galaxy_population_enc'].fillna(-1).astype('int32')
            ).astype('int32')
        df['spectral_calc_x_pop_calc'] = (
            df['spectral_type_calc'] * 10 + df['galaxy_population_calc']
        ).astype('int32')

    return train, test


# ============================================================
# 10. Pipeline: run all features
# ============================================================

def engineer_all(train: pd.DataFrame, test: pd.DataFrame, y,
                 include_advanced: bool = True, random_state: int = 42) -> tuple:
    """Run complete feature engineering pipeline.

    Args:
        train, test: raw DataFrames with columns [id, alpha, delta, u, g, r, i, z,
                      redshift, spectral_type, galaxy_population, class?]
        y: training labels
        include_advanced: include flux, color, hash, redshift advanced features
        random_state: seed

    Returns:
        (train_df, test_df, feature_list)
    """
    # Phase 1: Basic (always)
    train, test = add_color_indices(train, test)
    train, test = add_alpha_cyclic(train, test)
    train, test = add_categorical_enc(train, test)

    # Phase 2: Deotte features (always)
    train, test = add_deotte_features(train, test)
    train, test = add_target_encoding(train, test, y, random_state)

    if include_advanced:
        # Phase 3: Flux
        train, test = add_flux_features(train, test)

        # Phase 4: Advanced color + magnitude
        train, test = add_advanced_color_features(train, test)

        # Phase 5: Redshift extended
        train, test = add_redshift_features(train, test)

        # Phase 6: Sky coordinates
        train, test = add_sky_features(train, test)

        # Phase 7: Expanded categoricals (quantile, round, mod, frac)
        train, test = add_expanded_categoricals(train, test)

        # Phase 8: Spectral proxies
        train, test = add_spectral_proxies(train, test)

        # Phase 9: Hash combos (slowest — do it last)
        train, test = add_hash_combos(train, test)

    # Build feature list
    exclude = ['id', 'class', 'target', 'spectral_type', 'galaxy_population']
    feature_list = [c for c in train.columns
                    if c not in exclude
                    and not train[c].dtype == 'object']
    feature_list = list(dict.fromkeys(feature_list))

    return train, test, feature_list
