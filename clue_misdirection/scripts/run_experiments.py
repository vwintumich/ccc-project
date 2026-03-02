"""Run all four classification experiments (1A, 1B, 2A, 2B) on Great Lakes.

This is a standalone script that mirrors the training logic from
NB 06 (06_experiments_easy.ipynb) and NB 07 (07_experiments_harder.ipynb).
It runs both the easy and harder dataset experiments in a single job,
saving results to the outputs/ directory.

Usage:
    # Full run (all data, full hyperparameter grids):
    python scripts/run_experiments.py

    # Sample mode (20K rows per dataset, reduced grids, for testing):
    python scripts/run_experiments.py --sample

The script must be run from the project root (clue_misdirection/).

Feature sets (from feature_utils.py):
    Exp 1A: ALL_FEATURE_COLS (47)          — easy dataset, all features
    Exp 1B: ALL - CONTEXT_INFORMED (41)    — easy dataset, context removed
    Exp 2A: HARDER_FEATURE_COLS (32)       — harder dataset, all non-context-free
    Exp 2B: HARDER - CONTEXT_INFORMED (26) — harder dataset, context removed

See PLAN.md Steps 6 and 8, design doc Section 8.3.
"""

import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, GroupKFold,
                                     RandomizedSearchCV, StratifiedKFold)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================
# Path setup
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import feature column lists (Decision 18)
sys.path.insert(0, str(SCRIPTS_DIR))
from feature_utils import (
    ALL_FEATURE_COLS,
    CONTEXT_INFORMED_COLS,
    RELATIONSHIP_COLS,
    SURFACE_COLS,
)

# ============================================================
# Constants
# ============================================================
RANDOM_SEED = 42
N_FOLDS = 5
SAMPLE_SIZE = 20_000

# --- Feature sets ---
# Easy dataset: 47 features (Exp 1A) and 41 features (Exp 1B)
EXP_1B_COLS = [c for c in ALL_FEATURE_COLS if c not in CONTEXT_INFORMED_COLS]

# Harder dataset: 32 features (Exp 2A) and 26 features (Exp 2B)
# The 15 context-free cosine features have been removed because they are
# artifacts of the cosine-similarity-based distractor construction
# (Decision 6).
HARDER_FEATURE_COLS = (
    CONTEXT_INFORMED_COLS + list(RELATIONSHIP_COLS) + SURFACE_COLS
)
EXP_2B_COLS = [c for c in HARDER_FEATURE_COLS
               if c not in CONTEXT_INFORMED_COLS]


def timestamp():
    """Return a formatted timestamp for log output."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def build_model_configs(sample_mode):
    """Build model configurations with appropriate hyperparameter grids.

    Parameters
    ----------
    sample_mode : bool
        If True, use reduced grids for fast iteration.
        If False, use full grids for final runs.

    Returns
    -------
    dict
        Model configurations for KNN, Logistic Regression, Random Forest.
    """
    if sample_mode:
        knn_grid = {
            'n_neighbors': [3, 7, 15],
            'weights': ['uniform', 'distance'],
        }
        logreg_grid = {
            'C': [0.1, 1.0, 10.0],
            'l1_ratio': [0.0, 0.5, 1.0],
        }
        rf_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
        }
        rf_n_iter = 10
    else:
        knn_grid = {
            'n_neighbors': [3, 5, 7, 11, 15, 21],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan'],
        }
        logreg_grid = {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'l1_ratio': [0.0, 0.5, 1.0],
        }
        rf_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
        }
        rf_n_iter = 20

    model_configs = {
        'KNN': {
            'estimator': KNeighborsClassifier(),
            'param_grid': knn_grid,
            'search': 'grid',
            'scale': True,
        },
        'Logistic Regression': {
            'estimator': LogisticRegression(
                solver='saga', penalty='elasticnet',
                max_iter=5000, random_state=RANDOM_SEED),
            'param_grid': logreg_grid,
            'search': 'grid',
            'scale': True,
        },
        'Random Forest': {
            'estimator': RandomForestClassifier(random_state=RANDOM_SEED),
            'param_grid': rf_grid,
            'search': 'random',
            'scale': False,
            'n_iter': rf_n_iter,
        },
    }

    return model_configs


def load_and_prepare(parquet_path, feature_cols, sample_mode):
    """Load a parquet dataset, validate, optionally subsample, assign folds.

    Parameters
    ----------
    parquet_path : Path
        Path to the dataset parquet file.
    feature_cols : list of str
        Feature columns that must be present.
    sample_mode : bool
        If True, stratified subsample to SAMPLE_SIZE rows.

    Returns
    -------
    pd.DataFrame
        Dataset with 'fold' column assigned via GroupKFold.
    """
    assert parquet_path.exists(), (
        f'Missing input file: {parquet_path}\n'
        f'Run 05_dataset_construction.ipynb first.'
    )

    df = pd.read_parquet(parquet_path)
    print(f'[{timestamp()}] Loaded {parquet_path.name}: '
          f'{len(df):,} rows x {len(df.columns)} columns')

    # Validate expected columns
    missing_feat = [c for c in feature_cols if c not in df.columns]
    assert not missing_feat, f'Missing feature columns: {missing_feat}'
    assert 'label' in df.columns, 'Missing label column'
    assert 'definition_wn' in df.columns, 'Missing definition_wn column'
    assert 'answer_wn' in df.columns, 'Missing answer_wn column'

    # Stratified subsample in sample mode
    if sample_mode:
        sampled_parts = []
        for label_val in df['label'].unique():
            group = df[df['label'] == label_val]
            sampled_parts.append(
                group.sample(n=min(SAMPLE_SIZE // 2, len(group)),
                             random_state=RANDOM_SEED)
            )
        df = pd.concat(sampled_parts, ignore_index=True)
        print(f'[{timestamp()}] SAMPLE MODE: subsampled to {len(df):,} rows')

    # Summary
    print(f'  Shape: {df.shape}')
    print(f'  Label distribution: '
          f'{dict(df["label"].value_counts().sort_index())}')
    n_unique_pairs = df.groupby(['definition_wn', 'answer_wn']).ngroups
    print(f'  Unique (definition_wn, answer_wn) pairs: {n_unique_pairs:,}')

    # Validate no NaNs in feature columns
    feat_nulls = df[feature_cols].isnull().any()
    assert not feat_nulls.any(), (
        f'NaN values found in features:\n'
        f'{feat_nulls[feat_nulls].to_string()}'
    )
    print(f'  No NaN values in {len(feature_cols)} feature columns')

    # --- GroupKFold assignment (Decision 7) ---
    groups = (df['definition_wn'].astype(str) + '|||'
              + df['answer_wn'].astype(str))
    gkf = GroupKFold(n_splits=N_FOLDS)

    df['fold'] = -1
    for fold_idx, (_, test_idx) in enumerate(
            gkf.split(df, y=df['label'], groups=groups)):
        df.loc[df.index[test_idx], 'fold'] = fold_idx

    assert (df['fold'] >= 0).all(), 'Some rows not assigned to any fold'

    # Verify no cross-fold leakage
    folds_per_group = (
        df.groupby(['definition_wn', 'answer_wn'])['fold'].nunique()
    )
    leaked = folds_per_group[folds_per_group > 1]
    assert len(leaked) == 0, (
        f'{len(leaked)} groups span multiple folds — GroupKFold failed!'
    )
    print(f'  GroupKFold: {N_FOLDS} folds, no cross-fold leakage')

    # Print fold sizes
    for fold_idx in range(N_FOLDS):
        fold_mask = df['fold'] == fold_idx
        fold_size = fold_mask.sum()
        n_pos = (df.loc[fold_mask, 'label'] == 1).sum()
        print(f'    Fold {fold_idx}: {fold_size:>8,d} rows  '
              f'({n_pos:,d} pos / {fold_size - n_pos:,d} neg)')

    return df


def run_experiment(df, feature_cols, experiment_name, model_configs,
                   n_folds=N_FOLDS):
    """Run a classification experiment using pre-assigned GroupKFold splits.

    Identical logic to the run_experiment() function in NB 06. For each
    outer fold, hyperparameters are tuned via inner 3-fold StratifiedKFold
    CV on the training portion, then the best model is evaluated on the
    held-out test fold. StandardScaler is fitted on the training fold
    only for scale-sensitive models (KNN, LogReg).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``feature_cols``, ``'label'``, and ``'fold'``.
    feature_cols : list of str
        Feature columns to use as model input.
    experiment_name : str
        Label for this experiment (e.g., ``"Exp_1A"``).
    model_configs : dict
        Model definitions (see build_model_configs).
    n_folds : int
        Number of outer CV folds.

    Returns
    -------
    results_df : pd.DataFrame
        One row per (model, fold) with accuracy, F1, precision, recall,
        ROC AUC, and best hyperparameters.
    best_params : dict
        ``{model_name: {fold_idx: best_params_dict}}``.
    """
    X = df[feature_cols].values
    y = df['label'].values

    all_results = []
    best_params = {name: {} for name in model_configs}

    inner_cv = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=RANDOM_SEED)

    t0_exp = time.time()
    print(f'\n[{timestamp()}] {"="*60}')
    print(f'[{timestamp()}] {experiment_name}: '
          f'{len(feature_cols)} features, {len(df):,} rows')
    print(f'[{timestamp()}] {"="*60}')

    for fold_idx in range(n_folds):
        test_mask = (df['fold'] == fold_idx).values
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        print(f'\n[{timestamp()}] Fold {fold_idx}: '
              f'train={train_mask.sum():,}  test={test_mask.sum():,}')

        for model_name, config in model_configs.items():
            t0 = time.time()
            estimator = clone(config['estimator'])

            # Feature scaling: fitted on training fold only.
            # Random Forest is scale-invariant and skips this step.
            if config['scale']:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train)
                X_te = scaler.transform(X_test)
            else:
                X_tr = X_train
                X_te = X_test

            # Inner CV hyperparameter search
            if config['search'] == 'random':
                search = RandomizedSearchCV(
                    estimator, config['param_grid'],
                    n_iter=config.get('n_iter', 20),
                    cv=inner_cv, scoring='accuracy',
                    n_jobs=-1, random_state=RANDOM_SEED,
                )
            else:
                search = GridSearchCV(
                    estimator, config['param_grid'],
                    cv=inner_cv, scoring='accuracy',
                    n_jobs=-1,
                )

            search.fit(X_tr, y_train)
            best_params[model_name][fold_idx] = search.best_params_

            # Evaluate on held-out test fold
            y_pred = search.predict(X_te)
            y_prob = search.predict_proba(X_te)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_prob)
            elapsed = time.time() - t0

            all_results.append({
                'experiment': experiment_name,
                'model': model_name,
                'fold': fold_idx,
                'accuracy': acc,
                'f1': f1,
                'precision': prec,
                'recall': rec,
                'roc_auc': roc,
                'best_params': str(search.best_params_),
            })

            print(f'[{timestamp()}]   {model_name:<22s} '
                  f'Acc={acc:.4f}  F1={f1:.4f}  AUC={roc:.4f}  '
                  f'[{elapsed:.1f}s]')

    elapsed_total = time.time() - t0_exp
    print(f'\n[{timestamp()}] {experiment_name} complete in '
          f'{elapsed_total:.0f}s')

    return pd.DataFrame(all_results), best_params


def summarize_and_save(results_list, experiment_names, model_configs,
                       summary_path, per_fold_path):
    """Compute mean +/- SD summaries, print deltas, and save CSVs.

    Parameters
    ----------
    results_list : list of pd.DataFrame
        Per-fold results DataFrames (one per experiment).
    experiment_names : list of str
        Experiment labels in order (e.g., ['Exp_1A', 'Exp_1B']).
    model_configs : dict
        Model configurations (for iterating model names).
    summary_path : Path
        Where to save the summary CSV.
    per_fold_path : Path
        Where to save the per-fold CSV.
    """
    results_all = pd.concat(results_list, ignore_index=True)
    metrics = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']

    summary_rows = []
    for exp_name in experiment_names:
        for model_name in model_configs:
            mask = ((results_all['experiment'] == exp_name) &
                    (results_all['model'] == model_name))
            subset = results_all[mask]
            row = {'Experiment': exp_name, 'Model': model_name}
            for m in metrics:
                mean_val = subset[m].mean()
                std_val = subset[m].std()
                row[f'{m}_mean'] = mean_val
                row[f'{m}_std'] = std_val
                row[m] = f'{mean_val:.4f} +/- {std_val:.4f}'
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Display formatted table
    display_cols = ['Experiment', 'Model'] + metrics
    print(f'\n{"="*115}')
    print(f'RESULTS SUMMARY (mean +/- SD across {N_FOLDS} folds)')
    print(f'{"="*115}')
    print(summary_df[display_cols].to_string(index=False))

    # Delta: A - B per model
    exp_a, exp_b = experiment_names
    print(f'\n{"="*65}')
    print(f'Delta ({exp_a} - {exp_b})')
    print(f'{"="*65}')
    for model_name in model_configs:
        row_a = summary_df[(summary_df['Experiment'] == exp_a) &
                           (summary_df['Model'] == model_name)].iloc[0]
        row_b = summary_df[(summary_df['Experiment'] == exp_b) &
                           (summary_df['Model'] == model_name)].iloc[0]
        d_acc = row_a['accuracy_mean'] - row_b['accuracy_mean']
        d_f1 = row_a['f1_mean'] - row_b['f1_mean']
        d_auc = row_a['roc_auc_mean'] - row_b['roc_auc_mean']
        print(f'  {model_name:<22s}  '
              f'dAcc={d_acc:+.4f}  dF1={d_f1:+.4f}  dAUC={d_auc:+.4f}')

    # Save summary CSV
    save_df = summary_df[['Experiment', 'Model'] +
                         [f'{m}_mean' for m in metrics] +
                         [f'{m}_std' for m in metrics]]
    save_df.to_csv(summary_path, index=False)
    print(f'\n[{timestamp()}] Summary saved to {summary_path}')

    # Save per-fold CSV
    results_all.to_csv(per_fold_path, index=False)
    print(f'[{timestamp()}] Per-fold results saved to {per_fold_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Run classification experiments 1A/1B/2A/2B.')
    parser.add_argument('--sample', action='store_true',
                        help='Use 20K-row subsample and reduced grids')
    parser.add_argument('--harder-only', action='store_true',
                        help='Skip easy dataset (1A/1B), run only harder (2A/2B)')
    args = parser.parse_args()

    sample_mode = args.sample
    harder_only = args.harder_only
    model_configs = build_model_configs(sample_mode)

    print(f'[{timestamp()}] === Classification Experiments ===')
    print(f'[{timestamp()}] Sample mode: {sample_mode}')
    print(f'[{timestamp()}] Harder only: {harder_only}')
    print(f'[{timestamp()}] Random seed: {RANDOM_SEED}')
    print(f'[{timestamp()}] CV folds: {N_FOLDS}')
    print(f'[{timestamp()}] Project root: {PROJECT_ROOT}')

    # Print grid sizes
    print(f'\n[{timestamp()}] Hyperparameter grids:')
    for name, cfg in model_configs.items():
        total = 1
        for v in cfg['param_grid'].values():
            total *= len(v)
        if cfg['search'] == 'grid':
            print(f'  {name}: GridSearchCV — {total} combinations x 3 inner')
        else:
            n_it = cfg.get('n_iter', 20)
            print(f'  {name}: RandomizedSearchCV — '
                  f'{n_it} of {total} combinations x 3 inner')

    t0_total = time.time()

    # ==============================================================
    # EASY DATASET — Experiments 1A and 1B (PLAN.md Step 6)
    # ==============================================================
    if not harder_only:
        print(f'\n\n[{timestamp()}] {"#"*65}')
        print(f'[{timestamp()}] EASY DATASET (Exp 1A: {len(ALL_FEATURE_COLS)} features, '
              f'Exp 1B: {len(EXP_1B_COLS)} features)')
        print(f'[{timestamp()}] {"#"*65}')

        df_easy = load_and_prepare(
            DATA_DIR / 'dataset_easy.parquet',
            ALL_FEATURE_COLS,
            sample_mode,
        )

        results_1a, params_1a = run_experiment(
            df_easy, ALL_FEATURE_COLS, 'Exp_1A', model_configs)
        results_1b, params_1b = run_experiment(
            df_easy, EXP_1B_COLS, 'Exp_1B', model_configs)

        # Best hyperparameters (fold 0 as representative)
        print(f'\n[{timestamp()}] Best hyperparameters (fold 0):')
        for model_name in model_configs:
            print(f'  {model_name}:')
            print(f'    Exp 1A: {params_1a[model_name][0]}')
            print(f'    Exp 1B: {params_1b[model_name][0]}')

        summarize_and_save(
            [results_1a, results_1b],
            ['Exp_1A', 'Exp_1B'],
            model_configs,
            OUTPUT_DIR / 'results_easy.csv',
            OUTPUT_DIR / 'results_easy_per_fold.csv',
        )

        # Free memory before harder dataset
        del df_easy
    else:
        print(f'\n[{timestamp()}] Skipping easy dataset (--harder-only)')

    # ==============================================================
    # HARDER DATASET — Experiments 2A and 2B (PLAN.md Step 8)
    # ==============================================================
    print(f'\n\n[{timestamp()}] {"#"*65}')
    print(f'[{timestamp()}] HARDER DATASET (Exp 2A: {len(HARDER_FEATURE_COLS)} features, '
          f'Exp 2B: {len(EXP_2B_COLS)} features)')
    print(f'[{timestamp()}] {"#"*65}')

    df_harder = load_and_prepare(
        DATA_DIR / 'dataset_harder.parquet',
        HARDER_FEATURE_COLS,
        sample_mode,
    )

    results_2a, params_2a = run_experiment(
        df_harder, HARDER_FEATURE_COLS, 'Exp_2A', model_configs)
    results_2b, params_2b = run_experiment(
        df_harder, EXP_2B_COLS, 'Exp_2B', model_configs)

    # Best hyperparameters (fold 0 as representative)
    print(f'\n[{timestamp()}] Best hyperparameters (fold 0):')
    for model_name in model_configs:
        print(f'  {model_name}:')
        print(f'    Exp 2A: {params_2a[model_name][0]}')
        print(f'    Exp 2B: {params_2b[model_name][0]}')

    summarize_and_save(
        [results_2a, results_2b],
        ['Exp_2A', 'Exp_2B'],
        model_configs,
        OUTPUT_DIR / 'results_harder.csv',
        OUTPUT_DIR / 'results_harder_per_fold.csv',
    )

    # ==============================================================
    # Final summary
    # ==============================================================
    elapsed_total = time.time() - t0_total
    print(f'\n\n[{timestamp()}] {"="*65}')
    print(f'[{timestamp()}] ALL EXPERIMENTS COMPLETE')
    print(f'[{timestamp()}] Total elapsed: {elapsed_total:.0f}s '
          f'({elapsed_total / 60:.1f} min)')
    print(f'[{timestamp()}] {"="*65}')
    print(f'\nOutputs:')
    if not harder_only:
        print(f'  {OUTPUT_DIR / "results_easy.csv"}')
        print(f'  {OUTPUT_DIR / "results_easy_per_fold.csv"}')
    print(f'  {OUTPUT_DIR / "results_harder.csv"}')
    print(f'  {OUTPUT_DIR / "results_harder_per_fold.csv"}')


if __name__ == '__main__':
    main()
