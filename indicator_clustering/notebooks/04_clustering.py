import os
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import pdist

import hdbscan  # install via: pip install hdbscan

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.dpi'] = 120

np.random.seed(42)

# --- Environment Auto-Detection ---
try:
    IS_COLAB = 'google.colab' in str(get_ipython())
except NameError:
    IS_COLAB = False

if IS_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_ROOT = Path('/content/drive/MyDrive/SIADS 692 Milestone II/Milestone II - NLP Cryptic Crossword Clues')
else:
    try:
        PROJECT_ROOT = Path(__file__).resolve().parent.parent
    except NameError:
        PROJECT_ROOT = Path.cwd().parent

DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
FIGURES_DIR = OUTPUT_DIR / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print(f'Project root: {PROJECT_ROOT}')
print(f'Data directory: {DATA_DIR}')
print(f'Output directory: {OUTPUT_DIR}')
print(f'Figures directory: {FIGURES_DIR}')

required_files = {
    'embeddings_umap_10d.npy': 'Run 03_dimensionality_reduction.ipynb (Stage 3)',
    'embeddings_umap_2d.npy': 'Run 03_dimensionality_reduction.ipynb (Stage 3)',
    'indicator_index_all.csv': 'Run 02_embedding_generation.ipynb (Stage 2)',
    'verified_clues_labeled.csv': 'Run 01_data_cleaning.ipynb (Stage 1)',
}

for fname, fix_msg in required_files.items():
    fpath = DATA_DIR / fname
    assert fpath.exists(), (
        f'Missing required file: {fpath}\n'
        f'Fix: {fix_msg}'
    )

print('All input files found:')
for fname in required_files:
    print(f'  {fname}')

# Load UMAP embeddings
embeddings_10d = np.load(DATA_DIR / 'embeddings_umap_10d.npy')
embeddings_2d = np.load(DATA_DIR / 'embeddings_umap_2d.npy')

# Load indicator index (maps row i -> indicator string)
df_index = pd.read_csv(DATA_DIR / 'indicator_index_all.csv', index_col=0)

# Load clue-level labels
df_labels = pd.read_csv(DATA_DIR / 'verified_clues_labeled.csv')

print(f'10D embeddings shape: {embeddings_10d.shape}')
print(f'2D embeddings shape:  {embeddings_2d.shape}')
print(f'Indicator index rows: {len(df_index)}')
print(f'Clue-label rows:      {len(df_labels)}')

# Sanity checks
n_indicators = len(df_index)
assert embeddings_10d.shape == (n_indicators, 10), (
    f'Expected 10D shape ({n_indicators}, 10), got {embeddings_10d.shape}'
)
assert embeddings_2d.shape == (n_indicators, 2), (
    f'Expected 2D shape ({n_indicators}, 2), got {embeddings_2d.shape}'
)
print(f'\nShape checks passed: {n_indicators:,} indicators')

# Get the indicator column name from the index file
indicator_col = df_index.columns[0]  # should be 'indicator'

# Group the clue-level labels by indicator to get the set of Ho labels
# and the set of GT labels for each unique indicator
ho_labels_per_indicator = (
    df_labels
    .dropna(subset=['wordplay_ho'])
    .groupby('indicator')['wordplay_ho']
    .apply(lambda x: set(x.unique()))
    .rename('ho_label_set')
)

gt_labels_per_indicator = (
    df_labels
    .dropna(subset=['wordplay_gt'])
    .groupby('indicator')['wordplay_gt']
    .apply(lambda x: set(x.unique()))
    .rename('gt_label_set')
)

# Join onto the indicator index
df_indicators = df_index.copy()
df_indicators = df_indicators.merge(
    ho_labels_per_indicator, left_on=indicator_col, right_index=True, how='left'
)
df_indicators = df_indicators.merge(
    gt_labels_per_indicator, left_on=indicator_col, right_index=True, how='left'
)

# For indicators with no Ho labels in the labeled file, fill with empty set
df_indicators['ho_label_set'] = df_indicators['ho_label_set'].apply(
    lambda x: x if isinstance(x, set) else set()
)
df_indicators['gt_label_set'] = df_indicators['gt_label_set'].apply(
    lambda x: x if isinstance(x, set) else set()
)

# Count how many Ho labels each indicator has
df_indicators['n_ho_labels'] = df_indicators['ho_label_set'].apply(len)

print(f'Indicators with Ho labels: {(df_indicators["n_ho_labels"] > 0).sum():,}')
print(f'Indicators with no Ho labels: {(df_indicators["n_ho_labels"] == 0).sum():,}')
print(f'\nMulti-label distribution:')
print(df_indicators['n_ho_labels'].value_counts().sort_index().to_string())

# For scatter plot coloring: pick the most frequent Ho label per indicator.
# When an indicator has multiple Ho labels (e.g., "about" = container, reversal,
# anagram), we pick the one that appears in the most clues for that indicator.
primary_ho = (
    df_labels
    .dropna(subset=['wordplay_ho'])
    .groupby(['indicator', 'wordplay_ho'])
    .size()
    .reset_index(name='count')
    .sort_values('count', ascending=False)
    .drop_duplicates(subset='indicator', keep='first')
    .set_index('indicator')['wordplay_ho']
    .rename('primary_ho_label')
)

df_indicators = df_indicators.merge(
    primary_ho, left_on=indicator_col, right_index=True, how='left'
)

print('Primary Ho label distribution:')
print(df_indicators['primary_ho_label'].value_counts().to_string())

# Sample 2,000 points for pairwise distance computation
sample_size = 2000
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(embeddings_10d), size=sample_size, replace=False)
sample_embeddings = embeddings_10d[sample_idx]

# Compute pairwise Euclidean distances (returns condensed form)
# pdist returns a 1D array of all unique pairs
pairwise_dists = pdist(sample_embeddings, metric='euclidean')

print(f'Sample size: {sample_size}')
print(f'Number of pairwise distances: {len(pairwise_dists):,}')
print(f'\nDistance statistics:')
print(f'  Min:    {pairwise_dists.min():.4f}')
print(f'  Max:    {pairwise_dists.max():.4f}')
print(f'  Mean:   {pairwise_dists.mean():.4f}')
print(f'  Median: {np.median(pairwise_dists):.4f}')
print(f'  Std:    {pairwise_dists.std():.4f}')

# Key percentiles — these will guide epsilon selection
percentiles = [5, 10, 25, 50, 75, 90, 95]
percentile_values = np.percentile(pairwise_dists, percentiles)

print(f'\nKey percentiles:')
for p, v in zip(percentiles, percentile_values):
    print(f'  {p:3d}th percentile: {v:.4f}')

fig, ax = plt.subplots(figsize=(12, 5))

ax.hist(pairwise_dists, bins=200, color='steelblue', alpha=0.7, edgecolor='none',
        density=True)

# Mark key percentiles
colors_pctl = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#4daf4a', '#ff7f00', '#e41a1c']
for p, v, c in zip(percentiles, percentile_values, colors_pctl):
    ax.axvline(v, color=c, linestyle='--', alpha=0.7, linewidth=1.2,
               label=f'{p}th pctl = {v:.3f}')

ax.set_xlabel('Euclidean Distance (10D UMAP space)')
ax.set_ylabel('Density')
ax.set_title('Pairwise Distance Distribution (2,000-point sample)')
ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
fig.savefig(FIGURES_DIR / 'pairwise_distance_distribution.png', dpi=150,
            bbox_inches='tight')
plt.show()

print(f'Saved: {FIGURES_DIR / "pairwise_distance_distribution.png"}')

# Select epsilon candidates from the distance distribution
# We use percentiles of the pairwise distance distribution as a principled basis.
# The candidates range from 0.0 (no epsilon merging) to around the 25th percentile.
# Values beyond the median would merge nearly everything.
epsilon_candidates = [
    0.0,                                            # baseline: no epsilon merging
    float(np.percentile(pairwise_dists, 1)),        # ~1st percentile
    float(np.percentile(pairwise_dists, 5)),        # ~5th percentile
    float(np.percentile(pairwise_dists, 10)),       # ~10th percentile
    float(np.percentile(pairwise_dists, 15)),       # ~15th percentile
    float(np.percentile(pairwise_dists, 20)),       # ~20th percentile
    float(np.percentile(pairwise_dists, 25)),       # ~25th percentile
]

# Round for cleaner display
epsilon_candidates = [round(e, 4) for e in epsilon_candidates]

print('Epsilon candidates for HDBSCAN sweep:')
for i, eps in enumerate(epsilon_candidates):
    print(f'  {i+1}. epsilon = {eps}')

hdbscan_results = []
hdbscan_labels_dict = {}  # store labels for each epsilon

for eps in epsilon_candidates:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,
        cluster_selection_epsilon=eps,
        metric='euclidean',
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(embeddings_10d)
    hdbscan_labels_dict[eps] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    noise_pct = n_noise / len(labels) * 100

    # Compute metrics on non-noise points only
    non_noise_mask = labels != -1
    n_clustered = non_noise_mask.sum()

    if n_clusters >= 2 and n_clustered >= 2:
        sil = silhouette_score(embeddings_10d[non_noise_mask], labels[non_noise_mask])
        db = davies_bouldin_score(embeddings_10d[non_noise_mask], labels[non_noise_mask])
    else:
        sil = float('nan')
        db = float('nan')

    hdbscan_results.append({
        'epsilon': eps,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_pct': noise_pct,
        'n_clustered': n_clustered,
        'silhouette': sil,
        'davies_bouldin': db,
    })

    print(f'eps={eps:.4f}: {n_clusters:4d} clusters, '
          f'{n_noise:5d} noise ({noise_pct:5.1f}%), '
          f'silhouette={sil:.3f}, DB={db:.3f}')

df_hdbscan = pd.DataFrame(hdbscan_results)
print('\n--- HDBSCAN Sweep Summary ---')
print(df_hdbscan.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: Number of clusters
axes[0].plot(df_hdbscan['epsilon'], df_hdbscan['n_clusters'],
             'o-', color='steelblue', linewidth=2, markersize=6)
axes[0].set_xlabel('Epsilon')
axes[0].set_ylabel('Number of Clusters')
axes[0].set_title('Clusters Found')
axes[0].grid(True, alpha=0.3)

# Panel 2: Noise percentage
axes[1].plot(df_hdbscan['epsilon'], df_hdbscan['noise_pct'],
             'o-', color='#e41a1c', linewidth=2, markersize=6)
axes[1].set_xlabel('Epsilon')
axes[1].set_ylabel('Noise Points (%)')
axes[1].set_title('Noise Percentage')
axes[1].grid(True, alpha=0.3)

# Panel 3: Silhouette score
axes[2].plot(df_hdbscan['epsilon'], df_hdbscan['silhouette'],
             'o-', color='#4daf4a', linewidth=2, markersize=6)
axes[2].set_xlabel('Epsilon')
axes[2].set_ylabel('Silhouette Score')
axes[2].set_title('Silhouette Score (non-noise only)')
axes[2].grid(True, alpha=0.3)

plt.suptitle('HDBSCAN Epsilon Sensitivity Analysis', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'hdbscan_epsilon_sensitivity.png', dpi=150,
            bbox_inches='tight')
plt.show()

print(f'Saved: {FIGURES_DIR / "hdbscan_epsilon_sensitivity.png"}')

best_hdbscan_idx = df_hdbscan['silhouette'].idxmax()
best_hdbscan_row = df_hdbscan.loc[best_hdbscan_idx]
best_eps = best_hdbscan_row['epsilon']
best_hdbscan_labels = hdbscan_labels_dict[best_eps]

print(f'Best HDBSCAN run by silhouette score:')
print(f'  Epsilon:    {best_eps}')
print(f'  Clusters:   {int(best_hdbscan_row["n_clusters"])}')
print(f'  Noise:      {int(best_hdbscan_row["n_noise"])} ({best_hdbscan_row["noise_pct"]:.1f}%)')
print(f'  Silhouette: {best_hdbscan_row["silhouette"]:.3f}')
print(f'  Davies-Bouldin: {best_hdbscan_row["davies_bouldin"]:.3f}')

k_values = [6, 8, 12, 26, 34]
k_descriptions = {
    6: 'CC for Dummies -> 6 Ho types',
    8: 'All 8 Ho wordplay types',
    12: 'Minute Cryptic -> Ho (12 subcategories)',
    26: 'Minute Cryptic ALL (26 subcategories)',
    34: 'Conceptual groups (34 categories)',
}

agglo_results = []
agglo_labels_dict = {}  # store labels for each k

for k in k_values:
    clusterer = AgglomerativeClustering(
        n_clusters=k,
        linkage='ward',
    )
    labels = clusterer.fit_predict(embeddings_10d)
    agglo_labels_dict[k] = labels

    sil = silhouette_score(embeddings_10d, labels)
    db = davies_bouldin_score(embeddings_10d, labels)
    ch = calinski_harabasz_score(embeddings_10d, labels)

    agglo_results.append({
        'k': k,
        'description': k_descriptions[k],
        'silhouette': sil,
        'davies_bouldin': db,
        'calinski_harabasz': ch,
    })

    print(f'k={k:2d} ({k_descriptions[k]:>40s}): '
          f'silhouette={sil:.3f}, DB={db:.3f}, CH={ch:.0f}')

df_agglo = pd.DataFrame(agglo_results)
print('\n--- Agglomerative Clustering Summary ---')
print(df_agglo.to_string(index=False))

# The 8 Ho wordplay types
HO_TYPES = ['anagram', 'reversal', 'hidden', 'container', 'insertion',
             'deletion', 'homophone', 'alternation']

# Consistent color palette for the 8 Ho wordplay types
HO_COLORS = {
    'anagram': '#e41a1c',
    'reversal': '#377eb8',
    'hidden': '#4daf4a',
    'container': '#984ea3',
    'insertion': '#ff7f00',
    'deletion': '#a65628',
    'homophone': '#f781bf',
    'alternation': '#999999',
}


def plot_clusters(embeddings_2d, labels, title, filename, noise_label=-1):
    """Scatter plot of 2D UMAP colored by cluster assignment."""
    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot noise points first (gray, small)
    noise_mask = labels == noise_label
    if noise_mask.any():
        ax.scatter(
            embeddings_2d[noise_mask, 0],
            embeddings_2d[noise_mask, 1],
            s=1, alpha=0.15, color='lightgray', label='noise', zorder=1
        )

    # Plot clustered points
    non_noise_mask = ~noise_mask
    unique_labels = sorted(set(labels[non_noise_mask]))
    n_clusters = len(unique_labels)

    # Use a colormap with enough distinct colors
    cmap = plt.cm.get_cmap('tab20', max(n_clusters, 20))
    for i, cl in enumerate(unique_labels):
        mask = labels == cl
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            s=2, alpha=0.4, color=cmap(i % 20), zorder=2
        )

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(f'{title} ({n_clusters} clusters)')

    if noise_mask.any():
        n_noise = noise_mask.sum()
        ax.annotate(f'Noise: {n_noise} ({n_noise/len(labels)*100:.1f}%)',
                    xy=(0.02, 0.98), xycoords='axes fraction',
                    fontsize=9, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {FIGURES_DIR / filename}')


def plot_ho_type_overlay(embeddings_2d, ho_type, df_indicators, indicator_col,
                         title_prefix, filename_prefix):
    """Scatter plot highlighting indicators of a single Ho wordplay type."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # All points in gray
    ax.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        s=1, alpha=0.1, color='lightgray', zorder=1
    )

    # Highlight indicators that have this Ho label (checking the set, not primary)
    mask = df_indicators['ho_label_set'].apply(lambda s: ho_type in s).values
    n_highlighted = mask.sum()

    ax.scatter(
        embeddings_2d[mask, 0], embeddings_2d[mask, 1],
        s=4, alpha=0.5, color=HO_COLORS[ho_type], zorder=2,
        label=f'{ho_type} (n={n_highlighted:,})'
    )

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(f'{title_prefix} — Ho type: {ho_type}')
    ax.legend(loc='upper right', fontsize=9)

    filename = f'{filename_prefix}_ho_{ho_type}.png'
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {FIGURES_DIR / filename}')

plot_clusters(
    embeddings_2d,
    agglo_labels_dict[8],
    title='Agglomerative Clustering (Ward, k=8)',
    filename='agglo_k8_clusters.png'
)

for ho_type in HO_TYPES:
    plot_ho_type_overlay(
        embeddings_2d,
        ho_type=ho_type,
        df_indicators=df_indicators,
        indicator_col=indicator_col,
        title_prefix='Agglomerative k=8',
        filename_prefix='agglo_k8'
    )

plot_clusters(
    embeddings_2d,
    best_hdbscan_labels,
    title=f'HDBSCAN (eps={best_eps})',
    filename='hdbscan_best_clusters.png'
)

for ho_type in HO_TYPES:
    plot_ho_type_overlay(
        embeddings_2d,
        ho_type=ho_type,
        df_indicators=df_indicators,
        indicator_col=indicator_col,
        title_prefix=f'HDBSCAN (eps={best_eps})',
        filename_prefix='hdbscan_best'
    )

def inspect_clusters(labels, embeddings_10d, df_indicators, indicator_col,
                     method_name, n_examples=10):
    """Print centroid-nearest indicators and Ho type distribution per cluster."""
    unique_labels = sorted(set(labels))

    for cl in unique_labels:
        if cl == -1:
            # Summarize noise points briefly
            n_noise = (labels == -1).sum()
            print(f'\n{"=" * 60}')
            print(f'{method_name} — Noise points: {n_noise}')
            print(f'{"=" * 60}')
            continue

        mask = labels == cl
        cluster_size = mask.sum()
        cluster_embeddings = embeddings_10d[mask]
        cluster_indicators = df_indicators.loc[mask]

        # Compute centroid and find nearest points
        centroid = cluster_embeddings.mean(axis=0)
        dists_to_centroid = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        nearest_idx = np.argsort(dists_to_centroid)[:n_examples]
        nearest_indicators = cluster_indicators.iloc[nearest_idx][indicator_col].tolist()

        # Ho type distribution within this cluster
        # Explode the label sets so each (indicator, type) pair is counted
        ho_exploded = cluster_indicators['ho_label_set'].explode()
        ho_exploded = ho_exploded[ho_exploded.apply(lambda x: isinstance(x, str))]

        print(f'\n{"=" * 60}')
        print(f'{method_name} — Cluster {cl} (n={cluster_size})')
        print(f'{"-" * 60}')
        print(f'Nearest to centroid: {", ".join(nearest_indicators)}')

        if len(ho_exploded) > 0:
            type_counts = ho_exploded.value_counts()
            total_labels = type_counts.sum()
            print(f'Ho type distribution ({total_labels} total label instances):')
            for wtype, count in type_counts.items():
                pct = count / total_labels * 100
                print(f'  {wtype:15s}: {count:5d} ({pct:5.1f}%)')
        else:
            print('  No Ho labels available for indicators in this cluster')

inspect_clusters(
    agglo_labels_dict[8],
    embeddings_10d,
    df_indicators,
    indicator_col,
    method_name='Agglomerative k=8'
)

# For HDBSCAN, inspect only the largest clusters to keep output concise
unique_hdbscan_labels = sorted(set(best_hdbscan_labels))
n_hdbscan_clusters = len([l for l in unique_hdbscan_labels if l != -1])

if n_hdbscan_clusters > 15:
    # Find the 15 largest clusters by size
    cluster_sizes = pd.Series(best_hdbscan_labels[best_hdbscan_labels != -1]).value_counts()
    top_15_labels = set(cluster_sizes.head(15).index)

    # Create a filtered label array: keep top 15, set rest to -1
    filtered_labels = np.where(
        np.isin(best_hdbscan_labels, list(top_15_labels)),
        best_hdbscan_labels,
        -1
    )
    print(f'HDBSCAN found {n_hdbscan_clusters} clusters. '
          f'Showing the 15 largest below.')
    inspect_clusters(
        filtered_labels,
        embeddings_10d,
        df_indicators,
        indicator_col,
        method_name=f'HDBSCAN (eps={best_eps})'
    )
else:
    inspect_clusters(
        best_hdbscan_labels,
        embeddings_10d,
        df_indicators,
        indicator_col,
        method_name=f'HDBSCAN (eps={best_eps})'
    )

# Save HDBSCAN cluster labels for each epsilon
for eps, labels in hdbscan_labels_dict.items():
    eps_str = f'{eps:.4f}'.replace('.', 'p')  # e.g., 0.1500 -> 0p1500
    fname = f'cluster_labels_hdbscan_eps_{eps_str}.csv'
    out_df = pd.DataFrame({
        'indicator': df_indicators[indicator_col].values,
        'cluster_label': labels,
    })
    out_df.to_csv(DATA_DIR / fname, index=False)
    print(f'Saved: {fname}')

# Save agglomerative cluster labels for each k
for k, labels in agglo_labels_dict.items():
    fname = f'cluster_labels_agglo_k{k}.csv'
    out_df = pd.DataFrame({
        'indicator': df_indicators[indicator_col].values,
        'cluster_label': labels,
    })
    out_df.to_csv(DATA_DIR / fname, index=False)
    print(f'Saved: {fname}')

# Build a combined metrics summary for all runs
all_metrics = []

# HDBSCAN runs
for _, row in df_hdbscan.iterrows():
    all_metrics.append({
        'method': 'HDBSCAN',
        'parameters': f'min_cluster_size=10, eps={row["epsilon"]}',
        'n_clusters': int(row['n_clusters']),
        'n_noise': int(row['n_noise']),
        'noise_pct': row['noise_pct'],
        'silhouette': row['silhouette'],
        'davies_bouldin': row['davies_bouldin'],
        'calinski_harabasz': float('nan'),  # not computed for HDBSCAN (noise points)
    })

# Agglomerative runs
for _, row in df_agglo.iterrows():
    all_metrics.append({
        'method': 'Agglomerative (Ward)',
        'parameters': f'k={int(row["k"])}',
        'n_clusters': int(row['k']),
        'n_noise': 0,
        'noise_pct': 0.0,
        'silhouette': row['silhouette'],
        'davies_bouldin': row['davies_bouldin'],
        'calinski_harabasz': row['calinski_harabasz'],
    })

df_all_metrics = pd.DataFrame(all_metrics)
metrics_path = OUTPUT_DIR / 'clustering_metrics_summary.csv'
df_all_metrics.to_csv(metrics_path, index=False)

print(f'Saved: {metrics_path}')
print(f'\n--- Full Metrics Summary ---')
print(df_all_metrics.to_string(index=False))

# Final file listing
print('=== All Output Files ===')
print(f'\nCluster labels (in {DATA_DIR}):')
for f in sorted(DATA_DIR.glob('cluster_labels_*.csv')):
    print(f'  {f.name}')

print(f'\nMetrics summary (in {OUTPUT_DIR}):')
print(f'  clustering_metrics_summary.csv')

print(f'\nFigures (in {FIGURES_DIR}):')
for f in sorted(FIGURES_DIR.glob('*.png')):
    print(f'  {f.name}')

print('\nDone. All outputs saved for Notebook 05.')
