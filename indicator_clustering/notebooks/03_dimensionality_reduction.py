import os
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.decomposition import PCA
import umap  # install via: pip install umap-learn
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.dpi'] = 120

# --- Environment Auto-Detection ---
try:
    IS_COLAB = 'google.colab' in str(get_ipython())
except NameError:
    IS_COLAB = False
IS_GREATLAKES = 'SLURM_JOB_ID' in os.environ  # Great Lakes sets this automatically

if IS_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_ROOT = Path('/content/drive/MyDrive/SIADS 692 Milestone II/Milestone II - NLP Cryptic Crossword Clues')
elif IS_GREATLAKES:
    # Update YOUR_UNIQNAME to your actual UMich uniqname
    PROJECT_ROOT = Path('/scratch/YOUR_UNIQNAME/ccc_project')
else:
    # Local: move up from notebooks/ to project root
    PROJECT_ROOT = Path.cwd().parent

DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'Project root: {PROJECT_ROOT}')
print(f'Data directory: {DATA_DIR}')
print(f'Output directory: {OUTPUT_DIR}')

np.random.seed(42)

# Check that both input files exist before proceeding
embedding_file = DATA_DIR / 'embeddings_bge_m3_all.npy'
index_file = DATA_DIR / 'indicator_index_all.csv'

assert embedding_file.exists(), (
    f'Missing embedding file: {embedding_file}\n'
    f'Run 02_embedding_generation.ipynb first to produce this file.'
)
assert index_file.exists(), (
    f'Missing index file: {index_file}\n'
    f'Run 02_embedding_generation.ipynb first to produce this file.'
)

print('Input files found:')
print(f'  {embedding_file}')
print(f'  {index_file}')

embeddings = np.load(embedding_file)
df_index = pd.read_csv(index_file, index_col=0)

print(f'Embeddings shape: {embeddings.shape}')
print(f'Index rows: {len(df_index)}')

assert embeddings.shape == (12622, 1024), (
    f'Expected shape (12622, 1024), got {embeddings.shape}'
)
assert len(df_index) == embeddings.shape[0], (
    f'Index length {len(df_index)} does not match embedding rows {embeddings.shape[0]}'
)

print(f'Shape verified: {embeddings.shape[0]:,} indicators x {embeddings.shape[1]} dimensions')

# Fit PCA with 100 components to examine the variance structure
pca_full = PCA(n_components=100, random_state=42)
pca_full.fit(embeddings)

# Print cumulative variance at key thresholds
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
for threshold in [0.50, 0.80, 0.90, 0.95, 0.99]:
    n_needed = np.searchsorted(cumvar, threshold) + 1
    print(f'{threshold:.0%} variance explained by {n_needed} components')

# Scree plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: individual explained variance (first 50 components)
ax1.bar(range(1, 51), pca_full.explained_variance_ratio_[:50],
        color='steelblue', alpha=0.8)
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Explained Variance Ratio')
ax1.set_title('Individual Explained Variance (first 50 components)')

# Right: cumulative explained variance (all 100 components)
ax2.plot(range(1, 101), cumvar, 'o-', markersize=3, color='steelblue')
ax2.axhline(y=0.90, color='red', linestyle='--', alpha=0.5, label='90% variance')
ax2.axhline(y=0.80, color='orange', linestyle='--', alpha=0.5, label='80% variance')
ax2.axvline(x=10, color='green', linestyle='--', alpha=0.5, label='10 components')
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Explained Variance')
ax2.set_title('Cumulative Explained Variance')
ax2.legend()

plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'pca_explained_variance.png', dpi=150, bbox_inches='tight')
plt.show()

# Project to 10 dimensions (for clustering comparison)
pca_10d = PCA(n_components=10, random_state=42)
embeddings_pca_10d = pca_10d.fit_transform(embeddings)
print(f'PCA 10D shape: {embeddings_pca_10d.shape}')
print(f'Variance explained by 10 components: {pca_10d.explained_variance_ratio_.sum():.1%}')

# Project to 2 dimensions (for visualization comparison)
pca_2d = PCA(n_components=2, random_state=42)
embeddings_pca_2d = pca_2d.fit_transform(embeddings)
print(f'PCA 2D shape: {embeddings_pca_2d.shape}')
print(f'Variance explained by 2 components: {pca_2d.explained_variance_ratio_.sum():.1%}')

# UMAP reduction to 10 dimensions for clustering input
reducer_10d = umap.UMAP(
    n_components=10,
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42
)

embeddings_umap_10d = reducer_10d.fit_transform(embeddings)
print(f'UMAP 10D shape: {embeddings_umap_10d.shape}')

# UMAP reduction to 2 dimensions for visualization
reducer_2d = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42
)

embeddings_umap_2d = reducer_2d.fit_transform(embeddings)
print(f'UMAP 2D shape: {embeddings_umap_2d.shape}')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# UMAP 2D
ax1.scatter(
    embeddings_umap_2d[:, 0],
    embeddings_umap_2d[:, 1],
    s=1, alpha=0.3, color='steelblue'
)
ax1.set_title('UMAP 2D Projection')
ax1.set_xlabel('UMAP 1')
ax1.set_ylabel('UMAP 2')

# PCA 2D for comparison
ax2.scatter(
    embeddings_pca_2d[:, 0],
    embeddings_pca_2d[:, 1],
    s=1, alpha=0.3, color='coral'
)
ax2.set_title('PCA 2D Projection (for comparison)')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')

plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'umap_vs_pca_2d_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Print coordinate ranges as a sanity check
print(f'UMAP 2D range — '
      f'x: [{embeddings_umap_2d[:, 0].min():.1f}, {embeddings_umap_2d[:, 0].max():.1f}], '
      f'y: [{embeddings_umap_2d[:, 1].min():.1f}, {embeddings_umap_2d[:, 1].max():.1f}]')
print(f'PCA 2D range  — '
      f'x: [{embeddings_pca_2d[:, 0].min():.2f}, {embeddings_pca_2d[:, 0].max():.2f}], '
      f'y: [{embeddings_pca_2d[:, 1].min():.2f}, {embeddings_pca_2d[:, 1].max():.2f}]')

np.save(DATA_DIR / 'embeddings_umap_10d.npy', embeddings_umap_10d)
np.save(DATA_DIR / 'embeddings_umap_2d.npy', embeddings_umap_2d)
np.save(DATA_DIR / 'embeddings_pca_10d.npy', embeddings_pca_10d)
np.save(DATA_DIR / 'embeddings_pca_2d.npy', embeddings_pca_2d)

print('Saved:')
for fname in ['embeddings_umap_10d.npy', 'embeddings_umap_2d.npy',
              'embeddings_pca_10d.npy', 'embeddings_pca_2d.npy']:
    fpath = DATA_DIR / fname
    print(f'  {fname} ({fpath.stat().st_size / 1024:.0f} KB)')

expected_shapes = {
    'embeddings_umap_10d.npy': (12622, 10),
    'embeddings_umap_2d.npy': (12622, 2),
    'embeddings_pca_10d.npy': (12622, 10),
    'embeddings_pca_2d.npy': (12622, 2),
}

for fname, expected_shape in expected_shapes.items():
    loaded = np.load(DATA_DIR / fname)
    assert loaded.shape == expected_shape, (
        f'{fname}: expected {expected_shape}, got {loaded.shape}'
    )
    print(f'{fname}: {loaded.shape} OK')

print('\nAll verification checks passed.')
