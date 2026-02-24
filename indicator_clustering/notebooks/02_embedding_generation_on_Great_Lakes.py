import os
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

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
    PROJECT_ROOT = Path('/home/vwinters/ccc-project/indicator_clustering')
else:
    # Local: move up from notebooks/ to project root
    PROJECT_ROOT = Path.cwd().parent

DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Batch size for embedding generation.
# Colab free-tier T4 GPUs have 16GB VRAM — use a smaller batch to avoid OOM.
# Great Lakes V100/A40 and local GPUs with more VRAM can handle larger batches.
BATCH_SIZE = 32 if IS_COLAB else 64

print(f'Project root: {PROJECT_ROOT}')
print(f'Data directory: {DATA_DIR}')
print(f'Batch size: {BATCH_SIZE}')

np.random.seed(42)

# Check that the input file exists before proceeding
input_file = DATA_DIR / 'verified_indicators_unique.csv'
assert input_file.exists(), (
    f'Missing input file: {input_file}\n'
    f'Run 01_data_cleaning_Victoria.ipynb first to produce this file.'
)

df_indicators = pd.read_csv(input_file)
indicators_list = df_indicators['indicator'].tolist()

print(f'Loaded {len(indicators_list):,} unique indicators')
print(f'Examples: {indicators_list[:5]}')
print(f'Shortest: "{min(indicators_list, key=len)}" ({len(min(indicators_list, key=len))} chars)')
print(f'Longest: "{max(indicators_list, key=len)}" ({len(max(indicators_list, key=len))} chars)')

# Load the BGE-M3 model
# First run will download the model (~2.3 GB). Subsequent runs use the cached version.
model = SentenceTransformer('BAAI/bge-m3')
print(f'Model loaded: {model.get_sentence_embedding_dimension()} dimensions')

# Generate embeddings for all unique indicators
# show_progress_bar=True displays a tqdm progress bar during encoding
embeddings = model.encode(
    indicators_list,
    batch_size=BATCH_SIZE,
    show_progress_bar=True
)

print(f'Embeddings shape: {embeddings.shape}')
print(f'Dtype: {embeddings.dtype}')
print(f'Memory: {embeddings.nbytes / 1024**2:.1f} MB')

# Save the embedding matrix
np.save(DATA_DIR / 'embeddings_bge_m3_all.npy', embeddings)
print(f'Saved embeddings to {DATA_DIR / "embeddings_bge_m3_all.npy"}')

# Save the indicator index (row number -> indicator string)
df_indicators.to_csv(DATA_DIR / 'indicator_index_all.csv', index=True)
print(f'Saved indicator index to {DATA_DIR / "indicator_index_all.csv"}')

# Reload and verify
embeddings_check = np.load(DATA_DIR / 'embeddings_bge_m3_all.npy')
index_check = pd.read_csv(DATA_DIR / 'indicator_index_all.csv', index_col=0)

assert embeddings_check.shape[0] == len(index_check), (
    f'Shape mismatch: embeddings has {embeddings_check.shape[0]} rows, '
    f'index has {len(index_check)} rows'
)
assert embeddings_check.shape[1] == 1024, (
    f'Expected 1024 dimensions, got {embeddings_check.shape[1]}'
)

print(f'Embeddings: {embeddings_check.shape}')
print(f'Index: {len(index_check)} rows')
print(f'All checks passed.')

# Spot-check: find a known indicator and verify it has a non-zero embedding
spot_check = 'about'
row = index_check[index_check['indicator'] == spot_check].index[0]
norm = np.linalg.norm(embeddings_check[row])
print(f'\nSpot check: "{spot_check}" is at row {row}, embedding L2 norm = {norm:.4f}')
