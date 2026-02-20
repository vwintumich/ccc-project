#!/usr/bin/env python
# coding: utf-8

# # EDA and Data Cleaning for Indicator Clustering
# Unsupervised Learning Component of Milestone II group project:
# 
# Exploring Wordplay and Misdirection in Cryptic Crossword Clues with Natural Language Processing

# ## Imports

# In[1]:


# imports
import os
from pathlib import Path
import pandas as pd
import numpy as np
import re
import string
import unicodedata
import matplotlib.pyplot as plt


# ## Loading the Data

# In[2]:


# ==========================
# PATHS & CONFIG
# ==========================
# 1. Detect environment
try:
    IS_COLAB = 'google.colab' in str(get_ipython())
except NameError:
    IS_COLAB = False

if IS_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_ROOT = Path('/content/drive/MyDrive/SIADS 692 Milestone II/Milestone II - NLP Cryptic Crossword Clues')
else:
    # On local, move up from notebooks/ to project root
    # Adjust the number of .parent calls based on where this notebook sits
    PROJECT_ROOT = Path.cwd().parent 

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# In[3]:


# Read each CSV file into a DataFrame
df_clues = pd.read_csv(f'{DATA_DIR}/clues_raw.csv')
df_indicators = pd.read_csv(f'{DATA_DIR}/indicators_raw.csv')
df_ind_by_clue = pd.read_csv(f'{DATA_DIR}/indicators_by_clue_raw.csv')
df_ind_consolidated = pd.read_csv(f'{DATA_DIR}/indicators_consolidated_raw.csv')
df_charades = pd.read_csv(f'{DATA_DIR}/charades_raw.csv')
df_charades_by_clue = pd.read_csv(f'{DATA_DIR}/charades_by_clue_raw.csv')


# ## Reformat `clue_ids`

# ### Indicators Table `clue_ids`

# In[4]:


# Uncomment to see how the clue_id data looks before cleaning
#df_indicators.sample().style.set_properties(**{"white-space": "pre-wrap"})


# In[5]:


# Instead of a string with redundant indices, extract only the clue_ids in
# brackets to create a list of integers
df_indicators["clue_ids"] = (
    df_indicators["clue_ids"]
    .str.findall(r"\[(\d+)\]")
    .apply(lambda xs: [int(x) for x in xs])
)

# Include a new column to keep track of how many clues have this indicator
df_indicators["num_clues"] = df_indicators["clue_ids"].apply(len)


# In[6]:


df_indicators.sample(3).style.set_properties(**{"white-space": "pre-wrap"})


# ### Charades Table `clue_ids`

# In[7]:


# Uncomment to see what the clue_ids look like before cleaning
#df_charades.sample().style.set_properties(**{"white-space": "pre-wrap"})


# In[8]:


# Instead of a string with redundant indices, extract only the clue_ids in
# brackets to create a list of integers
df_charades["clue_ids"] = (
    df_charades["clue_ids"]
    .str.findall(r"\[(\d+)\]")
    .apply(lambda xs: [int(x) for x in xs])
)

# Include a new column to keep track of how many clues have this charade
df_charades["num_clues"] = df_charades["clue_ids"].apply(len)


# In[9]:


df_charades.sample(3).style.set_properties(**{"white-space": "pre-wrap"})


# ## Helper Functions

# ### `clue_info()` - Investigate A Clue
# 
# `clue_info(n)` displays all the basic and derived information for the clue with `clue_id = n`.

# In[10]:


# View all the info for a specific clue (by clue_id), including
# clue surface, answer, definition, charades, and indicators
def clue_info(n):
  clue_cols = ['clue_id', 'clue', 'answer', 'definition', 'source_url']
  print(
      df_clues[df_clues['clue_id'] == n][clue_cols].style.set_properties(
        subset=["clue", 'source_url'],
        **{"white-space": "pre-wrap"}
    )
      )
  print()
  print(df_charades_by_clue[df_charades_by_clue['clue_id']== n])
  print()
  print(df_ind_by_clue[df_ind_by_clue["clue_id"] == n])
  print()
  print(df_indicators[df_indicators['clue_ids'].apply(lambda lst: n in lst)])


# In[11]:


clue_info(623961)


# ### `normalize()` - Remove punctuation, accents, make lowercase

# In[12]:


# Normalize takes a string (clue surface, indicator, definition, answer),
# And returns the same text but with punctuation (including dashes) and
# accents removed, and all lowercase.
def normalize(s: str) -> str:
  # remove accents and punctuation, convert to lowercase
  s_normalized = ''.join(
      ch for ch in unicodedata.normalize('NFD', s)
      if unicodedata.category(ch).startswith(('L', 'N', 'Zs'))
  ).lower()

  return s_normalized


# #### Normalization Question: Remove Dashes in Answer?
# 
# See Clue 624269. Should LA-DI-DA be normalized as:
# * la di da
# * la-di-da
# * ladida

# ### `count_unique_clues()`
# This helper function will let us count how many unique clues are represented in an indicator DataFrame.

# In[13]:


def count_unique_clues(series):
  """
  Calculates the total number of unique elements across all lists in a pandas
  Series. Applied to a column of `clue_ids`, this will count the number of
  unique clues represented in an indicator dataframe.

  Args:
    series (pandas.Series): A Series where each element is a list.

  Returns:
    int: The total count of unique elements.
  """
  unique_elements = set()
  for sublist in series:
    if isinstance(sublist, list):
      unique_elements.update(sublist)
  return len(unique_elements)


# ## All Available Tables
# * Indicators
# * Indicator By Clue
# * Indicators Consolidated
# * Bonus Dictionary Version of Indicators Consolidated
# * Clue
# * Charade
# * Charade by Clue

# ### Indicators

# In[14]:


df_indicators.sample(3).style.set_properties(
        subset=["clue_ids"],
        **{"white-space": "pre-wrap"}
    )


# In[15]:


# Uncomment to prove that `indicator` is already normalized - no accents,
# punctuation (including dashes), or capital letters

# Create a column of normalized indicators
#df_indicators['indicator_normalized'] = df_indicators['indicator'].apply(normalize)

# Check out all rows where normalization changed the indicator
#df_indicators.loc[df_indicators['indicator'] != df_indicators['indicator_normalized']]


# ### Indicators by Clue

# In[16]:


df_ind_by_clue.head()


# ### Indicators Consolidated

# This dataframe contains eight columns--one for each type of wordplay--and one row with a string of all consolidated indicators found in the dataset by George Ho.
# 
# This data is better represented as a dictionary, so we create `ind_by_wordplay_dict` from `df_ind_consolidated`.

# In[17]:


df_ind_consolidated


# ### Dictionary for Indicators Consolidated
# 
# `ind_by_wordplay_dict` is a dictionary with wordplay types for the keys and a list of all indicators consolidated for each wordplay type.
# 
# Nathan points out that some words in this dictionary have '\' or other suspicious characters. But because we use `indicators` instead (it has clue IDs for each indicator), we're not bothering to clean this dictionary.

# In[18]:


# Create a dictionary where the key is the wordplay type, and the value is
# the list of associated unique indicators.
ind_by_wordplay_dict = {}

for wordplay in df_ind_consolidated.columns:
  ind_by_wordplay_dict[wordplay] = df_ind_consolidated[wordplay].values[0].split('\n')


# In[19]:


# Uncomment or change key to view all indicators for that wordplay
#ind_by_wordplay_dict['insertion']


# In[20]:


# See how many unique indicators there are for each type of wordplay
for wordplay in ind_by_wordplay_dict:
  print(f"{wordplay}: {len(ind_by_wordplay_dict[wordplay])}")


# ### Clues
# 
# Create normalized entries for the `clue`, `answer` and `definition` by removing punctuation and accents and making them all lowercase:
# * `surface`: The clue without the '(n)' at the end. The surface reading only, with capitalization and punctuation preserved.
# * `surface_normalized`: The clue surface without capitalization, punctuation, or accents.
# * `answer_normalized`: The answer in lower case with punctuation (hypthens) and accents removed.
# * `definition_normalized`: The definition in lower case with punctuation and accents removed.
# 
# There are 323 rows with NaN for a clue. Remove these rows before proceeding. There are also 2,259 rows with NaN for `answer` and 149,096 rows with NaN for `definition`. However, because we're only concerned here with indicators (and verifying that the indicators are found in the clue), we will only drop the rows with NaN for `clue`.

# In[21]:


# Uncomment to see how many rows have NaN for 'clue', 'answer', or 'definition'
#df_clues['clue'].value_counts(dropna=False).head()
#df_clues['answer'].value_counts(dropna=False).head()
#df_clues['definition'].value_counts(dropna=False).head()

# Drop all rows where the clue or answer is NaN (they are type float, and we want clue to be a string)
df_clues.dropna(subset=['clue', 'answer'], inplace=True)


# In[22]:


# Surface: remove trailing numeric parentheses in clue
df_clues['surface'] = df_clues['clue'].astype(str).apply(lambda x: re.sub(r'\s*\(\d+(?:[,\s-]+\d+)*\)$', '', x))


# In[23]:


# Create surface normalized - no accents, punctuation, capitalized letters
df_clues['surface_normalized'] = df_clues['surface'].astype(str).apply(normalize)

# Create answer normalized - no accents, punctuation, capitalized letters
df_clues['answer_normalized'] = df_clues['answer'].astype(str).apply(normalize)

# Create definition normalized - no accents, punctuation, capitalized letters
#df_clues['definition_normalized'] = df_clues['definition'].astype(str).apply(normalize)


# In[24]:


df_clues.head()


# ### Charades by Clue

# In[25]:


df_charades_by_clue.sample(3)


# ### Charades

# In[26]:


df_charades.sample(3).style.set_properties(
        subset=["clue_ids"],
        **{"white-space": "pre-wrap"}
    )


# # Data Requirements & Unresolved Dilemmas
# 

# As we apply the requirements, our dataset of valid indicators will keep decreasing. Create a dataframe to keep track of how much data we're losing at each step.
# 
# * Once we restrict our dataset, do we have enough indicators for clustering (assume $2 < k < 12$)?

# In[27]:


# Create a dataframe and add the counts from Indicators
df_ind_counts = pd.DataFrame(columns=["unique_inds"])
df_ind_counts['unique_inds'] = df_indicators.groupby(by=['wordplay']).count()['indicator']

# Also keep track of the total number of indicators
ind_total = df_ind_counts['unique_inds'].sum()


# In[28]:


# Include a column that counts indicators by clue, which will
# double-count any indicator appearing in multiple clues
df_ind_counts['all_instances'] = df_ind_by_clue.count()

# Rearrange the columns to go from large to small, remove counts from
# ind_consolidated because they don't have associated clue IDs.
df_ind_counts = df_ind_counts[['all_instances', 'unique_inds']]


# In[29]:


print(f"Total Number of Clues: {len(df_clues):,}")
print(f"Total Unique Indicators: {ind_total:,}")
print(f"Total Instances of Indicators in All Clues: {df_ind_counts['all_instances'].sum():,}")
print(f"Total Number of Clues Containing Indicator(s): {df_ind_by_clue['clue_id'].count():,}")


# In[30]:


df_ind_counts


# Summary:
# * Of the entire dataset of 660,613 cryptic crossword clues, 88,037 clues came from blog posts where indicators could be identified. (from `df_ind_by_clue`)
# * Because sometimes clues have more than one indicator, a total of 93,867 indicators were found in the dataset, and are associated with a parsed clue. (from `df_ind_by_clue`)
# * CCCs reuse indicators. Of the 93,867 indicators identified in the data, only 15,735 are unique.
# * More unique indicators appear in `df_ind_consolidated` (16,061) than in `df_indicators` (15,735). We cannot easily discover why because the Indicators Consolidated table was stripped of context.
# * <b>We will use the Indicators table</b> going forward because it cites which clues used that indicator. We can verify the quality of the data better.
# * Note that a common indicator like "within" may be counted twice: once as a hidden indicator and once as a container indicator. Therefore, if we were to export the 15,735 indicator words, there would be duplicates for the different types of wordplay.
# 
# 

# ### Indicator word(s) must appear in the clue surface text
# 
# To make sure that the indicator word wasn't incorrectly parsed, it must appear in the clue as a fully intact word, not just a segment of a word.
# 
# This will exclude some clues that use a compound word to contain both the indicator and fodder, like Minute Cryptic's "This semicircle encircles you (4)". Semi is a selection indicator telling you to take half of "circle".

# In[31]:


# Add a column with a list of VERIFIED clue IDs: where we know the indicator
# appeared in the surface text as intact words.

# Build fast lookup dictionary
clue_lookup = df_clues.set_index("clue_id")["surface_normalized"].to_dict()

# Given an indicator and its list of clue_ids where it appears,
# return a new list of clue_ids where the indicator definitely
# appears in the normalized clue surface as intact words.
def verify_clues(indicator, clue_ids):
    if not clue_ids:
        return []

    # Escape regex special characters inside indicator
    pattern = rf"\b{re.escape(indicator)}\b"

    verified = []

    for cid in clue_ids:
        surface = clue_lookup.get(cid)

        if surface and re.search(pattern, surface):
            verified.append(cid)

    return verified


# add the column for the list of verified clue_ids
df_indicators["clue_ids_verified"] = df_indicators.apply(
    lambda row: verify_clues(row["indicator"], row["clue_ids"]),
    axis=1
)


# In[32]:


# Add a column that counts the number of verified clue_ids for each indicator
df_indicators['num_clues_verified'] = df_indicators['clue_ids_verified'].apply(len)


# In[33]:


# Uncomment to inspect the indicators table
df_indicators.sample(3)


# In[34]:


# Inspect some clues where the indicators were invalid
#clue_info(635505) # indicator not in clue or on webpage
#clue_info(591484) # indicator not in clue or on webpage
#clue_info(627621) # indicator not in clue, defn NaN
#clue_info(422350) # indicator is a partial word in clue bc blogger error
#clue_info(76808) # misparsed 'hidden' formatting, the identified indicator is actually fodder


# In[35]:


# Keep track of how many indicators are left if we keep only ones with
# at least one verified clue_id
mask = df_indicators['num_clues_verified'] > 0
df_ind_counts['verified_inds'] = df_indicators[mask].groupby(by=['wordplay']).count()['indicator']


# In[36]:


df_ind_counts.style.format('{:,}')


# In[37]:


df_ind_counts.sum()


# ### Character Lengths of Indicators Must Be Reasonable
# 
# Investigate indicators that are 1, 2, or 3 characters long for invalid words. These may already be caught when we excluded indicators that did not appear intact in the clue.
# 
# Also investigate the longest indicators.
# 
# NOTE: Once we limit ourselves to only verified indicators (they appear as intact words in the clue surface), the indicators suspicious because of their length all get excluded. All the shortest and longest indicators look like real words.

# In[38]:


# Create a column for the number of characters in the indicator phrase
df_indicators['num_chars'] = df_indicators['indicator'].apply(len)


# In[39]:


# See the counts for each indicator length, just for verified indicators
mask = (df_indicators['num_clues_verified'] > 0)
print(df_indicators[mask]['num_chars'].value_counts(dropna=False).sort_index())


# In[40]:


# Visualize the distribution of indicator length (as number of characters)
# just for unique indicators with verified clues
df_indicators[mask]['num_chars'].value_counts().sort_index().plot(kind='bar')


# In[41]:


# Uncomment to manually inspect 2-character verified indicators
#cols = ['wordplay', 'indicator', 'clue_ids_verified', 'num_clues_verified', 'num_clues']
#mask = (df_indicators['num_clues_verified'] > 0) & (df_indicators['num_chars'] == 2)
#df_indicators[mask][cols].head(12).sort_values(by='num_clues_verified', ascending=False)


# In[42]:


# Uncomment to manually inspect 3-character verified indicators
#cols = ['wordplay', 'indicator', 'clue_ids_verified', 'num_clues_verified', 'num_clues']
#mask = (df_indicators['num_clues_verified'] > 0) & (df_indicators['num_chars'] == 3)
#df_indicators[mask][cols].head(83).sort_values(by='num_clues_verified', ascending=False)


# In[77]:


# Uncomment to manually inspect the longest verified indicators
#cols = ['wordplay', 'indicator', 'clue_ids_verified', 'num_clues', 'num_chars']
#mask = (df_indicators['num_clues_verified'] > 0) & (df_indicators['num_chars'] > 25)
#df_indicators[mask][cols].sort_values(by='num_chars', ascending=False)


# ### Issue: Some indicator phrases may contain some fodder
# Inspecting the longest verified indicators, it's possible that some of these phrases contain more than just the indicator, but they all look like an indicator is at least present. 
# 
# If we later represent these as semantic vectors using a SentenceTransformer model, the extra fodder words could be a source of noise. We may want to exclue indicators with long character counts, or even create a data cleaning step that reduces these longer phrases to the known indicator (phrases) they contain.

# # Verifiable Wordplay Types

# ### Hiddens (FWD & REV)
# 
# Letters to hiddens appear directly in the clue surface, either normally or in reverse, ignoring punctuation and spaces.
# 
# This finds 23,079 clues where the answer is hidding going forwards and 6,823 where the answer is hidden in reverse. However, these are overestimates because the answers have not been verified and include some very short malformed answers that are easy to find.

# In[44]:


df_clues.sample()


# In[45]:


df_indicators.sample()


# In[46]:


# Compute hidden_fwd and hidden_rev
# Helper function to remove all whitespace for hidden word search
def remove_all_whitespace(text: str) -> str:
    if isinstance(text, str):
        return text.replace(" ", "")
    return ""

# Create 'answer_no_spaces' from 'answer_normalized'
df_clues['answer_no_spaces'] = df_clues['answer_normalized'].apply(remove_all_whitespace)

# Create 'surface_no_spaces' from 'surface_normalized'
df_clues['surface_no_spaces'] = df_clues['surface_normalized'].apply(remove_all_whitespace)

# Calculate 'hidden_fwd'
df_clues['hidden_fwd'] = df_clues.apply(
    lambda row: row['answer_no_spaces'] in row['surface_no_spaces'],
    axis=1
)

# Calculate 'hidden_rev'
df_clues['answer_no_spaces_rev'] = df_clues['answer_no_spaces'].apply(lambda x: x[::-1])
df_clues['hidden_rev'] = df_clues.apply(
    lambda row: row['answer_no_spaces_rev'] in row['surface_no_spaces'],
    axis=1
)


# In[47]:


df_clues[df_clues['hidden_fwd']].shape[0]


# In[48]:


df_clues[df_clues['hidden_rev']].shape[0]


# In[84]:


df_clues.head()


# In[ ]:


# Compute answer letter count (needed for hidden_fwd filtering below and later exports)
df_clues['answer_letter_count'] = df_clues['answer_no_spaces'].apply(len)


# In[88]:


# See the counts for each answer length, just for verified hidden_fwd
mask = (df_clues['hidden_fwd']) & (df_clues['answer_letter_count'] == 4)
cols = ['clue', 'answer']
df_clues[mask][cols]


# Maybe restrict to 4+ letter answer? Also verify that answer have the correct letter count and format. 

# ## Alternation
# 
# If the answer word appears in the surface as alternating letters, label it as verified alternation wordplay.
# 
# This found 4,220 clues with alternation, but some of those will be erroneous (short) answers.

# In[49]:


# An efficient way to find alternation

# We cache the regex pattern to avoid re-compiling inside the loop
# This looks for the answer characters with exactly one char between them
def check_alternation_seq(ans, clue):
    if not ans or not clue:
        return False
    # Creates "A.N.S.W.E.R"
    pattern = ".".join(re.escape(c) for c in ans)
    return bool(re.search(pattern, clue))

# Applying to the dataframe
df_clues['alternation'] = [
    check_alternation_seq(ans, clue)
    for ans, clue in zip(df_clues['answer_no_spaces'], df_clues['surface_no_spaces'])
]


# In[50]:


df_clues[df_clues['alternation']].shape[0]


# In[ ]:





# # Summary of Indicators

# In[51]:


df_ind_counts.style.format('{:,}')


# In[52]:


df_ind_counts.sum().to_frame().T.style.format('{:,}')


# In[53]:


df_ind_counts.sort_values(by='all_instances').plot.barh(stacked=False, figsize=(8, 5))


# In[54]:


# Add a column for the number of words within an indicator
df_indicators['ind_wc'] = df_indicators['indicator'].apply(lambda x: len(x.split()))


# In[55]:


# Visualize the valid indicators by word count
mask = df_indicators['num_clues_verified'] > 0
df_indicators[mask]['ind_wc'].value_counts().sort_index().plot(kind='bar')


# In[56]:


# Visualize the prevalence/redundancy of valid indicators
mask = df_indicators['num_clues_verified'] > 0
df_indicators[mask]['num_clues_verified'].value_counts().head(15).sort_index().plot(kind='bar', figsize=(8, 5))


# In[57]:


# View some examples of the most common indicators
df_indicators[['num_clues_verified', 'indicator', 'wordplay']].sort_values(by='num_clues_verified', ascending=False).head(10)


# # Export Verified Indicators for Downstream Stages
# 
# This section produces the output files consumed by Stage 2 (embedding generation) and Stage 5 (evaluation).
# 
# **Output files:**
# - `verified_indicators_unique.csv` — One row per unique indicator string (12,622 rows). No labels. This is the input to Stage 2 (`02_embedding_generation_Victoria.ipynb`).
# - `verified_clues_labeled.csv` — One row per verified (clue_id, indicator) pair (76,015 rows). Includes Ho blog labels and algorithmic ground-truth labels. Used for evaluation.
# 
# Note that `df_indicators` contains 14,195 verified rows because the same indicator string can appear under multiple wordplay types (e.g., "about" appears as container, reversal, and anagram — three rows). The deduplicated export collapses these to 12,622 unique strings.

# In[58]:


df_indicators.head()


# In[59]:


mask = df_indicators['num_clues_verified'] > 0
print(len(df_indicators['indicator'].unique()))
print(len(df_indicators[mask]['indicator'].unique()))


# # Export Deduplicated Unique Indicators
# 
# Deduplicated list of 12,622 unique indicator strings for Stage 2 embedding input. Each indicator appears exactly once regardless of how many wordplay types it is associated with.

# In[ ]:


# Deduplicated list of unique indicator strings for Stage 2 embedding input
mask = df_indicators['num_clues_verified'] > 0
unique_indicators = (
    df_indicators[mask]['indicator']
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)
unique_indicators.to_csv(
    DATA_DIR / 'verified_indicators_unique.csv',
    index=False,
    header=['indicator']
)
print(f"Saved {len(unique_indicators)} unique indicator strings to verified_indicators_unique.csv")


# In[63]:


df_ind_counts.style.format('{:,}')


# In[64]:


df_ind_counts.sum().to_frame().T.style.format('{:,}')


# In[ ]:





# # Task
# Create a DataFrame named `df_alternation` containing only indicators related to 'alternation' wordplay. This DataFrame should include the 'wordplay', 'indicator', and 'clue_ids_verified' columns, where 'clue_ids_verified' is a refined list of `clue_id`s for each indicator that corresponds to clues definitively identified as 'alternation' wordplay in `df_clues`. Finally, display the first few rows of `df_alternation`.

# ## Extract Alternation Clue IDs
# 
# ### Subtask:
# Create a set of `clue_id`s from `df_clues` where the 'alternation' column is True. This set will be used for efficient lookup.
# 

# **Reasoning**:
# To efficiently store and retrieve clue IDs that exhibit alternation wordplay, I will filter the `df_clues` DataFrame to include only rows where the 'alternation' column is True, then extract the 'clue_id' column and convert it into a Python set.
# 
# 

# In[65]:


alternation_clue_ids = set(df_clues[df_clues['alternation'] == True]['clue_id'])


# **Reasoning**:
# Now that we have a set of verified `clue_id`s for alternation wordplay, the next step is to filter the `df_indicators` DataFrame to include only indicators related to 'alternation' wordplay and then refine their `clue_ids_verified` lists to only contain `clue_id`s that are also present in the `alternation_clue_ids` set. This ensures that the indicators truly correspond to the 'alternation' wordplay identified in `df_clues`.
# 
# 

# In[66]:


df_alternation = df_indicators[df_indicators['wordplay'] == 'alternation'].copy()

df_alternation['clue_ids_verified'] = df_alternation['clue_ids_verified'].apply(
    lambda x: [clue_id for clue_id in x if clue_id in alternation_clue_ids]
)

# Filter out rows where clue_ids_verified is empty after refinement
df_alternation = df_alternation[df_alternation['clue_ids_verified'].apply(len) > 0]

df_alternation = df_alternation[['wordplay', 'indicator', 'clue_ids_verified']]

df_alternation.head()


# In[67]:


df_alternation.shape[0]


# In[68]:


df_indicators[df_indicators['wordplay'] == 'alternation'].shape[0]


# In[69]:


count_unique_clues(df_alternation['clue_ids_verified'])


# In[70]:


count_unique_clues(df_indicators['clue_ids_verified'])


# In[71]:


count_unique_clues(df_indicators[df_indicators['wordplay'] == 'alternation']['clue_ids_verified'])


# In[ ]:





# ## Summary:
# 
# ### Data Analysis Key Findings
# 
# *   A set named `alternation_clue_ids` was successfully created, containing 153 unique `clue_id`s corresponding to clues definitively identified as 'alternation' wordplay from the `df_clues` DataFrame.
# *   A new DataFrame, `df_alternation`, was successfully constructed. It initially filtered `df_indicators` for rows where the 'wordplay' column was 'alternation'.
# *   The `clue_ids_verified` column in `df_alternation` was refined to include only those `clue_id`s that were present in the `alternation_clue_ids` set, ensuring that each indicator is linked exclusively to verified 'alternation' clues.
# *   Rows in `df_alternation` where the `clue_ids_verified` list became empty after refinement were removed, ensuring that all remaining indicators are associated with at least one verified alternation clue.
# *   The final `df_alternation` DataFrame contains only the 'wordplay', 'indicator', and `clue_ids_verified` columns, with entries like 'after regular excisions', 'alternately', and 'alternatives' as indicators.
# 
# ### Insights or Next Steps
# 
# *   The `df_alternation` DataFrame now provides a clean, verified dataset of alternation wordplay indicators and their associated clue IDs, which can be used for training a model to identify 'alternation' wordplay or for further linguistic analysis.
# *   Further analysis could involve examining the commonality of specific indicators within the `clue_ids_verified` lists to understand which indicators are most frequently used for 'alternation' wordplay.
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[72]:


def check_anagram_in_surface(answer_no_spaces_text, surface_no_spaces_text):
    if not answer_no_spaces_text or not surface_no_spaces_text:
        return False

    answer_len = len(answer_no_spaces_text)
    if answer_len == 0:
        return False

    sorted_answer_chars = sorted(answer_no_spaces_text)

    for i in range(len(surface_no_spaces_text) - answer_len + 1):
        substring = surface_no_spaces_text[i : i + answer_len]
        if sorted(substring) == sorted_answer_chars:
            return True
    return False

# Apply this function to df_clues
df_clues['is_anagram_in_surface'] = df_clues.apply(
    lambda row: check_anagram_in_surface(row['answer_no_spaces'], row['surface_no_spaces']),
    axis=1
)

# Display the DataFrame with the new column
df_clues.head()


# In[73]:


df_clues[df_clues['is_anagram_in_surface']].sample(10)


# In[74]:


clue_info(261488)


# In[ ]:


# answer_letter_count was computed earlier (after hidden detection)
# Verify it exists
assert 'answer_letter_count' in df_clues.columns, "answer_letter_count missing from df_clues"


# In[76]:


df_clues['answer_letter_count'].value_counts(dropna=False)


# # Export Verified Clues with Labels
# 
# This cell produces `verified_clues_labeled.csv`: one row per verified (clue, indicator) pair, with both the original Ho blog label and an algorithmically determined ground-truth label.
# 
# ## Schema
# 
# | Column | Description |
# |--------|-------------|
# | `clue_id` | The clue ID from `df_clues` |
# | `indicator` | The verified indicator string |
# | `wordplay_ho` | The wordplay type label assigned by George Ho's blog parser (one of: alternation, anagram, container, deletion, hidden, homophone, insertion, reversal) |
# | `wordplay_gt` | A ground-truth label derived from pattern detection on the clue surface. See priority ordering below. Null if no pattern fires or if `answer_letter_count < 4`. |
# | `wordplay_gt_all` | Comma-separated list of ALL ground-truth labels that fired before priority resolution (for debugging/analysis). Also null if `answer_letter_count < 4`. |
# | `answer_letter_count` | Number of letters in the answer, so downstream users can apply their own length filters |
# | `label_match` | Boolean: True if `wordplay_ho == wordplay_gt` |
# 
# ## Ground-truth priority ordering
# 
# When multiple patterns fire for the same clue, `wordplay_gt` is assigned by this priority:
# 
# 1. **hidden** (if `hidden_fwd` is True) — most constrained: the exact answer letters appear consecutively in the surface
# 2. **reversal** (if `hidden_rev` is True) — same constraint but letters appear in reverse
# 3. **alternation** (if `alternation` is True) — answer letters appear at every-other position
# 4. **anagram** (if `is_anagram_in_surface` is True) — loosest: any permutation of a surface substring matches the answer
# 
# Hidden takes precedence over anagram because hidden is a strict subset of anagram (any hidden word is trivially also an anagram of the same substring). Without this priority, hidden clues would be mislabeled as anagram. Similarly, reversal is a strict subset of anagram. Alternation is prioritized over anagram because it requires a specific letter pattern rather than any permutation.
# 
# Answers shorter than 4 letters are excluded from ground-truth labeling because short answers produce many false-positive pattern matches (e.g., a 2-letter answer is easily "hidden" in any surface by coincidence).
# 
# ## Note on duplicate rows
# 
# The same `(clue_id, indicator)` pair may appear multiple times if the same indicator string is listed under different wordplay types in `df_indicators`. Each such row will have a different `wordplay_ho`. This preserves the multi-label structure of the data.

# In[ ]:


# === Step A: Explode clue_ids_verified and join with df_clues ===

# Start with verified indicators only
df_export = (
    df_indicators[df_indicators['num_clues_verified'] > 0]
    [['wordplay', 'indicator', 'clue_ids_verified']]
    .copy()
)

# Explode so each row is one (indicator, clue_id) pair
df_export = df_export.explode('clue_ids_verified').rename(
    columns={'clue_ids_verified': 'clue_id', 'wordplay': 'wordplay_ho'}
)

# Ensure clue_id is int for the merge
df_export['clue_id'] = df_export['clue_id'].astype(int)

# Merge with df_clues to get pattern detection columns and answer length
clue_cols = ['clue_id', 'hidden_fwd', 'hidden_rev', 'alternation',
             'is_anagram_in_surface', 'answer_letter_count']
df_export = df_export.merge(df_clues[clue_cols], on='clue_id', how='left')

print(f"Rows after explode + merge: {len(df_export):,}")
print(f"Unique clue_ids: {df_export['clue_id'].nunique():,}")
print(f"Unique indicators: {df_export['indicator'].nunique():,}")

# === Step B: Compute ground-truth labels ===

# Gate all ground truth on answer length >= 4
length_ok = df_export['answer_letter_count'] >= 4

# Define pattern columns and their corresponding labels, in priority order
gt_checks = [
    ('hidden_fwd', 'hidden'),
    ('hidden_rev', 'reversal'),
    ('alternation', 'alternation'),
    ('is_anagram_in_surface', 'anagram'),
]

# wordplay_gt_all: all labels that fired (comma-separated), null if none or short answer
fired_labels = pd.DataFrame({
    label: df_export[col].fillna(False) & length_ok
    for col, label in gt_checks
})
df_export['wordplay_gt_all'] = fired_labels.apply(
    lambda row: ','.join(col for col in fired_labels.columns if row[col]) or None,
    axis=1
)

# wordplay_gt: single winning label using priority order (first match wins)
df_export['wordplay_gt'] = None
for col, label in reversed(gt_checks):
    mask = df_export[col].fillna(False) & length_ok
    df_export.loc[mask, 'wordplay_gt'] = label

# label_match: does the Ho label agree with our ground-truth label?
df_export['label_match'] = df_export['wordplay_ho'] == df_export['wordplay_gt']

# === Step C: Select final columns and save ===

final_cols = ['clue_id', 'indicator', 'wordplay_ho', 'wordplay_gt',
              'wordplay_gt_all', 'answer_letter_count', 'label_match']
df_export = df_export[final_cols]

df_export.to_csv(DATA_DIR / 'verified_clues_labeled.csv', index=False)
print(f"\nSaved {len(df_export):,} rows to verified_clues_labeled.csv")


# In[ ]:


# === Summary statistics for verified_clues_labeled.csv ===

print("=== wordplay_ho (Ho blog label) distribution ===")
print(df_export['wordplay_ho'].value_counts().to_string())

print(f"\n=== wordplay_gt (ground-truth label) distribution ===")
print(df_export['wordplay_gt'].value_counts(dropna=False).to_string())

print(f"\n=== label_match ===")
# Only meaningful where wordplay_gt is not null
has_gt = df_export['wordplay_gt'].notna()
print(f"Rows with a ground-truth label: {has_gt.sum():,} / {len(df_export):,}")
if has_gt.sum() > 0:
    match_rate = df_export.loc[has_gt, 'label_match'].mean()
    print(f"Label match rate (where GT exists): {match_rate:.1%}")

print(f"\n=== Duplicate (clue_id, indicator) pairs (multi-label cases) ===")
dupes = df_export.duplicated(subset=['clue_id', 'indicator'], keep=False)
print(f"Rows involved in multi-label pairs: {dupes.sum():,}")
print(f"Unique (clue_id, indicator) pairs with >1 wordplay_ho: "
      f"{df_export[dupes].groupby(['clue_id', 'indicator']).ngroups:,}")

print(f"\n=== Sample rows ===")
df_export.sample(5, random_state=42)


# # Row Count Reconciliation
# 
# The raw dataset contains **~93,867 total indicator instances** across all clues (one row per clue-indicator pair in `df_ind_by_clue`). These span 15,735 unique indicator strings across 8 wordplay types.
# 
# Victoria's **checksum verification** filters this to only instances where the indicator phrase appears as intact words in the normalized clue surface text. This removes misparsed indicators (e.g., blogger formatting artifacts, partial word matches) and leaves **14,195 verified (wordplay, indicator) pairs** in `df_indicators` — covering **12,622 unique indicator strings** (the difference reflects indicators like "about" that appear under multiple wordplay types).
# 
# The final export, `verified_clues_labeled.csv`, explodes each verified indicator's list of verified clue IDs to produce one row per (clue, indicator) pair. This yields **76,015 rows** covering **70,959 unique clues**. The difference between unique clues (70,959) and total rows (76,015) reflects clues that contributed more than one verified indicator — for example, a clue that uses both an anagram indicator and a container indicator will appear as two separate rows.

# # New Section
