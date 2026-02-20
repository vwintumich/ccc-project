#!/usr/bin/env python
# coding: utf-8

# # Create .CSV raw data files from cryptic.SQLITE3

# Dataset created by George Ho.
# 
# Available for download: https://cryptics.georgeho.org/data.db

# This notebook takes the complete database downloaded from cryptics.georgeho.org, and outputs the six most relevant tables as .csv files. 
# 
# It changes the feature names so that they are descriptive and consistent across all tables:
# * `rowid` becomes `clue_id`, `ind_id`, or `charade_id`
# * `answer` becomes `charade_answer` when it refers to the charade, remains `answer` when referring to the clue answer
# * `clue_rowids` becomes `clue_ids` for consistency

# In[1]:


# imports
import sqlite3
import pandas as pd
import numpy as np


# In[ ]:


# Connect to the sqlite3 file
data_file = "../data/data.sqlite3"
conn = sqlite3.connect(data_file)


# In[3]:


# Uncomment to see what data tables exist in the file
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
#tables


# In[4]:


# Keep track of all tables that might be of interest from the original dataset
# Display the names and sizes of all tables.

tables = [
    "clues",
    "indicators",
    "charades",
    "indicators_by_clue",
    "charades_by_clue",
    "indicators_consolidated"
]

summary = []

for t in tables:
    # count rows
    row_count = pd.read_sql(f"SELECT COUNT(*) AS n FROM {t};", conn).iloc[0]["n"]

    # count rows and columns
    col_info = pd.read_sql(f"PRAGMA table_info({t});", conn)
    col_count = len(col_info)

    summary.append({
        "table": t,
        "rows": row_count,
        "columns": col_count
    })

summary_df = pd.DataFrame(summary)
summary_df.style.format({"rows": "{:,}"}) # display with commas 


# In[5]:


# Create the dataframes related to indicators
df_indicators = pd.read_sql("SELECT * FROM indicators;", conn)
df_ind_by_clue = pd.read_sql("SELECT * FROM indicators_by_clue;", conn)
df_indicators_consolidated = pd.read_sql("SELECT * FROM indicators_consolidated;", conn)

# Create dataframes pertaining to clue and charade
df_clues = pd.read_sql("SELECT * FROM clues;", conn)
df_charades = pd.read_sql("SELECT * FROM charades;", conn)
df_charades_by_clue = pd.read_sql("SELECT * FROM charades_by_clue;", conn)


# In[6]:


# Indicators
print(df_indicators.head(3))

# Rename columns for consistency across tables/dataframes
df_indicators = df_indicators.rename(columns={'rowid': 'ind_id', 'clue_rowids': 'clue_ids'})
print(df_indicators.head(3))


# In[7]:


# Indicators by Clue
print(df_ind_by_clue.head(4))
print()

# Rename columns for consistancy across tables/dataframes
df_ind_by_clue = df_ind_by_clue.rename(columns={'clue_rowid': 'clue_id'})
print(df_ind_by_clue.head(4))

# See how many contextualized indicators are in this table
#print("Instances of each CONTEXTUALIZED wordplay (multiple per clue, redundant indicators)")
#df_ind_by_clue.replace("", np.nan).count()


# In[8]:


# Indicators Consolidated
print(df_indicators_consolidated.head())
print()


# In[9]:


# Clues
print(df_clues.head(3))

# Rename columns for consistency across tables/dataframes
df_clues = df_clues.rename(columns={'rowid': 'clue_id'})
print(df_clues.head(3))


# In[10]:


# Charades
print(df_charades.head(3))

# Rename columns for consistency across tables/dataframes
df_charades = df_charades.rename(columns={'rowid': 'charade_id', 'answer':'charade_answer', 'clue_rowids': 'clue_ids'})
print(df_charades.head(3))


# In[11]:


# Charades by Clue
print(df_charades_by_clue.head(3))

# Rename columns for consistency across tables/dataframes
df_charades_by_clue = df_charades_by_clue.rename(columns={'clue_rowid': 'clue_id', 'answer':'charade_answer'})
print(df_charades_by_clue.head(3))


# In[12]:


# Write each dataframe to a CSV file in the data directory (without the index)
df_indicators.to_csv("../data/indicators_raw.csv", index=False)
df_ind_by_clue.to_csv("../data/indicators_by_clue_raw.csv", index=False)
df_indicators_consolidated.to_csv("../data/indicators_consolidated_raw.csv", index=False)
df_clues.to_csv("../data/clues_raw.csv", index=False)
df_charades.to_csv("../data/charades_raw.csv", index=False)
df_charades_by_clue.to_csv("../data/charades_by_clue_raw.csv", index=False)


# In[ ]:




