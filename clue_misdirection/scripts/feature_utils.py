"""Feature computation utilities extracted from NB 03 (03_feature_engineering.ipynb).

This module provides the same feature computation logic used in NB 03,
packaged for reuse when computing features for distractor pairs in
Steps 5 and 7 (dataset construction). See Decision 18 in DECISIONS.md:
NB 03 keeps all logic inline for grading readability; this module avoids
duplicating that logic across NB 05 and NB 07.

If a bug is found, update both NB 03 and this module.
"""

import numpy as np
import nltk
from nltk.corpus import wordnet as wn


# ============================================================
# Feature Group Column-Name Lists
# ============================================================
# These match the columns produced by NB 03 and stored in
# data/features_all.parquet.  Hardcoded here (rather than derived
# from a DataFrame) so the module is importable without side effects.

# --- Context-Free Meaning (15): cosine similarities not involving clue context ---
CONTEXT_FREE_COLS = sorted([
    'cos_w1all_w2all', 'cos_w1all_w2common', 'cos_w1all_w2obscure',
    'cos_w1common_w2all', 'cos_w1common_w2common', 'cos_w1common_w2obscure',
    'cos_w1obscure_w2all', 'cos_w1obscure_w2common', 'cos_w1obscure_w2obscure',
    'cos_w1all_w1common', 'cos_w1all_w1obscure', 'cos_w1common_w1obscure',
    'cos_w2all_w2common', 'cos_w2all_w2obscure', 'cos_w2common_w2obscure',
])

# --- Context-Informed Meaning (6): cosine similarities involving w1clue ---
CONTEXT_INFORMED_COLS = sorted([
    'cos_w1clue_w1all', 'cos_w1clue_w1common', 'cos_w1clue_w1obscure',
    'cos_w1clue_w2all', 'cos_w1clue_w2common', 'cos_w1clue_w2obscure',
])

# --- Relationship (22): 20 boolean + path similarity + shared synset count ---
RELATIONSHIP_COLS = (
    sorted([
        'wn_rel_synonym',
        'wn_rel_hyponym', 'wn_rel_hypernym',
        'wn_rel_part_holonym', 'wn_rel_part_meronym',
        'wn_rel_substance_meronym', 'wn_rel_member_meronym',
        'wn_rel_hyponym_of_hypernym', 'wn_rel_hypernym_of_hyponym',
        'wn_rel_hyponym_of_hyponym', 'wn_rel_hypernym_of_hypernym',
        'wn_rel_part_holonym_of_hyponym', 'wn_rel_hyponym_of_part_holonym',
        'wn_rel_substance_meronym_of_hyponym',
        'wn_rel_part_meronym_of_hyponym', 'wn_rel_hyponym_of_part_meronym',
        'wn_rel_part_meronym_of_hypernym', 'wn_rel_part_holonym_of_hypernym',
        'wn_rel_part_holonym_of_part_meronym',
        'wn_rel_member_meronym_of_member_holonym',
    ])
    + ['wn_max_path_sim', 'wn_shared_synset_count']
)

# --- Surface (4): orthographic similarity ---
SURFACE_COLS = [
    'surface_edit_distance', 'surface_length_ratio',
    'surface_shared_first_letter', 'surface_char_overlap_ratio',
]

# --- All 47 features ---
ALL_FEATURE_COLS = (
    CONTEXT_FREE_COLS + CONTEXT_INFORMED_COLS
    + RELATIONSHIP_COLS + SURFACE_COLS
)

# --- Metadata columns (not features, carried through for joins) ---
METADATA_COLS = [
    'clue_id', 'clue', 'surface', 'surface_normalized',
    'definition', 'answer', 'definition_wn', 'answer_wn',
    'def_answer_pair_id', 'answer_format', 'num_definitions',
    'def_num_usable_synsets', 'ans_num_usable_synsets',
]


# ============================================================
# Cosine Similarity
# ============================================================

def rowwise_cosine(A, B):
    """Compute row-wise cosine similarity between corresponding rows of A and B.

    Parameters
    ----------
    A, B : np.ndarray, shape (N, D)
        Two matrices with the same shape.

    Returns
    -------
    np.ndarray, shape (N,)
        Cosine similarity for each row pair: cos(A[i], B[i]).
    """
    # L2-normalize each row, then take the row-wise dot product.
    A_norm = np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = np.linalg.norm(B, axis=1, keepdims=True)
    return np.sum((A / A_norm) * (B / B_norm), axis=1)


# ============================================================
# Surface Features (4)
# ============================================================

def compute_surface_features(def_word, ans_word):
    """Compute all 4 surface features for a (definition, answer) pair.

    Surface features measure shallow orthographic similarity between words.

    Parameters
    ----------
    def_word : str
        The definition word (from the definition_wn column).
    ans_word : str
        The answer word (from the answer_wn column).

    Returns
    -------
    dict
        Keys are feature column names, values are feature values:
        - surface_edit_distance (int): Levenshtein edit distance
        - surface_length_ratio (float): len(shorter) / len(longer), in [0, 1]
        - surface_shared_first_letter (int): 1 if same first character, else 0
        - surface_char_overlap_ratio (float): Jaccard similarity of character sets
    """
    features = {}

    # --- Levenshtein edit distance ---
    # Uses NLTK's edit_distance which computes the standard dynamic-programming
    # Levenshtein distance (insertions, deletions, substitutions each cost 1).
    features['surface_edit_distance'] = nltk.edit_distance(def_word, ans_word)

    # --- Length ratio: len(shorter) / len(longer) ---
    # Always in [0, 1]. A value of 1.0 means identical lengths.
    # Default to 0.0 if both strings are empty (should not occur given
    # WordNet filter, but be safe).
    len_def = len(def_word)
    len_ans = len(ans_word)
    if len_def == 0 and len_ans == 0:
        features['surface_length_ratio'] = 0.0
    else:
        shorter = min(len_def, len_ans)
        longer = max(len_def, len_ans)
        features['surface_length_ratio'] = shorter / longer

    # --- Shared first letter ---
    # Binary indicator: do the words start with the same character?
    if def_word and ans_word:
        features['surface_shared_first_letter'] = int(
            def_word[0] == ans_word[0])
    else:
        features['surface_shared_first_letter'] = 0

    # --- Character overlap ratio (Jaccard similarity of character sets) ---
    # Measures what fraction of distinct characters appear in both words.
    # For example, "plant" {p,l,a,n,t} and "aster" {a,s,t,e,r} share
    # {a, t} out of {p,l,a,n,t,s,e,r} → 2/8 = 0.25.
    def_chars = set(def_word)
    ans_chars = set(ans_word)
    union = def_chars | ans_chars
    if len(union) == 0:
        features['surface_char_overlap_ratio'] = 0.0
    else:
        features['surface_char_overlap_ratio'] = (
            len(def_chars & ans_chars) / len(union))

    return features


# ============================================================
# WordNet Relationship Features (22)
# ============================================================

def get_wordnet_synsets(word):
    """Look up all WordNet synsets for a word, handling multi-word entries.

    Tries the word as-is first (which works for both single-word and
    underscore-separated multi-word entries like "ice_cream"). If no
    synsets are found and the word contains spaces, retries with spaces
    converted to underscores — WordNet's convention for multi-word entries.

    Parameters
    ----------
    word : str
        The word to look up (from the definition_wn or answer_wn column).

    Returns
    -------
    list of nltk.corpus.reader.wordnet.Synset
        All synsets found for the word. Empty list if no synsets exist.
    """
    synsets = wn.synsets(word)
    if not synsets and ' ' in word:
        synsets = wn.synsets(word.replace(' ', '_'))
    return synsets


def _check_synonym(def_synsets, ans_word):
    """Check if the answer word is a synonym of the definition in WordNet.

    Two words are WordNet synonyms if they share at least one synset — i.e.,
    the answer word appears as a lemma in some synset of the definition word.
    """
    for syn in def_synsets:
        lemma_names = {lemma.name().lower() for lemma in syn.lemmas()}
        if ans_word in lemma_names:
            return True
    return False


def _check_synset_reachable(def_synsets, ans_synsets_set, hops):
    """Check if any answer synset is reachable from definition synsets via
    a sequence of WordNet relationship hops.

    For single-hop relationships (e.g., "hyponym"), ``hops`` contains one
    method name. For compound two-hop relationships (e.g.,
    "hyponym_of_hypernym"), ``hops`` contains two method names: the first
    hop follows the second word of the compound name (hypernyms), then the
    second hop follows the first word (hyponyms). This right-to-left
    reading matches the English semantics.
    """
    current = set(def_synsets)
    for method_name in hops:
        next_level = set()
        for synset in current:
            next_level.update(getattr(synset, method_name)())
        current = next_level
        if not current:
            return False
    return bool(current & ans_synsets_set)


def _compute_max_path_similarity(def_synsets, ans_synsets):
    """Compute the maximum path similarity across all definition-answer
    synset pairs.

    Path similarity (Rada et al., 1989) measures the inverse of the shortest
    path length between two synsets in the WordNet hypernym/hyponym hierarchy,
    normalized to [0, 1].
    """
    max_sim = 0.0
    for ds in def_synsets:
        for as_ in ans_synsets:
            sim = ds.path_similarity(as_)
            if sim is not None and sim > max_sim:
                max_sim = sim
    return max_sim


def _compute_shared_synset_count(def_synsets, ans_synsets):
    """Count synsets that contain both the definition and answer as lemmas."""
    return len(set(def_synsets) & set(ans_synsets))


# --- Relationship type definitions ---
# Maps each relationship type name to the sequence of WordNet synset methods
# needed to check reachability. For compound types "X_of_Y", the first hop
# follows Y and the second hop follows X (read right-to-left).
_RELATIONSHIP_HOPS = {
    # --- Single-hop relationships (6) ---
    'hyponym':           ['hyponyms'],
    'hypernym':          ['hypernyms'],
    'part_holonym':      ['part_holonyms'],
    'part_meronym':      ['part_meronyms'],
    'substance_meronym': ['substance_meronyms'],
    'member_meronym':    ['member_meronyms'],

    # --- Two-hop relationships (13) ---
    'hyponym_of_hypernym':  ['hypernyms', 'hyponyms'],   # co-hyponymy
    'hypernym_of_hyponym':  ['hyponyms', 'hypernyms'],   # co-hypernymy
    'hyponym_of_hyponym':   ['hyponyms', 'hyponyms'],    # grandchild
    'hypernym_of_hypernym': ['hypernyms', 'hypernyms'],  # grandparent

    'part_holonym_of_hyponym':      ['hyponyms', 'part_holonyms'],
    'hyponym_of_part_holonym':      ['part_holonyms', 'hyponyms'],
    'substance_meronym_of_hyponym': ['hyponyms', 'substance_meronyms'],
    'part_meronym_of_hyponym':      ['hyponyms', 'part_meronyms'],
    'hyponym_of_part_meronym':      ['part_meronyms', 'hyponyms'],
    'part_meronym_of_hypernym':     ['hypernyms', 'part_meronyms'],
    'part_holonym_of_hypernym':     ['hypernyms', 'part_holonyms'],

    'part_holonym_of_part_meronym':     ['part_meronyms', 'part_holonyms'],
    'member_meronym_of_member_holonym': ['member_holonyms', 'member_meronyms'],
}


def compute_relationship_features(def_word, ans_word):
    """Compute all 22 WordNet relationship features for a (definition, answer) pair.

    This is the master function that combines all relationship checks into
    a single feature dictionary.

    Parameters
    ----------
    def_word : str
        The definition word (lowercase, underscored for multi-word; from the
        definition_wn column).
    ans_word : str
        The answer word (lowercase, underscored for multi-word; from the
        answer_wn column).

    Returns
    -------
    dict
        Keys are feature column names, values are feature values:
        - 20 boolean features (int 0/1): wn_rel_synonym, wn_rel_hyponym, ...
        - wn_max_path_sim (float): max path similarity, default 0.0
        - wn_shared_synset_count (int): shared synset count, default 0
    """
    def_synsets = get_wordnet_synsets(def_word)
    ans_synsets = get_wordnet_synsets(ans_word)
    ans_synsets_set = set(ans_synsets)

    features = {}

    # --- Synonym (lemma-based, one hop) ---
    features['wn_rel_synonym'] = int(_check_synonym(def_synsets, ans_word))

    # --- Synset-based relationship checks (one-hop and two-hop) ---
    for rel_name, hops in _RELATIONSHIP_HOPS.items():
        features[f'wn_rel_{rel_name}'] = int(
            _check_synset_reachable(def_synsets, ans_synsets_set, hops)
        )

    # --- Max path similarity ---
    features['wn_max_path_sim'] = _compute_max_path_similarity(
        def_synsets, ans_synsets)

    # --- Shared synset count ---
    features['wn_shared_synset_count'] = _compute_shared_synset_count(
        def_synsets, ans_synsets)

    return features


# ============================================================
# Cosine Feature Convenience Function (for distractor pairs)
# ============================================================

# Embedding type names used in the column naming scheme.
_DEF_EMB_TYPES = ['w1all', 'w1common', 'w1obscure']
_ANS_EMB_TYPES = ['w2all', 'w2common', 'w2obscure']


def compute_cosine_features_for_pair(def_embs_dict, ans_embs_dict,
                                     clue_emb=None):
    """Compute cosine similarity features for a single (definition, answer) pair.

    This is a convenience wrapper around the same cosine logic used in NB 03
    (rowwise_cosine), adapted for computing features one pair at a time when
    constructing distractor rows in Steps 5 and 7.

    Parameters
    ----------
    def_embs_dict : dict of str → np.ndarray (1D, shape (D,))
        Definition embeddings keyed by type name. Expected keys:
        ``'allsense'``, ``'common'``, ``'obscure'``.
    ans_embs_dict : dict of str → np.ndarray (1D, shape (D,))
        Answer embeddings keyed by type name. Expected keys:
        ``'allsense'``, ``'common'``, ``'obscure'``.
    clue_emb : np.ndarray (1D, shape (D,)), optional
        The word1_clue_context embedding. If provided, the 6 context-informed
        cosine features are also computed; otherwise only the 15 context-free
        features are returned.

    Returns
    -------
    dict of str → float
        Feature name → cosine similarity value. Contains 15 features if
        ``clue_emb`` is None, or 21 features if ``clue_emb`` is provided.
    """
    # Map from column-name embedding shorthand to dict key.
    _emb_key = {
        'w1all': ('def', 'allsense'),
        'w1common': ('def', 'common'),
        'w1obscure': ('def', 'obscure'),
        'w2all': ('ans', 'allsense'),
        'w2common': ('ans', 'common'),
        'w2obscure': ('ans', 'obscure'),
    }

    def _get_vec(name):
        side, key = _emb_key[name]
        return def_embs_dict[key] if side == 'def' else ans_embs_dict[key]

    def _cosine(a, b):
        """Cosine similarity between two 1D vectors (matching rowwise_cosine logic)."""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a / a_norm, b / b_norm))

    features = {}

    # --- All 6 context-free embedding types ---
    all_types = _DEF_EMB_TYPES + _ANS_EMB_TYPES  # 6 total

    # --- Context-Free Meaning: C(6,2) = 15 pairwise cosine similarities ---
    for i in range(len(all_types)):
        for j in range(i + 1, len(all_types)):
            name_a = all_types[i]
            name_b = all_types[j]
            col = f'cos_{name_a}_{name_b}'
            features[col] = _cosine(_get_vec(name_a), _get_vec(name_b))

    # --- Context-Informed Meaning: 6 features (only if clue_emb provided) ---
    if clue_emb is not None:
        clue_norm = np.linalg.norm(clue_emb)
        if clue_norm == 0:
            clue_unit = clue_emb
        else:
            clue_unit = clue_emb / clue_norm

        for emb_name in all_types:
            col = f'cos_w1clue_{emb_name}'
            vec = _get_vec(emb_name)
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                features[col] = 0.0
            else:
                features[col] = float(np.dot(clue_unit, vec / vec_norm))

    return features
