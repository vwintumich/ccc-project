"""Shared utilities for locating cryptic-clue definitions inside surface text.

This module provides the canonical definition-finding and delimiter-placement
logic used by both ``structural_filtering.ipynb`` (to validate that a
definition appears as an intact whole word in the surface) and component
phrase-construction notebooks (to wrap that definition in ``<t></t>``
delimiters when building f_clue phrases). Centralizing the logic here
ensures filtering and phrase construction never disagree.

Public API
----------
- ``find_definition_in_surface(definition, surface) -> (start, end) | None``
- ``tag_definition_in_surface(definition, surface) -> str | None``

Both functions return spans / strings in the *original* ``surface``
coordinates. Normalization (lowercasing, accent stripping, punctuation
removal) is applied internally only for the matching step — never carried
through to the returned value, since downstream embedding models perform
best on naturalistic text that preserves capitalization and punctuation.
"""

import re
import unicodedata


# Apostrophe characters (straight ASCII and curly Unicode) treated as
# possessive markers when validating ``<word>'s`` matches.
_APOSTROPHES = ("'", "\u2019")


def _normalize(s):
    """Lowercase, strip accents and punctuation; keep letters, digits, spaces.

    Used internally for matching only — never returned to callers, since
    downstream embeddings prefer the original surface form. Mirrors the
    ``normalize()`` helper from
    ``clue_misdirection/notebooks/01_data_cleaning.ipynb``.

    Examples
    --------
        _normalize("LA-DI-DA")    -> "ladida"
        _normalize("café")        -> "cafe"
        _normalize("Don't stop!") -> "dont stop"
    """
    return "".join(
        ch
        for ch in unicodedata.normalize("NFD", s)
        if unicodedata.category(ch).startswith(("L", "N", "Zs"))
    ).lower()


def _normalize_with_index_map(s):
    """Normalize ``s`` while recording, for each output character, the index
    of the original character in ``s`` it was derived from.

    Returns
    -------
    (normalized, index_map) where ``len(index_map) == len(normalized)`` and
    ``index_map[i]`` is the position in the original ``s`` of the i-th
    normalized character. This lets a regex match on the normalized form
    be translated back to a span in the original surface.
    """
    out_chars = []
    index_map = []
    for i, ch in enumerate(s):
        # NFD-decompose this single original character. A precomposed
        # accented letter (e.g. "é") expands to multiple chars; we keep
        # only base letters / digits / spaces and map them all back to ``i``.
        for d in unicodedata.normalize("NFD", ch):
            if unicodedata.category(d).startswith(("L", "N", "Zs")):
                out_chars.append(d.lower())
                index_map.append(i)
    return "".join(out_chars), index_map


def find_definition_in_surface(definition, surface):
    """Locate ``definition`` inside ``surface`` using word-boundary matching.

    Matching is performed on a normalized form of both strings (lowercase,
    accent-stripped, punctuation-stripped) but the returned span is in
    *original* surface coordinates so callers can slice or wrap the surface
    without distortion.

    The match accepts ``<word>'s`` (straight or curly apostrophe) as a
    valid match for ``<word>``. In that case the returned span covers only
    the ``<word>`` portion — the trailing ``'s`` is excluded so that
    delimiter placement produces e.g. ``<t>Bear</t>'s`` rather than
    ``<t>Bear's</t>``.

    Parameters
    ----------
    definition : str
        The definition substring to locate.
    surface : str
        The clue surface text.

    Returns
    -------
    tuple[int, int] or None
        ``(start, end)`` indices into the original ``surface``, or ``None``
        if no valid match exists. ``surface[start:end]`` recovers the
        original-case definition substring.
    """
    if not isinstance(definition, str) or not isinstance(surface, str):
        return None

    # Collapse internal whitespace before normalizing so that a definition
    # like "old  man" still matches "old man" in the surface.
    def_clean = re.sub(r"\s+", " ", definition).strip()
    def_norm = _normalize(def_clean).strip()
    if not def_norm:
        return None

    surf_norm, idx_map = _normalize_with_index_map(surface)
    if not surf_norm:
        return None

    escaped = re.escape(def_norm)

    # 1. Plain whole-word match: \b<def>\b on the normalized surface.
    m = re.search(r"\b" + escaped + r"\b", surf_norm)
    if m:
        start_norm, end_norm = m.start(), m.end()
        return (idx_map[start_norm], idx_map[end_norm - 1] + 1)

    # 2. Possessive match: in the normalized surface, "<word>'s" collapses
    #    to "<word>s" because apostrophes are stripped. Look for the
    #    definition immediately followed by an "s" at a word boundary, then
    #    confirm in the *original* surface that the character preceding
    #    that "s" is actually an apostrophe (not a plain plural).
    for m in re.finditer(r"\b" + escaped + r"s\b", surf_norm):
        start_norm, end_norm = m.start(), m.end()
        s_orig_idx = idx_map[end_norm - 1]
        if s_orig_idx > 0 and surface[s_orig_idx - 1] in _APOSTROPHES:
            # Span covers only the <word> portion, not the trailing "'s".
            word_end_norm = end_norm - 1
            return (idx_map[start_norm], idx_map[word_end_norm - 1] + 1)

    return None


def tag_definition_in_surface(definition, surface):
    """Wrap the located definition span in ``<t></t>`` delimiters.

    Returns the tagged surface string with original capitalization and
    punctuation preserved, or ``None`` if the definition cannot be located
    by :func:`find_definition_in_surface`.
    """
    span = find_definition_in_surface(definition, surface)
    if span is None:
        return None
    start, end = span
    return surface[:start] + "<t>" + surface[start:end] + "</t>" + surface[end:]
