from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein as rf_lev


def jaccard_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 0.0


def levenshtein_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return rf_lev.normalized_similarity(a, b) / 100.0


def compute_similarity_df(candidates: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
    sims_j = []
    sims_l = []
    final = []

    sim_tokens_1 = candidates["sim_tokens_1"].tolist()
    sim_tokens_2 = candidates["sim_tokens_2"].tolist()

    for t1, t2 in zip(sim_tokens_1, sim_tokens_2):
        sj = jaccard_similarity(t1, t2)
        name1 = " ".join(t1)
        name2 = " ".join(t2)
        sl = levenshtein_similarity(name1, name2)
        sims_j.append(sj)
        sims_l.append(sl)
        final.append(alpha * sj + (1 - alpha) * sl)

    candidates = candidates.copy()
    candidates["sim_jaccard"] = sims_j
    candidates["sim_levenshtein"] = sims_l
    candidates["sim_final"] = final
    return candidates


def classify_matches(candidates: pd.DataFrame, high_thr: float = 0.75, low_thr: float = 0.4) -> pd.DataFrame:
    cand = candidates.copy()
    s = cand["sim_final"].to_numpy()

    labels = np.full(len(cand), "NON_MATCH", dtype=object)
    mask_review_zone = s >= low_thr
    labels[mask_review_zone] = "REVIEW"

    suf1 = cand["suffix_1"].fillna("").astype(str).str.extract(r"(\d+)\b", expand=False)
    suf2 = cand["suffix_2"].fillna("").astype(str).str.extract(r"(\d+)\b", expand=False)

    both_have = suf1.notna() & suf2.notna()
    diff_branch = both_have & (suf1 != suf2)

    mask_match = (s >= high_thr) & (~diff_branch)
    labels[mask_match] = "MATCH"

    cand["match_label"] = labels
    return cand


def compute_similarity_and_classify_polars(
    candidates: Any,
    alpha: float,
    high_thr: float,
    low_thr: float,
    pl: Any,
) -> Any:
    """Compute similarities + labels with Polars set ops + RapidFuzz Levenshtein."""
    base = candidates.with_columns(
        [
            pl.col("sim_tokens_1").list.join(" ").alias("sim_name_1"),
            pl.col("sim_tokens_2").list.join(" ").alias("sim_name_2"),
        ]
    )

    inter_len = pl.col("sim_tokens_1").list.set_intersection(pl.col("sim_tokens_2")).list.len()
    union_len = pl.col("sim_tokens_1").list.set_union(pl.col("sim_tokens_2")).list.len()
    sim_jaccard = pl.when(union_len == 0).then(1.0).otherwise(inter_len / union_len)

    sim_lev = pl.struct(["sim_name_1", "sim_name_2"]).map_elements(
        lambda row: rf_lev.normalized_similarity(row["sim_name_1"], row["sim_name_2"]) / 100.0
    )

    sim_final = alpha * sim_jaccard + (1.0 - alpha) * sim_lev

    suf1 = pl.col("suffix_1").fill_null("").str.extract(r"(\d+)", 1)
    suf2 = pl.col("suffix_2").fill_null("").str.extract(r"(\d+)", 1)
    both_have = suf1.is_not_null() & suf2.is_not_null()
    diff_branch = both_have & (suf1 != suf2)

    match_label = (
        pl.when(sim_final < low_thr)
        .then(pl.lit("NON_MATCH"))
        .when((sim_final >= high_thr) & (~diff_branch))
        .then(pl.lit("MATCH"))
        .otherwise(pl.lit("REVIEW"))
    )

    return base.with_columns(
        [
            sim_jaccard.alias("sim_jaccard"),
            sim_lev.alias("sim_levenshtein"),
            sim_final.alias("sim_final"),
            match_label.alias("match_label"),
        ]
    )
