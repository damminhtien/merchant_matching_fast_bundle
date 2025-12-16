from __future__ import annotations

from typing import Any

import pandas as pd

from domain import SUFFIX_CANDIDATES
from parsing import get_similarity_tokens, parse_merchant


def build_block_key(parsed) -> str:
    t = parsed.mtype.value
    c = parsed.core or ""
    return f"{t}|{c}"


def prepare_blocking_dataframe(df: pd.DataFrame, col_name: str, source_label: str) -> pd.DataFrame:
    records = []
    for idx, raw in df[col_name].items():
        parsed = parse_merchant(raw, SUFFIX_CANDIDATES)
        records.append(
            {
                "source": source_label,
                "row_id": idx,
                "raw_name": parsed.raw_name,
                "normalized": parsed.normalized,
                "merchant_type": parsed.mtype.value,
                "core": parsed.core,
                "suffix": " ".join(parsed.suffix_tokens),
                "location_key": parsed.location_key,
                "block_key": build_block_key(parsed),
                "sim_tokens": get_similarity_tokens(parsed),
            }
        )
    return pd.DataFrame(records)


def prepare_blocking_polars(
    df: Any,
    col_name: str,
    source_label: str,
    suffix: str,
    pl: Any,
) -> Any:
    values = df[col_name].to_list()
    records = []
    for idx, raw in enumerate(values):
        parsed = parse_merchant(raw, SUFFIX_CANDIDATES)
        records.append(
            {
                "block_key": build_block_key(parsed),
                f"source{suffix}": source_label,
                f"row_id{suffix}": idx,
                f"raw_name{suffix}": parsed.raw_name,
                f"normalized{suffix}": parsed.normalized,
                f"merchant_type{suffix}": parsed.mtype.value,
                f"core{suffix}": parsed.core,
                f"suffix{suffix}": " ".join(parsed.suffix_tokens),
                f"location_key{suffix}": parsed.location_key,
                f"sim_tokens{suffix}": get_similarity_tokens(parsed),
            }
        )
    return pl.DataFrame(records)


def build_candidate_pairs(df_block_1: pd.DataFrame, df_block_2: pd.DataFrame) -> pd.DataFrame:
    candidates = df_block_1.merge(
        df_block_2,
        on="block_key",
        how="inner",
        suffixes=("_1", "_2"),
    )
    loc1 = candidates["location_key_1"].fillna("")
    loc2 = candidates["location_key_2"].fillna("")
    mask = (loc1 == "") | (loc2 == "") | (loc1 == loc2)
    return candidates[mask].reset_index(drop=True)


def build_candidate_pairs_polars(df_block_1: Any, df_block_2: Any, pl: Any) -> Any:
    candidates = df_block_1.join(df_block_2, on="block_key", how="inner")
    return candidates.filter(
        (pl.col("location_key_1") == "")
        | (pl.col("location_key_2") == "")
        | (pl.col("location_key_1") == pl.col("location_key_2"))
    )


def prepare_candidates_polars(
    input_path: str,
    col_name_1: str,
    col_name_2: str,
) -> tuple[Any, int, int, Any]:
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "Polars engine requested but polars is not installed. Please install polars."
        ) from exc

    df = pl.read_csv(input_path)
    df_b1 = prepare_blocking_polars(df, col_name_1, "col1", "_1", pl)
    df_b2 = prepare_blocking_polars(df, col_name_2, "col2", "_2", pl)

    candidates = build_candidate_pairs_polars(df_b1, df_b2, pl)
    return candidates, len(df_b1), len(df_b2), pl
