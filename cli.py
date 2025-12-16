from __future__ import annotations

import argparse
from datetime import datetime

from blocking import (
    build_candidate_pairs,
    prepare_blocking_dataframe,
    prepare_candidates_polars,
)
from config import DEFAULT_CONFIG, VERSION
from similarity import (
    classify_matches,
    compute_similarity_and_classify_polars,
    compute_similarity_df,
)


def core_pipeline_pandas(
    input_path: str,
    col_name_1: str,
    col_name_2: str,
    output_path: str,
    high_thr: float,
    low_thr: float,
    alpha: float,
) -> dict:
    import pandas as pd

    df = pd.read_csv(input_path)
    df_b1 = prepare_blocking_dataframe(df, col_name_1, "col1")
    df_b2 = prepare_blocking_dataframe(df, col_name_2, "col2")
    num_b1, num_b2 = len(df_b1), len(df_b2)

    candidates = build_candidate_pairs(df_b1, df_b2)
    candidates = compute_similarity_df(candidates, alpha=alpha)
    candidates = classify_matches(candidates, high_thr=high_thr, low_thr=low_thr)

    meta_ts = datetime.utcnow().isoformat()
    candidates = candidates.assign(
        engine="pandas",
        alpha=alpha,
        high_thr=high_thr,
        low_thr=low_thr,
        timestamp=meta_ts,
        version=VERSION,
    )

    candidates.sort_values("sim_final", ascending=False, inplace=True)
    candidates.to_csv(output_path, index=False)

    return {
        "num_b1": num_b1,
        "num_b2": num_b2,
        "num_candidates": len(candidates),
        "num_block_keys": candidates["block_key"].nunique(),
        "num_match": (candidates["match_label"] == "MATCH").sum(),
        "num_review": (candidates["match_label"] == "REVIEW").sum(),
        "num_non_match": (candidates["match_label"] == "NON_MATCH").sum(),
    }


def core_pipeline_polars(
    input_path: str,
    col_name_1: str,
    col_name_2: str,
    output_path: str,
    high_thr: float,
    low_thr: float,
    alpha: float,
) -> dict:
    candidates_pl, num_b1, num_b2, pl = prepare_candidates_polars(
        input_path=input_path,
        col_name_1=col_name_1,
        col_name_2=col_name_2,
    )
    candidates_pl = compute_similarity_and_classify_polars(
        candidates_pl,
        alpha=alpha,
        high_thr=high_thr,
        low_thr=low_thr,
        pl=pl,
    )
    candidates_pl = candidates_pl.with_columns(
        [
            pl.col("sim_tokens_1").list.join(" ").alias("sim_tokens_1"),
            pl.col("sim_tokens_2").list.join(" ").alias("sim_tokens_2"),
            pl.lit("polars").alias("engine"),
            pl.lit(alpha).alias("alpha"),
            pl.lit(high_thr).alias("high_thr"),
            pl.lit(low_thr).alias("low_thr"),
            pl.lit(datetime.utcnow().isoformat()).alias("timestamp"),
            pl.lit(VERSION).alias("version"),
        ]
    )
    candidates_pl = candidates_pl.sort("sim_final", descending=True)
    candidates_pl.write_csv(output_path)

    return {
        "num_b1": num_b1,
        "num_b2": num_b2,
        "num_candidates": len(candidates_pl),
        "num_block_keys": candidates_pl.select(pl.col("block_key").n_unique()).item(),
        "num_match": (candidates_pl["match_label"] == "MATCH").sum(),
        "num_review": (candidates_pl["match_label"] == "REVIEW").sum(),
        "num_non_match": (candidates_pl["match_label"] == "NON_MATCH").sum(),
    }


def run_matching(
    input_path: str,
    col_name_1: str = "Merchant_Name_1",
    col_name_2: str = "Merchant_Name_2",
    output_path: str = "merchant_matching_results_fast.csv",
    high_thr: float = DEFAULT_CONFIG["thresholds"]["high_thr"],
    low_thr: float = DEFAULT_CONFIG["thresholds"]["low_thr"],
    alpha: float = DEFAULT_CONFIG["thresholds"]["alpha"],
    engine: str = "pandas",
) -> None:
    engine = engine.lower()
    if engine not in {"pandas", "polars"}:
        raise ValueError("engine must be 'pandas' or 'polars'")

    if engine == "polars":
        stats = core_pipeline_polars(
            input_path=input_path,
            col_name_1=col_name_1,
            col_name_2=col_name_2,
            output_path=output_path,
            high_thr=high_thr,
            low_thr=low_thr,
            alpha=alpha,
        )
    else:
        stats = core_pipeline_pandas(
            input_path=input_path,
            col_name_1=col_name_1,
            col_name_2=col_name_2,
            output_path=output_path,
            high_thr=high_thr,
            low_thr=low_thr,
            alpha=alpha,
        )

    print("Done.")
    print(f"Records col1       : {stats['num_b1']}")
    print(f"Records col2       : {stats['num_b2']}")
    print(f"Candidate pairs    : {stats['num_candidates']}")
    print(f"Block keys         : {stats['num_block_keys']}")
    print(f"MATCH              : {stats['num_match']}")
    print(f"REVIEW             : {stats['num_review']}")
    print(f"NON_MATCH          : {stats['num_non_match']}")
    print(f"Output saved to    : {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast merchant name blocking + matching.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--col1", default="Merchant_Name_1", help="Column name for first merchant list.")
    parser.add_argument("--col2", default="Merchant_Name_2", help="Column name for second merchant list.")
    parser.add_argument("--output", default="merchant_matching_results_fast.csv", help="Output CSV path.")
    parser.add_argument("--high_thr", type=float, default=DEFAULT_CONFIG["thresholds"]["high_thr"], help="High threshold for MATCH.")
    parser.add_argument("--low_thr", type=float, default=DEFAULT_CONFIG["thresholds"]["low_thr"], help="Low threshold for REVIEW.")
    parser.add_argument("--alpha", type=float, default=DEFAULT_CONFIG["thresholds"]["alpha"], help="Weight for Jaccard vs Levenshtein.")
    parser.add_argument(
        "--engine",
        choices=["pandas", "polars"],
        default="pandas",
        help="Dataframe engine for blocking step.",
    )

    args = parser.parse_args()
    run_matching(
        input_path=args.input,
        col_name_1=args.col1,
        col_name_2=args.col2,
        output_path=args.output,
        high_thr=args.high_thr,
        low_thr=args.low_thr,
        alpha=args.alpha,
        engine=args.engine,
    )


if __name__ == "__main__":
    main()
