from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein as rf_lev


class MerchantType(str, Enum):
    COMPANY_CT = "COMPANY_CT"
    HOUSEHOLD_HKD = "HOUSEH"
    PHARMACY = "PHARMACY"
    GAS = "GAS"
    SHOP = "SHOP"
    CAFE = "CAFE"
    RESTAURANT_QUAN = "REST_QUAN"
    HAIR_SALON = "HAIR_SALON"
    OFFICE_VP = "OFFICE_VP"
    OTHER = "OTHER"


GENERIC_TOKENS = {
    "CH", "CUA", "HANG", "TIEM",
    "SHOP", "STORE", "MART", "POS",
    "QUAN", "AN"
}

SUFFIX_CANDIDATES = {
    "BTL", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9",
    "Q10", "Q11", "Q12", "GO", "VAP", "GV", "OCP"
}


@dataclass
class ParsedMerchant:
    raw_name: str
    normalized: str
    tokens: List[str]
    mtype: MerchantType
    core: str
    suffix_tokens: List[str]
    location_key: str


@dataclass
class TypeRule:
    name: MerchantType
    detect_sequences: List[List[str]] = field(default_factory=list)
    detect_tokens_any: List[str] = field(default_factory=list)
    strip_prefix_sequences: List[List[str]] = field(default_factory=list)
    strip_prefix_tokens_any: List[str] = field(default_factory=list)
    strip_after_tokens_any: List[str] = field(default_factory=list)


DOMAIN_PREFIX_SEQUENCES = [
    ["VAN", "TAI"],
    ["TAP", "HOA"],
]

TYPE_RULES: List[TypeRule] = [
    TypeRule(
        name=MerchantType.HOUSEHOLD_HKD,
        detect_sequences=[["HO", "KINH", "DOANH"]],
        detect_tokens_any=["HKD"],
        strip_prefix_sequences=[["HO", "KINH", "DOANH"]],
        strip_prefix_tokens_any=["HKD"],
    ),
    TypeRule(
        name=MerchantType.PHARMACY,
        detect_sequences=[["NHA", "THUOC"]],
        strip_prefix_sequences=[["NHA", "THUOC"]],
    ),
    TypeRule(
        name=MerchantType.RESTAURANT_QUAN,
        detect_sequences=[["QUAN", "AN"], ["NHA", "HANG"]],
        strip_prefix_sequences=[["QUAN", "AN"], ["NHA", "HANG"]],
    ),
    TypeRule(
        name=MerchantType.HAIR_SALON,
        detect_sequences=[["SALON", "TOC"], ["TIEM", "TOC"]],
        strip_prefix_sequences=[["SALON", "TOC"], ["TIEM", "TOC"]],
    ),
    TypeRule(
        name=MerchantType.GAS,
        detect_tokens_any=["GAS"],
        strip_prefix_tokens_any=["GAS"],
    ),
    TypeRule(
        name=MerchantType.CAFE,
        detect_tokens_any=["CAFE", "COFFEE"],
        strip_after_tokens_any=["CAFE", "COFFEE"],
    ),
    TypeRule(
        name=MerchantType.SHOP,
        detect_sequences=[["TAP", "HOA"], ["CUA", "HANG"]],
        detect_tokens_any=["SHOP", "STORE", "MART"],
        strip_prefix_sequences=[["TAP", "HOA"], ["CUA", "HANG"]],
        strip_prefix_tokens_any=["CH", "TIEM"],
    ),
    TypeRule(
        name=MerchantType.OFFICE_VP,
        detect_tokens_any=["VP"],
        detect_sequences=[["VAN", "PHONG"]],
        strip_prefix_tokens_any=["VP"],
        strip_prefix_sequences=[["VAN", "PHONG"]],
    ),
    TypeRule(
        name=MerchantType.COMPANY_CT,
        detect_tokens_any=["CT", "CTY", "TNHH"],
        detect_sequences=[["CONG", "TY"]],
        strip_prefix_tokens_any=["CT", "CTY", "CONG", "TY", "TNHH"],
    ),
]

TYPE_RULE_MAP = {rule.name: rule for rule in TYPE_RULES}


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.upper().strip()
    s = s.replace("CO.OP", "COOP")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(name: str) -> List[str]:
    if not name:
        return []
    return name.split()


def _has_sequence(tokens: List[str], seq: List[str]) -> bool:
    if not tokens or not seq:
        return False
    n, m = len(tokens), len(seq)
    for i in range(n - m + 1):
        if tokens[i:i + m] == seq:
            return True
    return False


def detect_type(tokens: List[str]) -> MerchantType:
    if not tokens:
        return MerchantType.OTHER

    for rule in TYPE_RULES:
        if any(_has_sequence(tokens, seq) for seq in rule.detect_sequences):
            return rule.name
        if any(tok in tokens for tok in rule.detect_tokens_any):
            return rule.name
    return MerchantType.OTHER


def _strip_type_prefix(tokens: List[str], mtype: MerchantType) -> List[str]:
    t = tokens[:]

    def strip_sequence(current: List[str], seq: List[str]) -> List[str]:
        if _has_sequence(current, seq):
            n, m = len(current), len(seq)
            for i in range(n - m + 1):
                if current[i:i + m] == seq:
                    return current[i + m:]
        return current

    for seq in DOMAIN_PREFIX_SEQUENCES:
        t = strip_sequence(t, seq)

    rule = TYPE_RULE_MAP.get(mtype)
    if not rule:
        return t

    for seq in rule.strip_prefix_sequences:
        t = strip_sequence(t, seq)

    while t and t[0] in set(rule.strip_prefix_tokens_any):
        t = t[1:]

    for tok in rule.strip_after_tokens_any:
        if tok in t:
            idx = t.index(tok)
            t = t[idx + 1:]
            break

    return t


def extract_suffix(tokens: List[str]) -> List[str]:
    suffix = []
    for tok in reversed(tokens):
        if tok.isdigit():
            suffix.append(tok)
        elif re.match(r"^T\d+$", tok):
            suffix.append(tok)
        elif tok in SUFFIX_CANDIDATES:
            suffix.append(tok)
        else:
            break
    return list(reversed(suffix))


def extract_location_key(suffix_tokens: List[str]) -> str:
    loc_tokens = [t for t in suffix_tokens if not t.isdigit()]
    if not loc_tokens:
        return ""
    if "GO" in loc_tokens and "VAP" in loc_tokens:
        return "GOVAP"
    if "GV" in loc_tokens:
        return "GOVAP"
    return "".join(loc_tokens)


def extract_core(tokens: List[str], mtype: MerchantType) -> str:
    if not tokens:
        return ""
    t = _strip_type_prefix(tokens, mtype)
    filtered = [tok for tok in t if tok not in GENERIC_TOKENS]
    return filtered[0] if filtered else ""


def parse_merchant(name: str) -> ParsedMerchant:
    normalized = normalize_name(name)
    tokens = tokenize(normalized)
    mtype = detect_type(tokens)
    core = extract_core(tokens, mtype)
    suffix_tokens = extract_suffix(tokens)
    loc_key = extract_location_key(suffix_tokens)
    return ParsedMerchant(
        raw_name=name,
        normalized=normalized,
        tokens=tokens,
        mtype=mtype,
        core=core,
        suffix_tokens=suffix_tokens,
        location_key=loc_key,
    )


def get_similarity_tokens(parsed: ParsedMerchant) -> List[str]:
    tokens = parsed.tokens
    stripped = _strip_type_prefix(tokens, parsed.mtype)
    suffix_set = set(parsed.suffix_tokens)
    kept = [
        tok for tok in stripped
        if tok not in GENERIC_TOKENS and tok not in suffix_set
    ]
    return kept


def build_block_key(parsed: ParsedMerchant) -> str:
    t = parsed.mtype.value
    c = parsed.core or ""
    return f"{t}|{c}"


def prepare_blocking_dataframe(df: pd.DataFrame, col_name: str, source_label: str) -> pd.DataFrame:
    records = []
    for idx, raw in df[col_name].items():
        parsed = parse_merchant(raw)
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
    """Build blocking frame using Polars; keeps suffixes aligned with pandas output."""
    values = df[col_name].to_list()
    records = []
    for idx, raw in enumerate(values):
        parsed = parse_merchant(raw)
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
    """Join on block_key and filter by location using Polars expressions."""
    candidates = df_block_1.join(df_block_2, on="block_key", how="inner")
    return candidates.filter(
        (pl.col("location_key_1") == "")
        | (pl.col("location_key_2") == "")
        | (pl.col("location_key_1") == pl.col("location_key_2"))
    )


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
    # rapidfuzz returns 0-100 normalized score; scale to 0-1
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


def core_pipeline_pandas(
    input_path: str,
    col_name_1: str,
    col_name_2: str,
    output_path: str,
    high_thr: float,
    low_thr: float,
    alpha: float,
) -> dict:
    df = pd.read_csv(input_path)
    df_b1 = prepare_blocking_dataframe(df, col_name_1, "col1")
    df_b2 = prepare_blocking_dataframe(df, col_name_2, "col2")
    num_b1, num_b2 = len(df_b1), len(df_b2)

    candidates = build_candidate_pairs(df_b1, df_b2)
    candidates = compute_similarity_df(candidates, alpha=alpha)
    candidates = classify_matches(candidates, high_thr=high_thr, low_thr=low_thr)

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


def run_matching(
    input_path: str,
    col_name_1: str = "Merchant_Name_1",
    col_name_2: str = "Merchant_Name_2",
    output_path: str = "merchant_matching_results_fast.csv",
    high_thr: float = 0.75,
    low_thr: float = 0.4,
    alpha: float = 0.5,
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fast merchant name blocking + matching.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--col1", default="Merchant_Name_1", help="Column name for first merchant list.")
    parser.add_argument("--col2", default="Merchant_Name_2", help="Column name for second merchant list.")
    parser.add_argument("--output", default="merchant_matching_results_fast.csv", help="Output CSV path.")
    parser.add_argument("--high_thr", type=float, default=0.75, help="High threshold for MATCH.")
    parser.add_argument("--low_thr", type=float, default=0.4, help="Low threshold for REVIEW.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for Jaccard vs Levenshtein.")
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
