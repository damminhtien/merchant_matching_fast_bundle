from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, List

import numpy as np
import pandas as pd


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

    if "HKD" in tokens or _has_sequence(tokens, ["HO", "KINH", "DOANH"]):
        return MerchantType.HOUSEHOLD_HKD

    if _has_sequence(tokens, ["NHA", "THUOC"]):
        return MerchantType.PHARMACY

    if _has_sequence(tokens, ["QUAN", "AN"]) or _has_sequence(tokens, ["NHA", "HANG"]):
        return MerchantType.RESTAURANT_QUAN

    if ("SALON" in tokens and "TOC" in tokens) or _has_sequence(tokens, ["TIEM", "TOC"]):
        return MerchantType.HAIR_SALON

    if "GAS" in tokens:
        return MerchantType.GAS

    if "CAFE" in tokens or "COFFEE" in tokens:
        return MerchantType.CAFE

    if _has_sequence(tokens, ["TAP", "HOA"]):
        return MerchantType.SHOP

    if (("CUA" in tokens and "HANG" in tokens)
        or "SHOP" in tokens
        or "STORE" in tokens
        or "MART" in tokens):
        return MerchantType.SHOP

    if "VP" in tokens or _has_sequence(tokens, ["VAN", "PHONG"]):
        return MerchantType.OFFICE_VP

    if ("CT" in tokens
        or "CTY" in tokens
        or ("CONG" in tokens and "TY" in tokens)
        or "TNHH" in tokens):
        return MerchantType.COMPANY_CT

    return MerchantType.OTHER


def _strip_type_prefix(tokens: List[str], mtype: MerchantType) -> List[str]:
    t = tokens[:]

    def strip_sequence(seq: List[str]) -> List[str]:
        nonlocal t
        if _has_sequence(t, seq):
            n, m = len(t), len(seq)
            for i in range(n - m + 1):
                if t[i:i + m] == seq:
                    t = t[i + m:]
                    break
        return t

    # Domain prefixes
    if _has_sequence(t, ["VAN", "TAI"]):
        t = strip_sequence(["VAN", "TAI"])
    if _has_sequence(t, ["TAP", "HOA"]):
        t = strip_sequence(["TAP", "HOA"])

    if mtype == MerchantType.HOUSEHOLD_HKD:
        if "HKD" in t:
            idx = t.index("HKD")
            return t[idx + 1:]
        return strip_sequence(["HO", "KINH", "DOANH"])

    if mtype == MerchantType.PHARMACY:
        return strip_sequence(["NHA", "THUOC"])

    if mtype == MerchantType.RESTAURANT_QUAN:
        if _has_sequence(t, ["QUAN", "AN"]):
            return strip_sequence(["QUAN", "AN"])
        return strip_sequence(["NHA", "HANG"])

    if mtype == MerchantType.HAIR_SALON:
        if _has_sequence(t, ["SALON", "TOC"]):
            return strip_sequence(["SALON", "TOC"])
        return strip_sequence(["TIEM", "TOC"])

    if mtype == MerchantType.GAS:
        if t and t[0] == "GAS":
            return t[1:]
        return t

    if mtype == MerchantType.CAFE:
        if "CAFE" in t:
            idx = t.index("CAFE")
            return t[idx + 1:]
        if "COFFEE" in t:
            idx = t.index("COFFEE")
            return t[idx + 1:]
        return t

    if mtype == MerchantType.SHOP:
        if _has_sequence(t, ["CUA", "HANG"]):
            return strip_sequence(["CUA", "HANG"])
        if t and t[0] == "CH":
            return t[1:]
        if t and t[0] == "TIEM":
            return t[1:]
        return t

    if mtype == MerchantType.OFFICE_VP:
        if t and t[0] == "VP":
            return t[1:]
        if _has_sequence(t, ["VAN", "PHONG"]):
            return strip_sequence(["VAN", "PHONG"])
        return t

    if mtype == MerchantType.COMPANY_CT:
        i = 0
        while i < len(t) and t[i] in {"CT", "CTY", "CONG", "TY", "TNHH"}:
            i += 1
        return t[i:]

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


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            x = prev[j] + 1
            y = curr[j - 1] + 1
            z = prev[j - 1] + cost
            if y < x:
                x = y
            if z < x:
                x = z
            curr[j] = x
        prev, curr = curr, prev
    return prev[lb]


def levenshtein_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    dist = levenshtein_distance(a, b)
    max_len = max(len(a), len(b))
    return 1.0 - dist / max_len


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
    try:
        from rapidfuzz.distance import Levenshtein as rf_lev
    except ImportError as exc:
        raise ImportError(
            "Polars engine needs rapidfuzz for Levenshtein. Please install rapidfuzz."
        ) from exc

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

        num_match = (candidates_pl["match_label"] == "MATCH").sum()
        num_review = (candidates_pl["match_label"] == "REVIEW").sum()
        num_non_match = (candidates_pl["match_label"] == "NON_MATCH").sum()
        num_candidates = len(candidates_pl)
        num_block_keys = candidates_pl.select(pl.col("block_key").n_unique()).item()
    else:
        df = pd.read_csv(input_path)
        df_b1 = prepare_blocking_dataframe(df, col_name_1, "col1")
        df_b2 = prepare_blocking_dataframe(df, col_name_2, "col2")
        num_b1, num_b2 = len(df_b1), len(df_b2)
        candidates = build_candidate_pairs(df_b1, df_b2)

        candidates = compute_similarity_df(candidates, alpha=alpha)
        candidates = classify_matches(candidates, high_thr=high_thr, low_thr=low_thr)

        candidates.sort_values("sim_final", ascending=False, inplace=True)
        candidates.to_csv(output_path, index=False)

        num_match = (candidates["match_label"] == "MATCH").sum()
        num_review = (candidates["match_label"] == "REVIEW").sum()
        num_non_match = (candidates["match_label"] == "NON_MATCH").sum()
        num_candidates = len(candidates)
        num_block_keys = candidates["block_key"].nunique()

    print("Done.")
    print(f"Records col1       : {num_b1}")
    print(f"Records col2       : {num_b2}")
    print(f"Candidate pairs    : {num_candidates}")
    print(f"Block keys         : {num_block_keys}")
    print(f"MATCH              : {num_match}")
    print(f"REVIEW             : {num_review}")
    print(f"NON_MATCH          : {num_non_match}")
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
