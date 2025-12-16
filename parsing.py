from __future__ import annotations

import re
from typing import List, Optional

from domain import (
    DOMAIN_PREFIX_SEQUENCES,
    GENERIC_TOKENS,
    SUFFIX_CANDIDATES,
    TYPE_RULE_MAP,
    TYPE_RULES,
    MerchantType,
    ParsedMerchant,
)


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


def extract_suffix(tokens: List[str], suffix_candidates: Optional[set[str]] = None) -> List[str]:
    suffix_candidates = suffix_candidates or SUFFIX_CANDIDATES
    suffix = []
    for tok in reversed(tokens):
        if tok.isdigit():
            suffix.append(tok)
        elif re.match(r"^T\d+$", tok):
            suffix.append(tok)
        elif tok in suffix_candidates:
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


def parse_merchant(name: str, suffix_candidates: Optional[set[str]] = None) -> ParsedMerchant:
    normalized = normalize_name(name)
    tokens = tokenize(normalized)
    mtype = detect_type(tokens)
    core = extract_core(tokens, mtype)
    suffix_tokens = extract_suffix(tokens, suffix_candidates or SUFFIX_CANDIDATES)
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
