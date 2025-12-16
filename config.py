from __future__ import annotations

DEFAULT_CONFIG = {
    "version": "0.1.0",
    "generic_tokens": {
        "CH", "CUA", "HANG", "TIEM",
        "SHOP", "STORE", "MART", "POS",
        "QUAN", "AN",
    },
    "suffix_candidates": {
        "BTL", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9",
        "Q10", "Q11", "Q12", "GO", "VAP", "GV", "OCP",
    },
    "domain_prefix_sequences": [
        ["VAN", "TAI"],
        ["TAP", "HOA"],
    ],
    "thresholds": {
        "high_thr": 0.75,
        "low_thr": 0.4,
        "alpha": 0.5,
    },
}

VERSION = DEFAULT_CONFIG["version"]
