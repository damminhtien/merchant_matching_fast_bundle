from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


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

DOMAIN_PREFIX_SEQUENCES = [
    ["VAN", "TAI"],
    ["TAP", "HOA"],
]


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
