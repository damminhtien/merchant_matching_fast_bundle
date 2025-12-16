import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domain import MerchantType
from parsing import (
    _has_sequence,
    _strip_type_prefix,
    detect_type,
    extract_location_key,
    extract_suffix,
    get_similarity_tokens,
    normalize_name,
    parse_merchant,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        (" co.op bank  ", "COOP BANK"),
        ("abc-123", "ABC 123"),
        (None, ""),
    ],
)
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


@pytest.mark.parametrize(
    "tokens, seq, expected",
    [
        (["A", "B", "C"], ["B", "C"], True),
        (["A", "B", "C"], ["C", "D"], False),
        ([], ["A"], False),
    ],
)
def test_has_sequence(tokens, seq, expected):
    assert _has_sequence(tokens, seq) is expected


@pytest.mark.parametrize(
    "tokens, expected_type",
    [
        (["HKD", "DUY", "05"], MerchantType.HOUSEHOLD_HKD),
        (["NHA", "THUOC", "ABC"], MerchantType.PHARMACY),
        (["QUAN", "AN", "PHO"], MerchantType.RESTAURANT_QUAN),
        (["CAFE", "HOA"], MerchantType.CAFE),
        (["SHOP", "HUONG"], MerchantType.SHOP),
        (["VP", "NAM"], MerchantType.OFFICE_VP),
        (["CTY", "ABC"], MerchantType.COMPANY_CT),
        (["RANDOM"], MerchantType.OTHER),
    ],
)
def test_detect_type(tokens, expected_type):
    assert detect_type(tokens) == expected_type


@pytest.mark.parametrize(
    "tokens, mtype, expected",
    [
        (["HKD", "DUY", "05"], MerchantType.HOUSEHOLD_HKD, ["DUY", "05"]),
        (["TAP", "HOA", "HUONG"], MerchantType.SHOP, ["HUONG"]),
        (["VAN", "TAI", "NAM", "BTL"], MerchantType.OTHER, ["NAM", "BTL"]),
        (["CAFE", "HOA"], MerchantType.CAFE, ["HOA"]),
    ],
)
def test_strip_type_prefix(tokens, mtype, expected):
    assert _strip_type_prefix(tokens, mtype) == expected


@pytest.mark.parametrize(
    "tokens, expected_suffix, expected_location",
    [
        (["HKD", "DUY", "05"], ["05"], ""),
        (["TAP", "HOA", "HUONG", "GO", "VAP"], ["GO", "VAP"], "GOVAP"),
        (["SHOP", "HUONG", "GO", "VAP", "Q1"], ["GO", "VAP", "Q1"], "GOVAP"),
        (["VAN", "TAI", "NAM", "OCP"], ["OCP"], "OCP"),
        (["AN"], [], ""),
        (["12345"], ["12345"], ""),
    ],
)
def test_extract_suffix_and_location(tokens, expected_suffix, expected_location):
    suffix = extract_suffix(tokens)
    assert suffix == expected_suffix
    assert extract_location_key(suffix) == expected_location


@pytest.mark.parametrize(
    "name, expected_mtype, expected_core, expected_suffix, expected_loc, expected_sim_tokens",
    [
        ("HKD DUY 05", MerchantType.HOUSEHOLD_HKD, "DUY", ["05"], "", ["DUY"]),
        ("HKD DUY 02", MerchantType.HOUSEHOLD_HKD, "DUY", ["02"], "", ["DUY"]),
        ("TAP HOA HUONG GO VAP", MerchantType.SHOP, "HUONG", ["GO", "VAP"], "GOVAP", ["HUONG"]),
        ("SHOP HUONG GO VAP Q1", MerchantType.SHOP, "HUONG", ["GO", "VAP", "Q1"], "GOVAP", ["HUONG"]),
        ("VAN TAI NAM BTL", MerchantType.OTHER, "NAM", ["BTL"], "BTL", ["NAM"]),
        ("VAN TAI NAM OCP", MerchantType.OTHER, "NAM", ["OCP"], "OCP", ["NAM"]),
        ("AN", MerchantType.OTHER, "", [], "", []),
        ("12345", MerchantType.OTHER, "12345", ["12345"], "", []),
    ],
)
def test_parse_merchant_table(name, expected_mtype, expected_core, expected_suffix, expected_loc, expected_sim_tokens):
    parsed = parse_merchant(name)
    assert parsed.mtype == expected_mtype
    assert parsed.core == expected_core
    assert parsed.suffix_tokens == expected_suffix
    assert parsed.location_key == expected_loc
    assert get_similarity_tokens(parsed) == expected_sim_tokens
