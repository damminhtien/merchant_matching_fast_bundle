# Fast Merchant Name Blocking + Matching

**Goal**: Optimize speed and scalability for matching merchant names between two columns.

## Core idea

1. **Per-record parsing (O(N))**
   - Normalize: upper-case, drop special characters, CO.OP -> COOP.
   - Detect merchant type (MerchantType):
     - COMPANY_CT, HOUSEHOLD_HKD, PHARMACY, GAS, SHOP, CAFE,
       RESTAURANT_QUAN, HAIR_SALON, OFFICE_VP, OTHER.
   - Remove domain prefixes: `VAN TAI`, `TAP HOA`.
   - Remove type prefixes: `HKD`, `HO KINH DOANH`, `NHA THUOC`, `QUAN AN`, `NHA HANG`, `SALON TOC`, `TIEM TOC`, `CUA HANG`, `CH`, `TIEM`, …
   - Split into:
     - `core` (main brand),
     - `suffix_tokens` (branch/location codes: 03, BTL, Q1, OCP, GO VAP, …),
     - `location_key` (standardized location code: BTL, Q1, GOVAP, OCP, …).
   - Generate:
     - `block_key = (merchant_type, core)`,
     - `sim_tokens` for similarity (after removing prefixes, suffixes, generic tokens).

2. **Blocking (shrink the search space)**
   - Join the two tables on `block_key` (pandas merge; Spark join later if needed).
   - Secondary blocking with `location_key` (vectorized):
     - If both sides have non-empty, different `location_key` → drop the pair.
     - If either side lacks a location → keep it (avoid recall loss).

   Complexity after blocking ~ \(\sum_b |B_{1,b}| \cdot |B_{2,b}|\), typically near linear when blocks are small.

3. **Similarity + classification**
   - Token-level: Jaccard on `sim_tokens`.
   - String-level: Levenshtein similarity on `" ".join(sim_tokens)`.
   - Combine:
     - `sim_final = alpha * sim_jaccard + (1 - alpha) * sim_levenshtein`.
   - Classification (vectorized, no apply):
     - `sim_final < low_thr` → NON_MATCH.
     - `low_thr <= sim_final < high_thr` → REVIEW.
     - `sim_final >= high_thr` and **no** differing numeric suffix → MATCH.
     - `sim_final >= high_thr` but differing numeric suffix (branch) → REVIEW (avoid merging different branches).

## CLI usage

```bash
pip install -r requirements.txt

python merchant_matching_fast.py \
    --input merchant_names_randomized.csv \
    --col1 Merchant_Name_1 \
    --col2 Merchant_Name_2 \
    --output merchant_matching_results_fast.csv \
    --high_thr 0.75 \
    --low_thr 0.4 \
    --alpha 0.5 \
    --engine pandas        # or polars (runs similarity/classification in Polars, no Python loops)
```

- `alpha`: weight between Jaccard and Levenshtein.
- `high_thr`, `low_thr`: thresholds for MATCH/REVIEW/NON_MATCH.
- `engine`: dataframe engine for the blocking/join step (Polars path also runs similarity + classification with Polars set ops and RapidFuzz Levenshtein to avoid Python row loops).

## Performance optimizations

- **Blocking & location filter**:
  - Merge on `block_key` using pandas/SQL/Spark join.
  - Fully vectorized location filter:
    - `mask = (loc1 == "") | (loc2 == "") | (loc1 == loc2)`.

- **Classification**:
  - Avoid `DataFrame.apply(axis=1)`.
  - Use NumPy vectorization:
    - `labels = np.full(...)` then assign via masks.
    - Extract numeric suffix with `Series.str.extract` (regex precompiled in pandas).

- **Levenshtein**:
  - Hand-implemented DP with two rows, applied only to candidates after blocking.
  - For large datasets, candidate count remains far below N1×N2.

## Scaling to big data

- Port `parse_merchant`, `build_block_key`, `extract_location_key` to Spark UDFs.
- Generate `block_key` + `location_key` in the two distributed tables, then join on:
  - `block_key` (can hash to int),
  - Filter by `location_key` in WHERE.
- The `compute_similarity_df`/`classify_matches` stage can run:
  - On Spark (UDF + withColumn),
  - Or in a separate service after blocking has drastically reduced candidates.

File `merchant_matching_results_fast.csv` in the zip is a sample output produced from `merchant_names_randomized.csv` for quick verification.
