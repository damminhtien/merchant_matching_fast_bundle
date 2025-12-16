# Fast Merchant Name Blocking + Matching

**Goal**: Fast, scalable merchant-name matching with configurable domain rules and pluggable engines (pandas / Polars).

## Quickstart

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
  --engine pandas   # or polars
```

- `alpha`: weight between Jaccard and Levenshtein.
- `high_thr`, `low_thr`: thresholds for MATCH/REVIEW/NON_MATCH.
- `engine`: dataframe engine; Polars path also computes similarity/classification inside Polars with RapidFuzz Levenshtein to avoid Python row loops.
- Default thresholds come from `config.py` (`high_thr`, `low_thr`, `alpha`); override via CLI flags.

## How it works

1. **Parsing (O(N))**  
   Normalize text, detect merchant type via configurable rules, strip prefixes, extract `core`, `suffix_tokens`, `location_key`, and `sim_tokens`.
2. **Blocking**  
   Block on `(merchant_type, core)` then filter by `location_key` (keep when missing to protect recall).
3. **Similarity + classification**  
   - Jaccard over `sim_tokens`.  
   - Levenshtein via RapidFuzz on `" ".join(sim_tokens)`.  
   - `sim_final = alpha * sim_jaccard + (1 - alpha) * sim_levenshtein`.  
   - Labels: `MATCH` / `REVIEW` / `NON_MATCH` with a guard against mismatched numeric suffix (branch codes).

## Project structure

- `domain.py`: `MerchantType`, constants, and the `TypeRule` config table (add a rule to extend).
- `config.py`: default tokens/prefixes and similarity thresholds + version metadata.
- `parsing.py`: normalization, tokenization, type detection via rules, prefix stripping, core/suffix/location extraction.
- `blocking.py`: block-key generation and pandas/Polars blocking + candidate building.
- `similarity.py`: Jaccard + RapidFuzz Levenshtein, pandas similarity/classification, Polars similarity/classification.
- `cli.py`: engine-specific pipelines and `run_matching` dispatcher.
- `merchant_matching_fast.py`: thin entrypoint importing `run_matching`.

## Performance notes

- Polars engine removes Python loops for blocking and similarity/classification; pandas keeps a Python loop for similarity but uses RapidFuzz (C-level) for Levenshtein.
- Avoid `DataFrame.apply`; classification is vectorized (pandas) or expression-based (Polars).
- Add more merchant types by appending to `TYPE_RULES` in `domain.py`.
- Output CSV now includes metadata columns: `engine`, `alpha`, `high_thr`, `low_thr`, `timestamp`, `version`.

### Max-speed tips

- Use `--engine polars` to keep blocking and similarity/classification out of Python loops.
- Keep `sim_tokens` small: add generic tokens to `config.py` to strip noisy words early.
- For very large inputs on pandas, parallelize `compute_similarity_df` with `joblib`/`multiprocessing` (row-level work is pure Python but Levenshtein runs in C via RapidFuzz).
- For distributed scale, port the parsing/blocking logic as Spark UDFs and keep the same block_key + location filter strategy before similarity.
- Profile blocking cardinality: overly broad blocks explode candidate pairs; adjust rules or core extraction to keep blocks tight.

## Testing

```bash
pytest
```

Unit tests focus on parsing primitives (normalization, type detection, prefix stripping, suffix/location extraction, similarity tokens) with table-driven cases.
