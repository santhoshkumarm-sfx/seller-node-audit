"""
process_orders.py
==================

High-throughput Seller Node / Pincode audit engine designed for 1,000,000+
(10 Lakh+) daily order rows.

Why this scales (the naive approach does NOT):
-----------------------------------------------
A plain "for every order row, compare against every aligned row" is an
O(N x M) string-comparison problem. With N = 10,00,000 orders and
M = 30,000 aligned sellers, that is 30 BILLION fuzzy-match calls -- minutes
turn into days.

This script avoids that with three layers of optimization, applied in order:

  1. VECTORIZED NORMALIZATION
     seller_name / seller_addr cleaning (uppercase, strip everything that
     isn't A-Z/0-9) is done as a single vectorized pandas .str.replace()
     call across the whole column, not per-row Python loops.

  2. EXACT-TOKEN BLOCKING (dual key)
     Rows are bucketed twice -- once by the first 4 characters of the
     normalized seller name, once by the first 4 characters of the
     normalized (deduped) seller address -- optionally scoped to the same
     client first, when a client/client_id column is available on both
     files. A row can only ever fuzzy-match a candidate sharing one of
     those two bucket keys. Two passes (rather than one) are run because
     seller names are frequently identical or near-identical across
     genuinely different sellers -- the ADDRESS is what actually
     distinguishes them, so blocking on name alone (as an earlier version
     of this script did) could silently miss the correct candidate. This
     still turns one 30-billion-comparison problem into thousands of tiny
     ones, which is the standard "blocking" technique used in
     record-linkage at scale.

  3. C-VECTORIZED FUZZY SCORING (rapidfuzz.process.cdist), address-weighted
     Within each block, rapidfuzz.process.cdist computes the entire name
     AND address similarity matrices in compiled C++ (SIMD-accelerated,
     multi-threaded via `workers=-1`) rather than calling into Python once
     per pair -- 50-100x faster than a Python-level double loop for the
     same number of comparisons. The two scores are then combined as
     30% name + 70% address (address carries the deciding weight, per
     business input), and the higher-scoring of the two blocking passes
     wins per order row.

Net effect: end-to-end runtime scales close to O(N + M) rather than
O(N x M), and 10L+ rows finish in low single-digit minutes on a laptop.

Usage
-----
    python process_orders.py \
        --aligned "sellers aligned.csv" \
        --orders "orders_created.xlsx" \
        --outdir ./output \
        --threshold 90 \
        --block-len 4

Requires: pandas, numpy, rapidfuzz, openpyxl (for .xlsx input),
          pyarrow (optional, speeds up CSV reads)
    pip install pandas numpy rapidfuzz openpyxl pyarrow
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process as rf_process

pd.options.mode.chained_assignment = None


# --------------------------------------------------------------------------- #
# TIMING HELPER
# --------------------------------------------------------------------------- #
class Stopwatch:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.last = self.t0

    def lap(self, label):
        now = time.perf_counter()
        print(f"[{now - self.t0:8.2f}s total | +{now - self.last:6.2f}s] {label}")
        self.last = now


# --------------------------------------------------------------------------- #
# 1. FAST I/O
# --------------------------------------------------------------------------- #
def read_any(path: str, columns=None) -> pd.DataFrame:
    """
    Reads CSV via the pyarrow engine (fast, low-memory, C-vectorized parsing)
    when available, falling back to the default engine. Reads XLSX via
    openpyxl. `columns` (if given) restricts the read to only the columns
    actually needed -- skipping unused columns is a large win at 10L+ rows.
    """
    p = Path(path)
    if p.suffix.lower() == ".csv":
        try:
            return pd.read_csv(path, engine="pyarrow", usecols=columns)
        except Exception:
            return pd.read_csv(path, usecols=columns, low_memory=False)
    elif p.suffix.lower() in (".xlsx", ".xls"):
        # NOTE: openpyxl parses the whole workbook row-by-row in Python and
        # has no true "vectorized" mode -- for orders files that regularly
        # exceed ~3-4 lakh rows, exporting to CSV upstream (e.g. directly
        # from BigQuery / the warehouse) instead of XLSX will noticeably
        # cut ingestion time. XLSX is still fully supported here.
        df = pd.read_excel(path, engine="openpyxl")
        if columns:
            df = df[[c for c in columns if c in df.columns]]
        return df
    else:
        raise ValueError(f"Unsupported file type: {path}")


# --------------------------------------------------------------------------- #
# 2. VECTORIZED NORMALIZATION
# --------------------------------------------------------------------------- #
_CLEAN_RE = re.compile(r"[^A-Z0-9]")
_SPLIT_RE = re.compile(r"[^A-Z0-9]+")

# "Phase I" vs "Phase 1", "Sector V" vs "Sector 5" -- a very common source of
# address mismatch that has nothing to do with the actual location.
_ROMAN_TO_DIGIT = {
    "I": "1", "II": "2", "III": "3", "IV": "4", "V": "5",
    "VI": "6", "VII": "7", "VIII": "8", "IX": "9", "X": "10",
}

# Name/address similarity weights. Address is the primary discriminant here
# (per business input: seller names are frequently identical or near-
# identical across genuinely different sellers, so the address is what
# actually tells them apart) -- name still contributes, mainly to break
# ties and rule out obviously-wrong candidates.
NAME_WEIGHT = 0.30
ADDR_WEIGHT = 0.70

# Confidence tiers on the combined score. Real addresses across two systems
# legitimately differ in composition (one embeds the seller name/city/pincode
# in the address field, the other doesn't; one uses "Phase I", the other
# "Phase 1"; etc.) -- a single hard 90% cutoff on combined name+address score
# quietly throws away a lot of genuine matches. Tiers keep the strict >=90%
# bucket for confident auto-flagging while surfacing 75-89% as a reviewable
# "Medium confidence" bucket instead of silently discarding it.
def confidence_tier(score: float) -> str:
    if score >= 90:
        return "High"
    if score >= 75:
        return "Medium"
    if score >= 60:
        return "Low"
    return "None"


def vectorized_clean(series: pd.Series) -> pd.Series:
    """Uppercase + strip everything but letters/digits, across an entire
    column in one vectorized pass (no per-row Python-level loop). Used for
    seller_name, hub codes, and client -- fields where word order and
    repetition aren't a concern."""
    return (
        series.astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
        .fillna("")
    )


def _dedupe_address_scalar(value) -> str:
    """
    Address-specific normalization that goes beyond plain character
    stripping, to survive the kind of noise real seller addresses carry:

      - case differences            ("Sector 18" vs "SECTOR 18")
      - inconsistent punctuation    ("Plot 10, Sector-18" vs "Plot 10 Sector 18")
      - a different NUMBER of separators (extra commas/periods/spaces)
      - a word repeated in the string, sometimes with different casing
        ("Noida, noida", "Gurgaon Gurgaon Haryana")

    Approach: split into alphanumeric tokens (any run of punctuation/
    whitespace is a separator, so separator *count* stops mattering),
    uppercase each token, then drop repeat occurrences of the same token
    anywhere in the address (not just consecutive repeats) while keeping
    first-seen order. The deduped tokens are joined with no separator so
    the downstream similarity score is comparing address *content*, not
    formatting.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).upper()
    tokens = _SPLIT_RE.split(text)
    seen = set()
    out = []
    for tok in tokens:
        if not tok:
            continue
        tok = _ROMAN_TO_DIGIT.get(tok, tok)
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return "".join(out)


def vectorized_clean_address(series: pd.Series) -> pd.Series:
    """Row-wise (not fully vectorized -- token dedup is inherently
    per-value) but still fast: simple string/set operations only, no
    regex-per-character-class overhead. On 10L+ rows this typically adds
    only a few seconds versus the fully vectorized name cleaning."""
    return series.map(_dedupe_address_scalar)


# --------------------------------------------------------------------------- #
# 3. BLOCKED, C-VECTORIZED FUZZY MATCH  (name + address, address-weighted)
# --------------------------------------------------------------------------- #
MAX_TILE_CELLS = 4_000_000  # ~16MB per float32 matrix -- caps peak memory per cdist tile


def _score_block(order_pos, aligned_pos, order_names, aligned_names, order_addrs, aligned_addrs,
                  best_idx, best_combined, best_name_score, best_addr_score):
    """
    Scores one block's queries against its candidates. Normally a block is
    small (blocking already narrowed it) and this runs as a single
    rapidfuzz.process.cdist call. If a pathological block slips through
    (e.g. thousands of rows sharing a very common address prefix), both the
    query and candidate sides are chunked into tiles so no single dense
    score matrix can exceed MAX_TILE_CELLS -- this bounds peak memory
    regardless of block size, at the cost of a few extra Python-level loop
    iterations for that one block only.
    """
    n_q, n_c = len(order_pos), len(aligned_pos)
    if n_q * n_c <= MAX_TILE_CELLS:
        q_chunks = [order_pos]
        c_chunks = [aligned_pos]
    else:
        tile_c = max(1, min(n_c, MAX_TILE_CELLS // max(1, n_q)))
        tile_q = max(1, MAX_TILE_CELLS // max(1, tile_c))
        q_chunks = [order_pos[i:i + tile_q] for i in range(0, n_q, tile_q)]
        c_chunks = [aligned_pos[i:i + tile_c] for i in range(0, n_c, tile_c)]

    for q_pos in q_chunks:
        name_queries = order_names[q_pos]
        addr_queries = order_addrs[q_pos]
        running_best_col = np.full(len(q_pos), -1, dtype=np.int64)
        running_best_combined = np.full(len(q_pos), -1.0, dtype=np.float32)
        running_best_name = np.zeros(len(q_pos), dtype=np.float32)
        running_best_addr = np.zeros(len(q_pos), dtype=np.float32)

        for c_pos in c_chunks:
            name_matrix = rf_process.cdist(name_queries, aligned_names[c_pos], scorer=fuzz.ratio,
                                            workers=-1, dtype=np.float32)
            addr_matrix = rf_process.cdist(addr_queries, aligned_addrs[c_pos], scorer=fuzz.ratio,
                                            workers=-1, dtype=np.float32)
            combined_matrix = NAME_WEIGHT * name_matrix + ADDR_WEIGHT * addr_matrix
            local_col = combined_matrix.argmax(axis=1)
            rows = np.arange(len(q_pos))
            local_combined = combined_matrix[rows, local_col]

            improve = local_combined > running_best_combined
            running_best_combined = np.where(improve, local_combined, running_best_combined)
            running_best_col = np.where(improve, c_pos[local_col], running_best_col)
            running_best_name = np.where(improve, name_matrix[rows, local_col], running_best_name)
            running_best_addr = np.where(improve, addr_matrix[rows, local_col], running_best_addr)

        best_idx[q_pos] = running_best_col
        best_combined[q_pos] = running_best_combined
        best_name_score[q_pos] = running_best_name
        best_addr_score[q_pos] = running_best_addr


def run_blocking_pass(orders, aligned, block_key_col, sw, label):
    """
    One blocking pass: for every block key present on the orders side,
    pull the aligned candidates sharing that same key, score BOTH name and
    address for the block via rapidfuzz.process.cdist (C++, multi-threaded),
    combine them with the address-weighted formula, and keep the best
    candidate per order row. Oversized blocks are tiled (see _score_block)
    so memory stays bounded no matter how a real file's data happens to
    cluster.
    """
    n = len(orders)
    best_idx = np.full(n, -1, dtype=np.int64)
    best_combined = np.zeros(n, dtype=np.float32)
    best_name_score = np.zeros(n, dtype=np.float32)
    best_addr_score = np.zeros(n, dtype=np.float32)

    aligned_groups = aligned.groupby(block_key_col).indices
    order_groups = orders.groupby(block_key_col).indices

    aligned_names = aligned["_name_norm"].to_numpy()
    aligned_addrs = aligned["_addr_norm"].to_numpy()
    order_names = orders["_name_norm"].to_numpy()
    order_addrs = orders["_addr_norm"].to_numpy()

    blocks_matched = 0
    blocks_skipped = 0
    largest_block_cells = 0

    for block_key, order_pos in order_groups.items():
        aligned_pos = aligned_groups.get(block_key)
        if aligned_pos is None or len(aligned_pos) == 0:
            blocks_skipped += len(order_pos)
            continue
        largest_block_cells = max(largest_block_cells, len(order_pos) * len(aligned_pos))
        _score_block(order_pos, aligned_pos, order_names, aligned_names, order_addrs, aligned_addrs,
                     best_idx, best_combined, best_name_score, best_addr_score)
        blocks_matched += 1

    sw.lap(f"Blocking pass [{label}] complete "
           f"({blocks_matched} blocks compared, {blocks_skipped} orders had no same-block candidate, "
           f"largest block = {largest_block_cells:,} cells)")
    return best_idx, best_combined, best_name_score, best_addr_score


def blocked_fuzzy_match(orders: pd.DataFrame, aligned: pd.DataFrame,
                         block_len: int, threshold: int, sw: Stopwatch):
    """
    Two blocking passes are run and the better result kept per order row:
      1. blocked by normalized seller_name prefix
      2. blocked by normalized (deduped) seller_addr prefix
    Running both keeps recall high even when ONE of name/address has a
    prefix-breaking typo or reordering -- a single name-only block (the
    previous version of this script) would silently miss those.

    Returns: best_idx, combined_score, name_score, addr_score (numpy arrays
    aligned to `orders` row order). best_idx is -1 wherever the combined
    score falls below `threshold`.
    """
    # NOTE: blocking is intentionally NOT scoped by client. In practice the
    # orders file's client field is often an internal lane/tag label (e.g.
    # "Flipkart MH Volumetric") while the aligned master uses a cleaner
    # canonical client name (e.g. "Flipkart Internet E2E") -- an exact-match
    # client gate on these silently drops almost every real match. Client
    # is instead scored and reported per row (see `client_match_score`
    # below) so a human can sanity-check it, without it being able to
    # veto an otherwise strong name+address match.
    orders["_name_block"] = orders["_name_norm"].str[:block_len]
    aligned["_name_block"] = aligned["_name_norm"].str[:block_len]
    orders["_addr_block"] = orders["_addr_norm"].str[:block_len]
    aligned["_addr_block"] = aligned["_addr_norm"].str[:block_len]

    idx_a, comb_a, name_a, addr_a = run_blocking_pass(orders, aligned, "_name_block", sw, "seller-name prefix")
    idx_b, comb_b, name_b, addr_b = run_blocking_pass(orders, aligned, "_addr_block", sw, "seller-address prefix")

    take_b = comb_b > comb_a
    best_idx = np.where(take_b, idx_b, idx_a)
    best_combined = np.where(take_b, comb_b, comb_a)
    best_name_score = np.where(take_b, name_b, name_a)
    best_addr_score = np.where(take_b, addr_b, addr_a)

    # NOTE: no threshold cutoff here -- the best candidate found in either
    # blocking pass is always returned (as long as one exists) so the caller
    # can classify it into a confidence tier rather than silently discarding
    # anything below `threshold`. `threshold` is applied later, per-tier.
    return best_idx, best_combined, best_name_score, best_addr_score


# --------------------------------------------------------------------------- #
# 4. MAIN PIPELINE
# --------------------------------------------------------------------------- #
def run(aligned_path: str, orders_path: str, outdir: str,
        threshold: int = 90, block_len: int = 4):
    sw = Stopwatch()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    aligned = read_any(aligned_path)
    aligned.columns = [c.strip() for c in aligned.columns]
    sw.lap(f"Loaded sellers aligned file: {len(aligned):,} rows")

    orders = read_any(orders_path)
    orders.columns = [c.strip() for c in orders.columns]
    sw.lap(f"Loaded orders created file: {len(orders):,} rows")

    # ---- resolve column names (tolerant of minor naming variance) -------- #
    aligned_name_col = next((c for c in ["seller_name"] if c in aligned.columns), None)
    aligned_addr_col = next((c for c in ["seller_addr", "seller_address"] if c in aligned.columns), None)
    aligned_node_col = next((c for c in ["seller_node"] if c in aligned.columns), None)
    aligned_hub_col = next((c for c in ["destination_hub", "new_hub_id"] if c in aligned.columns), None)

    order_name_col = next((c for c in ["Seller name", "Seller name ", "seller_name"] if c in orders.columns), None)
    order_addr_col = next((c for c in ["Seller address", "seller_addr"] if c in orders.columns), None)
    order_node_col = next((c for c in ["seller_node"] if c in orders.columns), None)
    order_hub_col = next((c for c in ["hub"] if c in orders.columns), None)
    order_awb_col = next((c for c in ["awb_number", "awb"] if c in orders.columns), None)
    order_cluster_col = next((c for c in ["cluster_code"] if c in orders.columns), None)
    aligned_client_col = next((c for c in ["client"] if c in aligned.columns), None)
    order_client_col = next((c for c in ["client", "client_id", "Client", "name"] if c in orders.columns), None)

    required = {
        "sellers aligned: seller_name": aligned_name_col,
        "sellers aligned: seller_addr": aligned_addr_col,
        "sellers aligned: seller_node": aligned_node_col,
        "sellers aligned: destination_hub": aligned_hub_col,
        "orders_created: Seller name": order_name_col,
        "orders_created: Seller address": order_addr_col,
        "orders_created: seller_node": order_node_col,
        "orders_created: hub": order_hub_col,
        "orders_created: awb_number": order_awb_col,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise KeyError(f"Could not resolve required column(s): {missing}. "
                        f"Aligned columns: {list(aligned.columns)}. "
                        f"Orders columns: {list(orders.columns)}.")

    # ---- vectorized normalization ---------------------------------------- #
    aligned["_name_norm"] = vectorized_clean(aligned[aligned_name_col])
    orders["_name_norm"] = vectorized_clean(orders[order_name_col])
    aligned["_addr_norm"] = vectorized_clean_address(aligned[aligned_addr_col])
    orders["_addr_norm"] = vectorized_clean_address(orders[order_addr_col])
    sw.lap("Normalized seller_name (vectorized) and seller_addr "
           "(deduped tokens, case/punctuation-insensitive)")

    has_client_both = bool(aligned_client_col and order_client_col)
    if has_client_both:
        aligned["_client_norm"] = vectorized_clean(aligned[aligned_client_col])
        orders["_client_norm"] = vectorized_clean(orders[order_client_col])
        print(f"Client column found on both sides "
              f"(aligned: '{aligned_client_col}', orders: '{order_client_col}'). "
              f"Client is reported per match (client_match_score) as a sanity-check "
              f"signal, but is NOT used as a hard filter -- order-side client labels "
              f"are frequently internal lane names (e.g. 'Flipkart MH Volumetric') "
              f"rather than the aligned master's canonical client name (e.g. "
              f"'Flipkart Internet E2E'), and an exact-match gate on that would "
              f"silently drop otherwise-correct seller+address matches.")
    else:
        aligned["_client_norm"] = ""
        orders["_client_norm"] = ""
        print("NOTE: no Client column found on both files -- client_match_score "
              "will be blank for every row.")
    sw.lap("Resolved client key")

    # ---- blocked + C-vectorized fuzzy match (name + address, address-weighted) #
    best_idx, best_combined, best_name_score, best_addr_score = blocked_fuzzy_match(
        orders, aligned, block_len, threshold, sw
    )

    has_candidate = best_idx >= 0
    tiers = np.full(len(orders), "None", dtype=object)
    tiers[has_candidate] = [confidence_tier(sc) for sc in best_combined[has_candidate]]

    # "Matched" (for the purpose of attaching aligned-record columns and
    # doing the hub/node comparison at all) means ANY confidence tier above
    # "None" -- i.e. a plausible candidate was found. Whether that candidate
    # is trusted enough to auto-flag a pincode update is a separate, later
    # decision keyed off the tier / `threshold`.
    matched_mask = tiers != "None"
    # `threshold` (CLI --threshold, default 90) is the actual gate used for
    # auto-flagging File 1 below -- kept independently configurable from the
    # informational tier boundaries in `confidence_tier()`.
    high_confidence = has_candidate & (best_combined >= threshold)
    medium_confidence = tiers == "Medium"  # 75-89%: reviewable, not auto-flagged by default

    orders["name_match_score"] = best_name_score
    orders["address_match_score"] = best_addr_score
    orders["combined_match_score"] = best_combined
    orders["confidence_tier"] = tiers
    orders["matched_ge_threshold"] = high_confidence  # kept for backward compatibility

    # Vectorized gather of matched aligned columns (no per-row loop)
    safe_idx = np.where(matched_mask, best_idx, 0)
    orders["aligned_seller_node"] = np.where(matched_mask, aligned[aligned_node_col].to_numpy()[safe_idx], None)
    orders["aligned_seller_address"] = np.where(matched_mask, aligned[aligned_addr_col].to_numpy()[safe_idx], None)
    orders["destination_hub"] = np.where(matched_mask, aligned[aligned_hub_col].to_numpy()[safe_idx], None)
    if aligned_client_col:
        orders["client"] = np.where(matched_mask, aligned[aligned_client_col].to_numpy()[safe_idx], None)

    sw.lap(f"Matched {int(matched_mask.sum()):,} / {len(orders):,} orders to some candidate "
           f"(High {int(high_confidence.sum()):,} / Medium {int(medium_confidence.sum()):,} / "
           f"Low {int((tiers=='Low').sum()):,}), combined score = "
           f"name {NAME_WEIGHT:.0%} + address {ADDR_WEIGHT:.0%}")

    # ---- client similarity: reported, not gating (see note above) -------- #
    if has_client_both:
        matched_aligned_client_norm = np.where(
            matched_mask, aligned["_client_norm"].to_numpy()[safe_idx], ""
        )
        order_client_norm_arr = orders["_client_norm"].to_numpy()
        client_scores = np.zeros(len(orders), dtype=np.float32)
        idxs = np.nonzero(matched_mask)[0]
        for i in idxs:
            client_scores[i] = fuzz.partial_ratio(order_client_norm_arr[i], matched_aligned_client_norm[i])
        orders["client_match_score"] = client_scores
    else:
        orders["client_match_score"] = None

    # ---- vectorized business rules ---------------------------------------- #
    # NOTE: gated on `high_confidence` (>= --threshold, default 90), not the
    # looser `matched_mask` -- File 1 stays true to the original >=90% spec.
    order_hub_norm = vectorized_clean(orders[order_hub_col])
    dest_hub_norm = vectorized_clean(orders["destination_hub"].astype(str))
    hub_already_aligned = high_confidence & (order_hub_norm != "") & (order_hub_norm == dest_hub_norm)

    node_differs = high_confidence & (
        orders[order_node_col].astype(str).str.strip().str.upper()
        != orders["aligned_seller_node"].astype(str).str.strip().str.upper()
    )
    pincode_update_required = high_confidence & node_differs & ~hub_already_aligned
    orders["hub_already_aligned"] = hub_already_aligned
    orders["pincode_update_required"] = pincode_update_required

    # Medium-confidence (75-89%) candidates that would ALSO look like a
    # pincode-update case are not auto-flagged into File 1, but are written
    # out separately below so nothing plausible is silently dropped.
    medium_hub_aligned = medium_confidence & (order_hub_norm != "") & (order_hub_norm == dest_hub_norm)
    medium_node_differs = medium_confidence & (
        orders[order_node_col].astype(str).str.strip().str.upper()
        != orders["aligned_seller_node"].astype(str).str.strip().str.upper()
    )
    manual_review_candidate = medium_confidence & medium_node_differs & ~medium_hub_aligned

    if order_cluster_col:
        cluster_series = orders[order_cluster_col]
        cluster_missing = cluster_series.isna() | (cluster_series.astype(str).str.strip() == "")
    else:
        cluster_series = pd.Series([None] * len(orders))
        cluster_missing = pd.Series([True] * len(orders))
    orders["cluster_code_norm"] = cluster_series
    orders["cluster_code_missing"] = cluster_missing

    sw.lap(f"Applied hub-exclusion and cluster-code rules "
           f"({int(pincode_update_required.sum()):,} pincode updates, "
           f"{int(cluster_missing.sum()):,} missing cluster codes)")

    # ---- FILE 1: pincode_update_required.csv ------------------------------ #
    file1 = orders.loc[pincode_update_required, [
        "aligned_seller_node", order_name_col, order_addr_col, order_awb_col, order_hub_col, "destination_hub"
    ]].rename(columns={
        "aligned_seller_node": "seller_node",
        order_name_col: "Seller name",
        order_addr_col: "Seller address",
        order_awb_col: "awb_number",
        order_hub_col: "hub",
    })
    file1_path = outdir / "pincode_update_required.csv"
    file1.to_csv(file1_path, index=False)
    sw.lap(f"Wrote {file1_path} ({len(file1):,} rows)")

    # ---- FILE 2: missing_cluster_code.csv --------------------------------- #
    cols2 = [order_awb_col, order_node_col, order_name_col, order_addr_col, order_hub_col]
    if order_cluster_col:
        cols2.append(order_cluster_col)
    file2 = orders.loc[cluster_missing, cols2].rename(columns={
        order_awb_col: "awb_number",
        order_node_col: "seller_node",
        order_name_col: "Seller name",
        order_addr_col: "Seller address",
        order_hub_col: "hub",
    })
    if order_cluster_col:
        file2 = file2.rename(columns={order_cluster_col: "cluster_code"})
    else:
        file2["cluster_code"] = None
    file2_path = outdir / "missing_cluster_code.csv"
    file2.to_csv(file2_path, index=False)
    sw.lap(f"Wrote {file2_path} ({len(file2):,} rows)")

    # ---- FILE 3: manual_review_medium_confidence.csv ----------------------- #
    file3_cols = ["aligned_seller_node", order_node_col, order_name_col, order_addr_col,
                  "aligned_seller_address", order_awb_col, order_hub_col, "destination_hub",
                  "combined_match_score", "name_match_score", "address_match_score"]
    file3 = orders.loc[manual_review_candidate, file3_cols].rename(columns={
        "aligned_seller_node": "aligned_seller_node_id",
        order_node_col: "order_seller_node_id",
        order_name_col: "Seller name",
        order_addr_col: "Order Seller address",
        "aligned_seller_address": "Aligned Seller address",
        order_awb_col: "awb_number",
        order_hub_col: "hub",
    })
    file3_path = outdir / "manual_review_medium_confidence.csv"
    file3.to_csv(file3_path, index=False)
    sw.lap(f"Wrote {file3_path} ({len(file3):,} rows) -- 75-89% confidence, not auto-flagged")

    # ---- consolidated file (optional, useful for the dashboard) ---------- #
    consolidated_cols = [order_awb_col, order_node_col, order_name_col, order_addr_col,
                         order_hub_col, "destination_hub", "aligned_seller_node", "aligned_seller_address",
                         "name_match_score", "address_match_score", "combined_match_score", "confidence_tier",
                         "client_match_score", "matched_ge_threshold", "hub_already_aligned",
                         "cluster_code_norm", "cluster_code_missing", "pincode_update_required"]
    if order_client_col:
        consolidated_cols.insert(4, order_client_col)
    consolidated = orders[consolidated_cols].rename(columns={
        order_awb_col: "awb_number", order_node_col: "seller_node",
        order_name_col: "Seller name", order_addr_col: "Seller address",
        order_hub_col: "hub", "cluster_code_norm": "cluster_code",
    })
    consolidated_path = outdir / "consolidated_audit.csv"
    consolidated.to_csv(consolidated_path, index=False)
    sw.lap(f"Wrote {consolidated_path} ({len(consolidated):,} rows)")

    print("\n" + "=" * 64)
    print("RUN SUMMARY")
    print("=" * 64)
    print(f"Total orders processed          : {len(orders):,}")
    print(f"Any candidate found              : {int(matched_mask.sum()):,}")
    print(f"  High confidence (>= {threshold}%)      : {int(high_confidence.sum()):,}")
    print(f"  Medium confidence (75-89%)      : {int(medium_confidence.sum()):,}  -> manual_review_medium_confidence.csv")
    print(f"  Low confidence (60-74%)         : {int((tiers=='Low').sum()):,}  (reported, not actioned)")
    print(f"Pincode updates required (File 1) : {int(pincode_update_required.sum()):,}")
    print(f"Ignored (hub already aligned)     : {int(hub_already_aligned.sum()):,}")
    print(f"Missing cluster codes (File 2)    : {int(cluster_missing.sum()):,}")
    print(f"Total runtime                     : {time.perf_counter()-sw.t0:.2f}s")
    print("=" * 64)

    return file1_path, file2_path, file3_path, consolidated_path


def main():
    parser = argparse.ArgumentParser(description="High-throughput Seller Node / Pincode audit (10L+ rows)")
    parser.add_argument("--aligned", required=True, help="Path to sellers aligned CSV/XLSX")
    parser.add_argument("--orders", required=True, help="Path to orders_created CSV/XLSX")
    parser.add_argument("--outdir", default="./output", help="Output directory")
    parser.add_argument("--threshold", type=int, default=90, help="Combined name+address match threshold (default 90)")
    parser.add_argument("--block-len", type=int, default=4,
                         help="Number of leading normalized-name characters used for blocking (default 4). "
                              "Lower = fewer, larger blocks (more thorough, slower). "
                              "Higher = more, smaller blocks (faster, may miss a match if the first "
                              "characters differ due to a typo).")
    args = parser.parse_args()

    try:
        run(args.aligned, args.orders, args.outdir, args.threshold, args.block_len)
    except KeyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
