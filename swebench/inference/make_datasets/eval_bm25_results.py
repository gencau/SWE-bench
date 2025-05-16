#!/usr/bin/env python
"""
Compute evaluation metrics for BM25 retrieval on SWE‑bench‑Lite.

Input  : a JSONL file produced by your retriever
         each line => {"instance_id": "...", "hits": [{"docid": "...", "score": ...}, ...]}
Output : a JSONL file with the same structure *plus* all metric columns
Console: aggregate metrics for single‑file and multi‑file bugs
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from datasets import load_dataset, load_from_disk, disable_caching
from swebench.inference.metrics import metrics                # your existing metrics implementation

disable_caching()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def remove_duplicates(topk_files: List[str]) -> List[str]:
    """Remove duplicates while preserving order."""
    if not isinstance(topk_files, list):
        return []
    seen, dedup = set(), []
    for f in topk_files:
        if f not in seen:
            seen.add(f)
            dedup.append(f)
    return dedup


def load_gold(dataset_name_or_path: str, split: str) -> Dict[str, List[str]]:
    """
    Returns a mapping: instance_id -> list(gold_files)
    (extracted from the 'patch' field of SWE‑bench instances)
    """
    try:
        ds = load_dataset(dataset_name_or_path, split=split)
    except Exception:
        ds = load_from_disk(dataset_name_or_path, split=split)

    import re

    pattern = re.compile(r"\-\-\- a/(.+)")
    gold = {}
    for row in ds:
        gold[row["instance_id"]] = pattern.findall(row["patch"])
    logger.info("Loaded gold file lists for %d instances", len(gold))
    return gold


def compute_instance_metrics(
    expected: List[str], predicted: List[str], k: int
) -> Dict[str, Any]:
    """Wrapper around `metrics` helpers that returns one dict per instance."""
    exp = expected
    print(f"There are {len(exp)} files in ground truth")
    if len(predicted) != k:
        print(f"There are {len(predicted)} files in predicted list vs k={k}")
    pred = remove_duplicates(predicted)

    return {
        # single‑file helpers return np.nan when len(exp) > 1
        "precision@2": metrics.compute_precision_at_2_single(exp, pred),
        "precision@k": metrics.compute_precision_at_k(exp, pred, k=len(pred)),
        "recall@1": metrics.compute_recall_at_1_single(exp, pred),
        "recall@2": metrics.compute_recall_at_2_single(exp, pred),
        "recall@k": metrics.compute_recall_at_k(exp, pred, k=len(pred)),
        "MAP": metrics.compute_average_precision(exp, pred),
        "f1@k": metrics.compute_f1_at_k(
            metrics.compute_recall_at_k(exp, pred, k=len(pred)),
            metrics.compute_precision_at_k(exp, pred, k=len(pred)),
            k=5,
        ),
        "hit_rate@k": metrics.hit_rate_at_k(exp, pred, k=len(pred)),
        "all_files_predicted": metrics.all_files_in_predicted(exp, pred),
        "MRR": metrics.mean_reciprocal_rank(exp, pred)
    }


# --------------------------------------------------------------------------- #
# main evaluation                                                             #
# --------------------------------------------------------------------------- #
def evaluate(results_path: str, dataset_name_or_path: str, split: str, out_path: str, k: int):
    gold = load_gold(dataset_name_or_path, split)

    enriched_rows, per_instance_df_rows = [], []

    # ------------------------------------------------------------------ #
    # 1. per‑instance processing                                         #
    # ------------------------------------------------------------------ #
    with Path(results_path).open() as f:
        for line in f:
            obj = json.loads(line)
            iid = obj["instance_id"]

            pred_files = []
            if "all_files" in obj and obj["all_files"]:
                pred_files = obj["all_files"]
            else:
                pred_files = [h["docid"] for h in obj["hits"]]
            exp_files = gold.get(iid, [])

            if not exp_files:
                logger.warning("No gold files for instance %s – skipping", iid)
                continue

            inst_metrics = compute_instance_metrics(exp_files, pred_files, k)

            # attach metrics to JSON row
            obj.update(inst_metrics)
            enriched_rows.append(obj)

            # store minimal info for dataframe aggregation
            per_instance_df_rows.append(
                {
                    "instance_id": iid,
                    "expected_files": exp_files,
                    "topk_files": pred_files,
                    **inst_metrics,
                }
            )

    # ------------------------------------------------------------------ #
    # 2. aggregate metrics                                               #
    # ------------------------------------------------------------------ #
    df = pd.DataFrame(per_instance_df_rows)

    # identify single‑file vs multi‑file bugs
    sf_subset = df[df["recall@1"].notna()]        # exactly 1 gold file
    mf_subset = df[df["recall@2"].notna()]        # 2+ gold files

    def _safe_mean(series):
        return series.dropna().mean()

    # single‑file aggregates
    sf_r1 = _safe_mean(sf_subset["recall@1"])
    sf_ap = _safe_mean(sf_subset["MAP"])
    sf_rk = _safe_mean(sf_subset["recall@k"])
    sf_pk = _safe_mean(sf_subset["precision@k"])
    sf_f1_k = _safe_mean(sf_subset["f1@k"])
    sf_hit_k = _safe_mean(sf_subset["hit_rate@k"])
    sf_all_pred = _safe_mean(sf_subset["all_files_predicted"])
    sf_mrr = _safe_mean(sf_subset["MRR"])

    # multi‑file aggregates
    mf_p2 = _safe_mean(mf_subset["precision@2"])
    mf_r2 = _safe_mean(mf_subset["recall@2"])
    mf_ap = _safe_mean(mf_subset["MAP"])
    mf_rk = _safe_mean(mf_subset["recall@k"])
    mf_pk = _safe_mean(mf_subset["precision@k"])
    mf_f1_k = _safe_mean(mf_subset["f1@k"])
    mf_hit_k = _safe_mean(mf_subset["hit_rate@k"])
    mf_all_pred = _safe_mean(mf_subset["all_files_predicted"])
    mf_mrr = _safe_mean(mf_subset["MRR"])

    # overall corpus‑level F1 (macro, like your original script)
    def overall_f1(subset, recall_col):
        tp = fp = fn = 0
        for _, row in subset.iterrows():
            gt_set, pr_set = set(row["expected_files"]), set(row["topk_files"])
            tp += len(gt_set & pr_set)
            fp += len(pr_set - gt_set)
            fn += len(gt_set - pr_set)
        if tp == 0 or tp + fp == 0 or tp + fn == 0:
            return np.nan
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * precision * recall / (precision + recall)

    overall_sf_f1 = overall_f1(sf_subset, "recall@1")
    overall_mf_f1 = overall_f1(mf_subset, "recall@2")

    # ------------------------------------------------------------------ #
    # 3. print summary                                                   #
    # ------------------------------------------------------------------ #
    print("---- Single‑file bugs ----")
    print(f"Recall@1              : {sf_r1:.4f}")
    print(f"Average Precision     : {sf_ap:.4f}")
    print(f"Precision@k           : {sf_pk:.4f}")
    print(f"F1 (overall)          : {overall_sf_f1:.4f}")
    print(f"Recall@k              : {sf_rk:.4f}")
    print(f"F1@k                  : {sf_f1_k:.4f}")
    print(f"Hit‑Rate@k            : {sf_hit_k:.4f}")
    print(f"All‑files‑predicted   : {sf_all_pred:.4f}\n")
    print(f"Mean Reciprocal Rank  : {sf_mrr:.4f}\n")

    print("---- Multi‑file bugs ----")
    print(f"Precision@2           : {mf_p2:.4f}")
    print(f"Precision@k           : {mf_pk:.4f}")
    print(f"Recall@2              : {mf_r2:.4f}")
    print(f"Average Precision     : {mf_ap:.4f}")
    print(f"F1 (overall)          : {overall_mf_f1:.4f}")
    print(f"Recall@k              : {mf_rk:.4f}")
    print(f"F1@k                  : {mf_f1_k:.4f}")
    print(f"Hit‑Rate@k            : {mf_hit_k:.4f}")
    print(f"All‑files‑predicted   : {mf_all_pred:.4f}")
    print(f"Mean Reciprocal Rank  : {mf_mrr:.4f}\n")


    # ------------------------------------------------------------------ #
    # 4. write enriched JSONL                                            #
    # ------------------------------------------------------------------ #
    out_path = Path(out_path)
    with out_path.open("w") as w:
        for obj in enriched_rows:
            w.write(json.dumps(obj) + "\n")
    logger.info("Wrote enriched results with metrics to %s", out_path)


# --------------------------------------------------------------------------- #
# entry‑point                                                                 #
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results_path", required=True, help="Path to BM25 JSONL predictions"
    )
    p.add_argument(
        "--dataset_name_or_path",
        default="princeton-nlp/SWE-bench_Verified",
        help="HF dataset name or local path with gold patches",
    )
    p.add_argument("--split", default="test", help="Dataset split")
    p.add_argument(
        "--output",
        required=False,
        help="Where to store JSONL with metric columns (default: <results>_with_metrics.jsonl)",
    )
    p.add_argument("--k", required=False, default=5)
    args = p.parse_args()

    output_path = (
        args.output
        if args.output
        else Path(args.results_path).with_name(
            Path(args.results_path).stem + "_with_metrics.jsonl"
        )
    )

    evaluate(
        args.results_path,
        args.dataset_name_or_path,
        args.split,
        output_path,
        args.k
    )


if __name__ == "__main__":
    main()
