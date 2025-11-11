#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Implicit-ALS baseline runner for Amazon Reviews 2023-style splits.
- Input: a CSV like project/data/interactions.split.csv with columns:
    user_id, item_id, rating (optional), timestamp (optional), split, category (optional)
- Output: prints mean/std of P@10, R@20, NDCG@10 on the test split.
- Usage examples:
    python run_als.py --input interactions.split.csv
    python run_als.py --input interactions.split.csv --category Electronics --use_rating
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from implicit.als import AlternatingLeastSquares

# -------------------------
# Evaluation helpers
# -------------------------

def build_ground_truth(df: pd.DataFrame, phase: str = "test"):
    """Return {user_id: set(item_id)} for the given phase split."""
    gt = {}
    for u, g in df[df["split"] == phase].groupby("user_id"):
        gt[u] = set(g["item_id"].tolist())
    return gt

def ndcg_at_k(pred, truth, k=10):
    """Binary relevance NDCG@K."""
    dcg = 0.0
    for idx, item in enumerate(pred[:k], start=1):
        if item in truth:
            dcg += 1.0 / np.log2(idx + 1)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(truth), k) + 1))
    return (dcg / idcg) if idcg > 0 else 0.0

def evaluate(rankings, ground_truth, k_prec=10, k_rec=20, k_ndcg=10):
    """Return a pandas describe() (mean/std) table for P@k, R@k, NDCG@k."""
    rows = []
    for u, pred in rankings.items():
        truth = ground_truth.get(u, set())
        if not truth:
            continue
        hit = len(set(pred[:k_rec]) & truth)
        prec = hit / k_prec
        rec = hit / len(truth)
        ndcg = ndcg_at_k(pred, truth, k_ndcg)
        rows.append((u, prec, rec, ndcg))
    df = pd.DataFrame(rows, columns=["user_id", "P@10", "R@20", "NDCG@10"])
    return df.describe().loc[["mean", "std"]] if not df.empty else df

# -------------------------
# ALS core
# -------------------------

def build_index(train: pd.DataFrame):
    users = pd.Index(train["user_id"].unique())
    items = pd.Index(train["item_id"].unique())
    u2i = {u: i for i, u in enumerate(users)}
    it2i = {i: j for j, i in enumerate(items)}
    return users, items, u2i, it2i

def build_mats(train: pd.DataFrame, u2i, it2i, alpha: float = 40.0, use_rating: bool = True):
    # implicit strength r: rating if provided else 1.0
    if use_rating and "rating" in train.columns:
        r = train["rating"].astype(float).to_numpy()
    else:
        r = np.ones(len(train), dtype=float)

    ui = train["user_id"].map(u2i).to_numpy()
    ii = train["item_id"].map(it2i).to_numpy()

    # User×Item binary matrix for filtering already-liked items when recommending
    UI = coo_matrix((np.ones_like(r), (ui, ii)), shape=(len(u2i), len(it2i))).tocsr()

    # Item×User confidence matrix C = alpha * r (Hu-Koren-Volinsky style)
    Ciu = coo_matrix((alpha * r, (ii, ui)), shape=(len(it2i), len(u2i))).tocsr()
    return UI, Ciu

def train_als(Ciu: csr_matrix, factors=128, reg=1e-2, iters=20):
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=reg,
        iterations=iters,
        calculate_training_loss=False,
    )
    model.fit(Ciu)  # expects item×user
    return model

def recommend_als(model, UI: csr_matrix, users_index, items_index, test_users, u2i, it2i, K=200):
    rankings = {}
    for u in test_users:
        # skip users that never appeared in train
        if u not in u2i:
            continue
        uid = u2i[u]
        # implicit filters already-liked items by default
        recs = model.recommend(
            userid=uid,
            user_items=UI,
            N=K,
            filter_already_liked_items=True,
            recalculate_user=True,
        )
        item_ids = [items_index[iid] for iid, _ in recs]
        rankings[u] = item_ids
    return rankings

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Path to interactions.split.csv")
    ap.add_argument("--category", type=str, default=None, help="Optional category filter, e.g., Electronics")
    ap.add_argument("--factors", type=int, default=128)
    ap.add_argument("--reg", type=float, default=1e-2)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=40.0)
    ap.add_argument("--use_rating", action="store_true", help="Use 'rating' column as implicit strength")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--kp", type=int, default=10)
    ap.add_argument("--kr", type=int, default=20)
    ap.add_argument("--kn", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(Path(args.input))
    if args.category is not None:
        if "category" not in df.columns:
            raise ValueError("No 'category' column in input, but --category was provided.")
        df = df[df["category"].str.lower() == args.category.lower()]

    train = df[df["split"] == "train"][["user_id", "item_id"] + (["rating"] if "rating" in df.columns else [])].copy()

    users_index, items_index, u2i, it2i = build_index(train)
    UI, Ciu = build_mats(train, u2i, it2i, alpha=args.alpha, use_rating=args.use_rating)

    model = train_als(Ciu, factors=args.factors, reg=args.reg, iters=args.iters)

    test_users = df[df["split"] == "test"]["user_id"].unique()
    rankings = recommend_als(model, UI, users_index, items_index, test_users, u2i, it2i, K=args.k)

    gt = build_ground_truth(df, phase="test")
    report = evaluate(rankings, gt, k_prec=args.kp, k_rec=args.kr, k_ndcg=args.kn)
    print("== ALS (mean/std) ==")
    print(report.to_string())
    
if __name__ == "__main__":
    main()
