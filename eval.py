# project/eval.py
import argparse
import numpy as np, pandas as pd
from collections import defaultdict
from pathlib import Path

def build_ground_truth(df, phase="test"):
    gt = defaultdict(set)
    for r in df[df.split==phase][['user_id','item_id']].itertuples(index=False):
        gt[r.user_id].add(r.item_id)
    return gt

def ndcg_at_k(pred, truth, k=10):
    dcg=0.0
    for idx,item in enumerate(pred[:k], start=1):
        if item in truth: dcg += 1.0/np.log2(idx+1)
    idcg = sum(1.0/np.log2(i+1) for i in range(1, min(len(truth), k)+1))
    return dcg/idcg if idcg>0 else 0.0

def evaluate(rankings, ground_truth, k_prec=10, k_rec=20, k_ndcg=10):
    rows=[]
    for u, pred in rankings.items():
        truth = ground_truth.get(u, set())
        if not truth: continue
        hitk = len(set(pred[:k_rec]) & truth)
        rows.append((
            u,
            hitk/ k_prec,
            hitk/ len(truth),
            ndcg_at_k(pred, truth, k_ndcg)
        ))
    df = pd.DataFrame(rows, columns=['user_id','P@10','R@20','NDCG@10'])
    return df.describe().loc[['mean','std']]

def main(args):
    path = Path(args.input)
    df = pd.read_csv(path)
    if args.category:
        df = df[df['category'].str.lower()==args.category.lower()]
    train = df[df.split=='train']
    users = df[df.split=='test']['user_id'].unique()

    # Popularity baseline
    from baselines.popularity import fit_popularity, recommend_popularity
    pop, seen = fit_popularity(train)
    rankings = recommend_popularity(pop, seen, users, K=args.k)

    gt = build_ground_truth(df, 'test')
    overall = evaluate(rankings, gt, k_prec=args.kp, k_rec=args.kr, k_ndcg=args.kn)
    print("== Overall (mean/std) ==")
    print(overall)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/interactions.split.csv")
    ap.add_argument("--category", default=None, help="Electronics / Video_Games / ...")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--kp", type=int, default=10)
    ap.add_argument("--kr", type=int, default=20)
    ap.add_argument("--kn", type=int, default=10)
    args = ap.parse_args()
    main(args)
