import argparse, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
from eval import build_ground_truth, evaluate
import os
import pickle
import time

def index_encode(train):
    u_idx = pd.Index(train['user_id'].unique())
    i_idx = pd.Index(train['item_id'].unique())
    umap = {u:k for k,u in enumerate(u_idx)}
    imap = {i:k for k,i in enumerate(i_idx)}
    return u_idx, i_idx, umap, imap

def build_bipartite(train, umap, imap):
    U, I = len(umap), len(imap)
    ui = train[['user_id','item_id']].drop_duplicates()
    rows = ui['user_id'].map(umap).to_numpy()
    cols = ui['item_id'].map(imap).to_numpy()
    # 二部图 -> 无向邻接（用户块在 [0,U)，物品块在 [U,U+I)）
    data = np.ones(len(rows), dtype=np.float32)
    A_ui = sp.coo_matrix((data, (rows, cols+U)), shape=(U+I, U+I))
    A = (A_ui + A_ui.T).tocsr()
    # 行归一 -> 随机游走转移矩阵 P（行随机）
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg==0] = 1.0
    Dinv = sp.diags(1.0/deg)
    P = Dinv @ A        # CSR, 行随机
    return P

def ppr_single(P_T, u_global, alpha=0.15, iters=30, tol=1e-6):
    n = P_T.shape[0]
    y = np.zeros(n, dtype=np.float32)
    y[u_global] = 1.0
    e = y.copy()
    for _ in range(iters):
        y_next = alpha*e + (1.0-alpha)*(P_T @ y)
        if np.linalg.norm(y_next - y, 1) < tol: 
            y = y_next; break
        y = y_next
    return y

def main(args):
    df = pd.read_csv(Path(args.input))
    if args.category:
        df = df[df['category'].str.lower()==args.category.lower()]
    train = df[df.split=='train'][['user_id','item_id']]
    test_users = df[df.split=='test']['user_id'].unique()
    u_idx, i_idx, umap, imap = index_encode(train)

    # Early evaluation-only branch: load checkpoint rankings and evaluate directly
    if args.eval_only:
        ckpt_path = os.path.join(args.checkpoint_dir, args.ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        with open(ckpt_path, "rb") as f:
            state = pickle.load(f)
        rankings = state.get('rankings', {})
        if not rankings:
            raise ValueError("Checkpoint has no rankings to evaluate.")
        gt = build_ground_truth(df, phase='test')
        rep = evaluate(rankings, gt, k_prec=args.kp, k_rec=args.kr, k_ndcg=args.kn)
        print("[EVAL-ONLY] Loaded rankings from checkpoint.")
        print("== PPR (mean/std) =="); print(rep)
        return

    P = build_bipartite(train, umap, imap)
    P_T = P.T.tocsr()
    print(f"[INFO] Built PPR graph: {P.shape}, nnz={P.nnz}")

    seen = train.groupby('user_id')['item_id'].apply(set).to_dict()
    U, I = len(u_idx), len(i_idx)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, args.ckpt_name)

    # Resume state
    rankings = {}
    processed = set()
    start_idx = 0
    if args.resume and os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            state = pickle.load(f)
        rankings = state.get('rankings', {})
        processed = set(rankings.keys())
        for idx, u in enumerate(test_users):
            if u not in processed:
                start_idx = idx
                break
        print(f"[CKPT] Resumed from {ckpt_path}; processed {len(processed)} users; starting at index {start_idx}")

    last_save = time.time()
    for idx in range(start_idx, len(test_users)):
        u = test_users[idx]
        if u not in umap:
            continue
        pr = ppr_single(P_T, umap[u], alpha=args.alpha, iters=args.iters, tol=args.tol)
        scores = pr[U:U+I]
        ban = seen.get(u, set())
        cand_idx = np.argpartition(-scores, args.k*2)[:args.k*2]
        pairs = [(i, float(scores[i])) for i in cand_idx if i_idx[i] not in ban]
        pairs.sort(key=lambda x: x[1], reverse=True)
        rankings[u] = [i_idx[i] for i,_ in pairs[:args.k]]

        total = len(test_users)
        progress = (idx+1)/total
        bar_len = 40
        filled = int(bar_len*progress)
        bar = '#' * filled + '-' * (bar_len - filled)
        print(f"\r[INFO] Progress |{bar}| {idx+1}/{total} ({progress:.1%})", end='')

        if (idx+1) % args.save_every == 0 or (time.time() - last_save) > args.save_secs:
            with open(ckpt_path, "wb") as f:
                pickle.dump({'rankings': rankings,'args': vars(args),'last_index': idx}, f)
            last_save = time.time()
            print(f"\n[CKPT] Saved interim rankings to {ckpt_path}")

    print()
    with open(ckpt_path, "wb") as f:
        pickle.dump({'rankings': rankings,'args': vars(args),'last_index': len(test_users)-1}, f)
    print(f"[CKPT] Final checkpoint saved: {ckpt_path}")

    gt = build_ground_truth(df, phase='test')
    rep = evaluate(rankings, gt, k_prec=args.kp, k_rec=args.kr, k_ndcg=args.kn)
    print("== PPR (mean/std) =="); print(rep)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/interactions.Video_Games.split.csv")
    ap.add_argument("--category", default=None)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--kp", type=int, default=10)
    ap.add_argument("--kr", type=int, default=20)
    ap.add_argument("--kn", type=int, default=10)
    ap.add_argument("--checkpoint-dir", default="checkpoints_ppr")
    ap.add_argument("--ckpt-name", default="ppr_rankings.pkl")
    ap.add_argument("--resume", action="store_true", help="Resume unfinished PPR computation.")
    ap.add_argument("--eval-only", action="store_true", help="Evaluate using an existing checkpoint and exit.")
    ap.add_argument("--save-every", type=int, default=100000000)
    ap.add_argument("--save-secs", type=int, default=120000000)
    args = ap.parse_args()
    main(args)
