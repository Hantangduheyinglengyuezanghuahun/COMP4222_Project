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
    data = np.ones(len(rows), dtype=np.float32)
    A_ui = sp.coo_matrix((data, (rows, cols+U)), shape=(U+I, U+I))
    A = (A_ui + A_ui.T).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg==0] = 1.0
    Dinv = sp.diags(1.0/deg)
    P = Dinv @ A        # CSR, row-stochastic
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
    print("[INFO] Loaded interaction data:", df.shape)
    if args.category:
        df = df[df['category'].str.lower()==args.category.lower()]
    train = df[df.split=='train'][['user_id','item_id']]
    test_users = df[df.split=='test']['user_id'].unique()
    u_idx, i_idx, umap, imap = index_encode(train)
    print(f"[INFO] Dataset after filtering: {len(u_idx)} users, {len(i_idx)} items, {len(train)} interactions")

    # Early evaluation-only branch: load scores and evaluate
    if args.eval_only:
        scores_path = Path(args.scores_path)
        meta_path = Path(args.meta_path)
        if not (scores_path.exists() and meta_path.exists()):
            raise FileNotFoundError(f"Missing {scores_path} or {meta_path}")
        S = sp.load_npz(scores_path).tocsr()  # shape: [num_test_users, num_items_in_index]
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        users_row = meta['users']       # list of user_ids aligned to rows of S
        items_col = meta['items']       # list of item_ids aligned to columns of S

        # Build ranking per test user from scores
        rankings = {}
        item_col_idx = pd.Index(items_col)
        for r, u in enumerate(users_row):
            row = S.getrow(r)
            if row.nnz == 0: continue
            cols = row.indices; vals = row.data
            order = np.argsort(-vals)[:args.k]
            cols = cols[order]
            rankings[u] = [item_col_idx[c] for c in cols]

        gt = build_ground_truth(df, phase='test')
        rep = evaluate(rankings, gt, k_prec=args.kp, k_rec=args.kr, k_ndcg=args.kn)
        print("[EVAL-ONLY] Loaded PPR scores and evaluated.")
        print("== PPR (mean/std) =="); print(rep)
        return
    
    print("[INFO] Building bipartite graph P...")
    P = build_bipartite(train, umap, imap)
    print("[INFO] Built transition matrix P.")
    P_T = P.T.tocsr()
    print("[INFO] Transposed P to P_T.")
    U, I = len(u_idx), len(i_idx)
    print(f"[INFO] Graph has {U} users and {I} items.")
    # Build seen dict for filtering
    seen = train.groupby('user_id')['item_id'].apply(set).to_dict()
    print(f"[INFO] Built bipartite graph P with shape {P.shape}")

    # SPARSE top-L per user only (no dense allocation)
    rows, cols, vals = [], [], []
    users_row_sparse = []
    last_save = time.time()
    print(f"[INFO] Starting PPR computation for {len(test_users)} test users (sparse only)...")
    for idx, u in enumerate(test_users):
        ui = umap.get(u)
        if ui is None:
            continue
        pr_full = ppr_single(P_T, ui, alpha=args.alpha, iters=args.iters, tol=args.tol)
        item_scores = pr_full[U:U+I]  # items slice

        # filter seen if requested
        if not args.keep_seen:
            ban = seen.get(u, set())
            if ban:
                for b in ban:
                    bi = imap.get(b)
                    if bi is not None:
                        item_scores[bi] = 0.0

        # keep only top-L per user
        L = args.topk_scores if args.topk_scores > 0 else I
        L = min(max(1, L), I)
        cand_idx = np.argpartition(-item_scores, L - 1)[:L]
        if cand_idx.size:
            users_row_sparse.append(str(u))  # store user ids as str for downstream compatibility
            r = len(users_row_sparse) - 1
            cand_vals = item_scores[cand_idx].astype(np.float32, copy=False)
            rows.extend([r] * cand_idx.size)
            cols.extend(cand_idx.tolist())
            vals.extend(cand_vals.tolist())

        # progress bar
        total = len(test_users)
        progress = (idx+1)/total
        bar_len = 40
        filled = int(bar_len*progress)
        bar = '#' * filled + '-' * (bar_len - filled)
        print(f"\r[INFO] Progress |{bar}| {idx+1}/{total} ({progress:.1%})", end='')

        # periodic sparse checkpoint
        if (idx+1) % args.save_every == 0 or (time.time() - last_save) > args.save_secs:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            S_ckpt = sp.csr_matrix((vals, (rows, cols)), shape=(len(users_row_sparse), I))
            sp.save_npz(Path(args.scores_path), S_ckpt)
            with open(Path(args.meta_path), "wb") as f:
                pickle.dump({'users': users_row_sparse, 'items': [str(x) for x in i_idx], 'args': vars(args)}, f)
            last_save = time.time()
            print(f"\n[CKPT] Interim sparse scores saved to {args.scores_path}")

    print()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # Final sparse save
    S = sp.csr_matrix((vals, (rows, cols)), shape=(len(users_row_sparse), I))
    # Append alpha and iters to the final output filenames
    tag = f"_a{args.alpha:g}_it{args.iters}"
    sp_path = Path(args.scores_path)
    args.scores_path = str(sp_path.with_name(f"{sp_path.stem}{tag}{sp_path.suffix}"))
    mp_path = Path(args.meta_path)
    args.meta_path = str(mp_path.with_name(f"{mp_path.stem}{tag}{mp_path.suffix}"))
    sp.save_npz(Path(args.scores_path), S)
    with open(Path(args.meta_path), "wb") as f:
        pickle.dump({'users': users_row_sparse, 'items': [str(x) for x in i_idx], 'args': vars(args)}, f)
    print(f"[CKPT] Sparse PPR scores saved: {args.scores_path}")
    print(f"[CKPT] Meta saved: {args.meta_path}")

    # Evaluation using saved sparse (top-K)
    rankings = {}
    item_col_idx = pd.Index(i_idx)
    for r, u in enumerate(users_row_sparse):
        row = S.getrow(r)
        if row.nnz == 0: continue
        c = row.indices; v = row.data
        order = np.argsort(-v)[:args.k]
        topc = c[order]
        rankings[u] = [item_col_idx[i] for i in topc]
    gt = build_ground_truth(df, phase='test')
    rep = evaluate(rankings, gt, k_prec=args.kp, k_rec=args.kr, k_ndcg=args.kn)
    print("== Sparse PPR (mean/std) ==")
    print(rep)

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
    # scores-only checkpoints
    ap.add_argument("--checkpoint-dir", default="checkpoints_ppr")
    ap.add_argument("--scores-path", default="checkpoints_ppr/ppr_scores.npz")
    ap.add_argument("--meta-path", default="checkpoints_ppr/ppr_scores.meta.pkl")
    ap.add_argument("--topk-scores", type=int, default=200, help="Store top-L item scores per user (0=all)")
    ap.add_argument("--keep-seen", action="store_true", help="Do not filter seen items from scores")
    ap.add_argument("--eval-only", action="store_true", help="Evaluate using an existing scores/meta and exit.")
    ap.add_argument("--save-every", type=int, default=200000000)
    ap.add_argument("--save-secs", type=int, default=120000000)   
    ap.add_argument("--dense-path", default="checkpoints_ppr", help="Dense user-item score matrix (test_users x items)")
    ap.add_argument("--user-vec-path", default="checkpoints_ppr/ppr_user_vectors.npy", help="Saved user PPR vectors (rows kept users)")
    ap.add_argument("--item-vec-path", default="checkpoints_ppr/ppr_item_vectors.npy", help="Saved item vectors (e.g., degree)")
    args = ap.parse_args()
    main(args)