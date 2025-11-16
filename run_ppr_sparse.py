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
    if args.category:
        df = df[df['category'].str.lower()==args.category.lower()]
    train = df[df.split=='train'][['user_id','item_id']]
    test_users = df[df.split=='test']['user_id'].unique()
    u_idx, i_idx, umap, imap = index_encode(train)

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

    P = build_bipartite(train, umap, imap)
    P_T = P.T.tocsr()
    U, I = len(u_idx), len(i_idx)
    seen = train.groupby('user_id')['item_id'].apply(set).to_dict()

    # DENSE full matrix (users x items)
    dense_scores = np.zeros((len(test_users), I), dtype=np.float32)
    kept_mask = np.zeros((len(test_users),), dtype=bool)  # mark users actually computed (in umap)

    rows, cols, vals = [], [], []  # still keep sparse incremental if you want
    users_row_sparse = []
    last_save = time.time()

    for idx, u in enumerate(test_users):
        ui = umap.get(u)
        if ui is None:
            continue
        pr_full = ppr_single(P_T, ui, alpha=args.alpha, iters=args.iters, tol=args.tol)
        item_scores = pr_full[U:U+I]  # slice for items only

        # filter seen if requested
        if not args.keep_seen:
            ban = seen.get(u, set())
            if ban:
                for b in ban:
                    bi = imap.get(b)
                    if bi is not None:
                        item_scores[bi] = 0.0

        dense_scores[idx] = item_scores
        kept_mask[idx] = True

        # Optional sparse top-L (keep existing behavior)
        L = args.topk_scores if args.topk_scores > 0 else I
        cand_idx = np.argpartition(-item_scores, L - 1)[:L]
        pairs = [(i, float(item_scores[i])) for i in cand_idx]
        if pairs:
            users_row_sparse.append(u)
            r = len(users_row_sparse) - 1
            for i, s in pairs:
                rows.append(r); cols.append(i); vals.append(s)

        # progress bar (unchanged)
        total = len(test_users)
        progress = (idx+1)/total
        bar_len = 40
        filled = int(bar_len*progress)
        bar = '#' * filled + '-' * (bar_len - filled)
        print(f"\r[INFO] Progress |{bar}| {idx+1}/{total} ({progress:.1%})", end='')

        if (idx+1) % args.save_every == 0 or (time.time() - last_save) > args.save_secs:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            # interim dense save
            np.save(args.dense_path, dense_scores)
            # interim sparse save (optional)
            if len(users_row_sparse):
                S = sp.csr_matrix((vals, (rows, cols)), shape=(len(users_row_sparse), I))
                sp.save_npz(Path(args.scores_path), S)
                with open(Path(args.meta_path), "wb") as f:
                    pickle.dump({'users': users_row_sparse, 'items': list(i_idx), 'args': vars(args)}, f)
            # save user/item vectors
            np.save(args.user_vec_path, dense_scores[kept_mask])
            # simple item vector: degree from train
            item_degree = train.groupby('item_id').size().reindex(i_idx, fill_value=0).to_numpy(dtype=np.float32)
            np.save(args.item_vec_path, item_degree)
            last_save = time.time()
            print(f"\n[CKPT] Interim dense scores saved to {args.dense_path}")

    print()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # Final dense save
    np.save(args.dense_path, dense_scores)
    # Final sparse (optional)
    if len(users_row_sparse):
        S = sp.csr_matrix((vals, (rows, cols)), shape=(len(users_row_sparse), I))
        sp.save_npz(Path(args.scores_path), S)
        with open(Path(args.meta_path), "wb") as f:
            pickle.dump({'users': users_row_sparse, 'items': list(i_idx), 'args': vars(args)}, f)
    # User/item vectors
    np.save(args.user_vec_path, dense_scores[kept_mask])
    item_degree = train.groupby('item_id').size().reindex(i_idx, fill_value=0).to_numpy(dtype=np.float32)
    np.save(args.item_vec_path, item_degree)
    print(f"[CKPT] Dense PPR scores saved: {args.dense_path}")
    print(f"[CKPT] User PPR vectors: {args.user_vec_path}")
    print(f"[CKPT] Item degree vectors: {args.item_vec_path}")

    # Evaluation using dense (top-K)
    rankings = {}
    for idx, u in enumerate(test_users):
        if not kept_mask[idx]:
            continue
        row = dense_scores[idx]
        topk = np.argpartition(-row, args.k)[:args.k]
        topk = topk[np.argsort(-row[topk])]
        rankings[u] = [i_idx[i] for i in topk]
    gt = build_ground_truth(df, phase='test')
    rep = evaluate(rankings, gt, k_prec=args.kp, k_rec=args.kr, k_ndcg=args.kn)
    print("== Dense PPR (mean/std) ==")
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
    ap.add_argument("--dense-path", default="checkpoints_ppr/ppr_scores_dense.npy", help="Dense user-item score matrix (test_users x items)")
    ap.add_argument("--user-vec-path", default="checkpoints_ppr/ppr_user_vectors.npy", help="Saved user PPR vectors (rows kept users)")
    ap.add_argument("--item-vec-path", default="checkpoints_ppr/ppr_item_vectors.npy", help="Saved item vectors (e.g., degree)")
    args = ap.parse_args()
    main(args)