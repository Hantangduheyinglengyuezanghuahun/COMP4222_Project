import argparse, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
from eval import build_ground_truth, evaluate  # 直接复用你的评测

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
    P = build_bipartite(train, umap, imap)
    P_T = P.T.tocsr()
    print(f"[INFO] Built PPR graph: {P.shape}, nnz={P.nnz}")

    # 训练集中已见过的物品，用于过滤
    seen = train.groupby('user_id')['item_id'].apply(set).to_dict()
    U, I = len(u_idx), len(i_idx)

    rankings = {}
    current_done = 0
    for u in test_users:
        if u not in umap:      # 只在val/test出现的用户，跳过
            continue
        pr = ppr_single(P_T, umap[u], alpha=args.alpha, iters=args.iters, tol=args.tol)
        # 只取物品块分数
        scores = pr[U:U+I]
        # 过滤已见
        ban = seen.get(u, set())
        # 拿到未见物品的排名
        cand_idx = np.argpartition(-scores, args.k*2)[:args.k*2]  # 先粗选
        pairs = [(i, float(scores[i])) for i in cand_idx if i_idx[i] not in ban]
        pairs.sort(key=lambda x: x[1], reverse=True)
        top_items = [i_idx[i] for i,_ in pairs[:args.k]]
        rankings[u] = top_items
        current_done += 1
        total = len(test_users) or 1
        progress = current_done / total
        bar_len = 40
        filled = int(bar_len * progress)
        bar = '#' * filled + '-' * (bar_len - filled)
        print(f"\r[INFO] Progress |{bar}| {current_done}/{total} ({progress:.1%})", end='')
        if current_done == total:
            print()

    gt = build_ground_truth(df, phase='test')
    rep = evaluate(rankings, gt, k_prec=args.kp, k_rec=args.kr, k_ndcg=args.kn)
    print("== PPR (mean/std) =="); print(rep)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/interactions.Video_Games.split.csv")
    ap.add_argument("--category", default=None)       # Electronics / Video_Games
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--kp", type=int, default=10)
    ap.add_argument("--kr", type=int, default=20)
    ap.add_argument("--kn", type=int, default=10)
    args = ap.parse_args(); main(args)
