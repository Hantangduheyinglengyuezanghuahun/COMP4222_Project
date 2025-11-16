#!/usr/bin/env python3
import argparse, os, pickle, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from torch_geometric.nn import to_hetero
from eval import build_ground_truth, evaluate

# Import only required pieces; avoid import *
from run_graphsage import (
    load_preprocessed, load_node2vec,
    SAGEEncoder, ItemRatingFusion, UserCommentFusion, ItemCommentFusion
)

try:
    import scipy.sparse as sp
    _SPARSE_OK = True
except Exception:
    _SPARSE_OK = False
    sp = None



def normalize_scores(vec, method="none"):
    if method == "none": return vec
    if method == "z":
        m, s = vec.mean(), vec.std()
        return (vec - m) / (s + 1e-12)
    if method == "minmax":
        vmin, vmax = vec.min(), vec.max()
        return vec*0.0 if vmax <= vmin + 1e-12 else (vec - vmin) / (vmax - vmin)
    return vec

def load_ppr_scores(scores_path: str, meta_path: str):
    if not _SPARSE_OK:
        raise RuntimeError("Install scipy to load .npz CSR scores: pip install scipy")
    S = sp.load_npz(scores_path).tocsr()
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    users_row = meta['users']  # list of user_ids aligned with S rows
    items_col = meta['items']  # list of item_ids aligned with S cols
    row_map = {str(u): r for r, u in enumerate(users_row)}
    return S, row_map, items_col

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="data/loaded_data")
    ap.add_argument("--category", default=None)
    ap.add_argument("--use-a2", action="store_true")
    ap.add_argument("--gs-ckpt", required=True, help="GraphSAGE checkpoint (.pt)")
    # PPR SCORE CHECKPOINTS (scores only)
    ap.add_argument("--ppr-scores", default = "checkpoints_ppr/ppr_scores.npz", help="PPR user-item score matrix (.npz, CSR)")
    ap.add_argument("--ppr-meta", default = "checkpoints_ppr/ppr_scores.meta.pkl", help="Meta pickle with {'users': [...], 'items': [...]} aligned to PPR scores")
    # Rebuild features as in training
    ap.add_argument("--n2v-dim", type=int, default=128, help="Node2Vec dim (default from ckpt args if present)")
    ap.add_argument("--use-rating", action="store_true")
    ap.add_argument("--use-comment-user", action="store_true")
    ap.add_argument("--use-comment-item", action="store_true")
    ap.add_argument("--comment-emb", default="data/textfeat/comment_text_emb.npy")
    ap.add_argument("--comment-index", default="data/textfeat/comment_text_index.csv")
    # Inference/eval and gamma sweep
    ap.add_argument("--norm", choices=["none","z","minmax"], default="none")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--kp", type=int, default=10)
    ap.add_argument("--kr", type=int, default=20)
    ap.add_argument("--kn", type=int, default=10)
    ap.add_argument("--filter-seen", action="store_true")
    ap.add_argument("--gamma-step", type=float, default=0.5, help="Sweep γ in [0,1] with this step")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and (not args.cpu) else 'cpu')
    
    class LinkPredictor(nn.Module):
        def __init__(self, hidden=64, out=64):
            super().__init__()
            self.user_id_emb = nn.Embedding(data['user'].num_nodes, hidden)
            self.item_id_emb = nn.Embedding(data['item'].num_nodes, hidden)
            # Optionally init with Node2Vec (not required for fusion to work)
            n2v_local = load_node2vec(args.dataset_dir, (args.category if args.category else "all"), hidden)
            if n2v_local is not None:
                with torch.no_grad():
                    self.user_id_emb.weight.copy_(n2v_local['user_emb'].to(device=device, dtype=torch.float32))
                    self.item_id_emb.weight.copy_(n2v_local['item_emb'].to(device=device, dtype=torch.float32))
            self.encoder = to_hetero(SAGEEncoder(hidden, out), data.metadata())

        def forward(self, data):
            # data['item'].x already fused to 64-d above
            return self.encoder(data.x_dict, data.edge_index_dict)

        def score(self, u_emb, i_emb):
            return (u_emb * i_emb).sum(dim=-1)
    # Interactions and graph
    prefix = (args.category if args.category else "all")
    pq = os.path.join(args.dataset_dir, f"{prefix}_interactions.parquet")
    csv = os.path.join(args.dataset_dir, f"{prefix}_interactions.csv")
    df = pd.read_parquet(pq) if os.path.exists(pq) else pd.read_csv(csv)

    loaded = load_preprocessed(args.dataset_dir, args.category, args.use_a2)
    data = loaded['data'].to(device)
    u_idx = pd.Index(loaded['u_idx'])
    i_idx = pd.Index(loaded['i_idx'])
    umap = loaded['umap']; imap = loaded['imap']

    # GraphSAGE checkpoint and features
    ck = torch.load(args.gs_ckpt, map_location=device)
    ck_args = ck.get('args', {})
    hidden = ck_args.get('hidden', 128); outdim = ck_args.get('out', 64); dropout = ck_args.get('dropout', 0.2)
    n2v_dim = args.n2v_dim if args.n2v_dim is not None else ck_args.get('n2v_dim', hidden)
    n2v = load_node2vec(args.dataset_dir, prefix, n2v_dim)
    if n2v is None: raise FileNotFoundError(f"Node2Vec ({n2v_dim}) not found.")
    data['user'].x = n2v['user_emb'].to(device=device, dtype=torch.float32)
    data['item'].x = n2v['item_emb'].to(device=device, dtype=torch.float32)

    # Optional: rating fusion
    item_rating_feat = None
    if args.use_rating:
        # rebuild rating feat from df
        tr = df[df.split=='train'][['item_id','rating']]
        if not tr.empty and 'rating' in df.columns:
            istats = tr.groupby('item_id')['rating'].agg(['mean','count'])
            gmean = float(istats['mean'].mean()) if len(istats) else 0.0
            i_mean = istats['mean'].reindex(pd.Index(i_idx), fill_value=gmean).to_numpy(dtype=np.float32)
            i_cnt  = istats['count'].reindex(pd.Index(i_idx), fill_value=0).to_numpy(dtype=np.float32)
            mean_z = (i_mean - float(i_mean.mean())) / (float(i_mean.std()) + 1e-8)
            cnt_l  = np.log1p(i_cnt)
            item_rating_feat = torch.from_numpy(np.stack([mean_z, cnt_l], axis=1)).to(device=device, dtype=torch.float32)
        ck_fuse = ck.get('fuse_state'); ck_meta = ck.get('fuse_meta', {}) or {}
        if ck_fuse is not None and item_rating_feat is not None:
            rf = ItemRatingFusion(
                n2v_dim=ck_meta.get('n2v_dim', data['item'].x.size(1)),
                rating_dim=ck_meta.get('rating_dim', item_rating_feat.size(1)),
                out_dim=ck_meta.get('out_dim', data['item'].x.size(1)),
                hid_dim=ck_meta.get('hid_dim', None)
            ).to(device)
            rf.load_state_dict(ck_fuse)
            with torch.no_grad():
                data['item'].x = rf(data['item'].x, item_rating_feat)

    # Optional: comment fusion
    uc_state = ck.get('user_comment_fuse_state'); uc_meta = ck.get('user_comment_meta', {}) or {}
    ic_state = ck.get('item_comment_fuse_state'); ic_meta = ck.get('item_comment_meta', {}) or {}
    if args.use_comment_user or args.use_comment_item:
        if os.path.exists(args.comment_emb) and os.path.exists(args.comment_index):
            c_emb = np.load(args.comment_emb); c_idx = pd.read_csv(args.comment_index)
            Dc = c_emb.shape[1]
            if args.use_comment_user:
                usr_sum = torch.zeros((len(u_idx), Dc), dtype=torch.float32); usr_cnt = torch.zeros((len(u_idx),), dtype=torch.float32)
            if args.use_comment_item:
                itm_sum = torch.zeros((len(i_idx), Dc), dtype=torch.float32); itm_cnt = torch.zeros((len(i_idx),), dtype=torch.float32)
            for r in c_idx.itertuples(index=False):
                row = r.row
                if row < 0 or row >= c_emb.shape[0]: continue
                v = torch.from_numpy(c_emb[row]).float()
                if args.use_comment_user:
                    ui = umap.get(str(r.user_id))
                    if ui is not None: usr_sum[ui] += v; usr_cnt[ui] += 1
                if args.use_comment_item:
                    ii = imap.get(str(r.item_id))
                    if ii is not None: itm_sum[ii] += v; itm_cnt[ii] += 1
            if args.use_comment_user and usr_cnt.sum() > 0:
                usr_cnt[usr_cnt==0]=1; usr_feat = (usr_sum / usr_cnt.unsqueeze(1)).to(device)
                uf = UserCommentFusion(uc_meta.get('base_dim', data['user'].x.size(1)), usr_feat.size(1),
                                       uc_meta.get('out_dim', data['user'].x.size(1)), uc_meta.get('hid_dim', None)).to(device)
                if uc_state is not None: uf.load_state_dict(uc_state)
                with torch.no_grad():
                    data['user'].x = uf(data['user'].x, usr_feat)
            if args.use_comment_item and itm_cnt.sum() > 0:
                itm_cnt[itm_cnt==0]=1; itm_feat = (itm_sum / itm_cnt.unsqueeze(1)).to(device)
                ifm = ItemCommentFusion(ic_meta.get('base_dim', data['item'].x.size(1)), itm_feat.size(1),
                                        ic_meta.get('out_dim', data['item'].x.size(1)), ic_meta.get('hid_dim', None)).to(device)
                if ic_state is not None: ifm.load_state_dict(ic_state)
                with torch.no_grad():
                    data['item'].x = ifm(data['item'].x, itm_feat)

    # GraphSAGE model
    model = LinkPredictor(hidden=hidden, out=outdim).to(device)
    model.load_state_dict(ck['model_state'])
    model.eval()
    with torch.no_grad():
        out = model(data)
        U = out['user']; I = out['item']
    I_norm = I / (I.norm(dim=1, keepdim=True) + 1e-12)

    # Load PPR scores (scores only)
    if args.ppr_scores.endswith(".npy"):
        ppr_dense = np.load(args.ppr_scores)  # shape: test_users x items
        with open(args.ppr_meta, "rb") as f:
            meta = pickle.load(f)
        ppr_users = meta['users']
        ppr_items = meta['items']
        ppr_user_map = {str(u): i for i, u in enumerate(ppr_users)}
        ppr_item_map = {str(it): j for j, it in enumerate(ppr_items)}
        # Remap item columns to global index
        col_remap = np.full((len(ppr_items),), -1, dtype=np.int64)
        for j, it in enumerate(ppr_items):
            gi = imap.get(str(it))
            if gi is not None:
                col_remap[j] = gi
    else:
        S, ppr_row_map, ppr_items = load_ppr_scores(args.ppr_scores, args.ppr_meta)
        ppr_items_idx = pd.Index(ppr_items)
        # Precompute PPR col remap: ppr col -> global item index
        col_remap = np.full((len(ppr_items_idx),), -1, dtype=np.int64)
        for j, it in enumerate(ppr_items_idx):
            ii = imap.get(str(it))
            if ii is not None:
                col_remap[j] = ii

    # Seen items to filter
    seen = df[df.split=='train'].groupby('user_id')['item_id'].apply(set).to_dict()
    test_users = df[df.split=='test']['user_id'].unique()
    gt = build_ground_truth(df, phase='test')

    gammas = np.round(np.arange(0.0, 1.0 + 1e-8, args.gamma_step), 4).tolist()
    results = []
    for gamma in gammas:
        rankings = {}
        total_users = len(test_users)
        for u in test_users:
            uid = umap.get(str(u))
            if uid is None:
                continue
            ue = U[uid:uid+1]
            ue = ue / (ue.norm(dim=1, keepdim=True) + 1e-12)
            gs = (ue @ I_norm.T).squeeze(0).detach().cpu().numpy()
            # print(f"[DEBUG]: The gs for \gamma = {gamma} is:", gs )
            if args.norm != "none":
                gs = normalize_scores(gs, args.norm)

            # add PPR scores row (sparse or dense)
            if args.ppr_scores.endswith(".npy"):
                if ppr_dense is not None:
                    r = ppr_user_map.get(str(u))
                    ppr_vec = np.zeros_like(gs)
                    if r is not None:
                        row = ppr_dense[r]
                        # copy all scores
                        for j, s in enumerate(row):
                            gi = col_remap[j]
                            if gi >= 0:
                                ppr_vec[gi] = s
                    fused = (1.0 - gamma) * gs + gamma * ppr_vec
            else:
                r = ppr_row_map.get(str(u))
                fused = gs.copy()
                if r is not None:
                    row = S.getrow(r)
                    if row.nnz > 0:
                        idx = row.indices
                        vals = row.data.astype(np.float32)
                        # remap indices to global item index space
                        gidx = col_remap[idx]
                        mask = gidx >= 0
                        gidx = gidx[mask]; vals = vals[mask]
                        if args.norm != "none":
                            # normalize only over non-zero entries
                            vmin, vmax = vals.min(), vals.max()
                            if args.norm == "minmax" and vmax > vmin + 1e-12:
                                vals = (vals - vmin) / (vmax - vmin)
                            elif args.norm == "z":
                                m, s = vals.mean(), vals.std()
                                vals = (vals - m) / (s + 1e-12)
                        fused[gidx] = (1.0 - gamma) * fused[gidx] + gamma * vals
                    else:
                        fused *= (1.0 - gamma)  # effectively γ*0

            if args.filter_seen:
                ban = seen.get(u, seen.get(str(u), set()))
                if ban:
                    ban_idx = [imap[str(i)] for i in ban if str(i) in imap]
                    if ban_idx:
                        fused[np.array(ban_idx, dtype=np.int64)] = -1e9

            topk = np.argpartition(-fused, args.k)[:args.k]
            topk = topk[np.argsort(-fused[topk])]
            rankings[u] = [i_idx[i] for i in topk]
            print(f"\r[INFO] Gamma {gamma:.2f} | Processed {len(rankings)}/{total_users} users", end='')

        rep = evaluate(rankings, gt, k_prec=args.kp, k_rec=args.kr, k_ndcg=args.kn)
        results.append((gamma, rep))

    rows = []
    for gamma, rep in results:
        m = rep.loc['mean'].to_dict()
        rows.append((gamma, m[f'P@{args.kp}'], m[f'R@{args.kr}'], m[f'NDCG@{args.kn}']))
    out_df = pd.DataFrame(rows, columns=['gamma', f'P@{args.kp}', f'R@{args.kr}', f'NDCG@{args.kn}'])
    print(out_df.to_string(index=False))
    best_row = out_df.sort_values(by=f'NDCG@{args.kn}', ascending=False).iloc[0]
    print(f"\nBest gamma by NDCG@{args.kn}: {best_row.gamma:.2f} | P={best_row[1]:.4f} R={best_row[2]:.4f} NDCG={best_row[3]:.4f}")

if __name__ == "__main__":
    main()