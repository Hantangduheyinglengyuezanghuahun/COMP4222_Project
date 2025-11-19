# Hyperparameters: k the threhold for top-k pruning in two-hop edges

import argparse, os, json, pandas as pd, numpy as np, torch
from torch_geometric.data import HeteroData

def _prune_csr_topk_rows(mat_csr, k):
    # mat_csr: scipy.sparse.csr_matrix
    import numpy as np
    mat = mat_csr.tocsr()
    indptr = mat.indptr
    indices = mat.indices
    data = mat.data
    new_indices = []
    new_data = []
    new_indptr = [0]
    for r in range(mat.shape[0]):
        start, end = indptr[r], indptr[r+1]
        if end > start:
            cols = indices[start:end]
            vals = data[start:end]
            order = np.argsort(vals)[::-1]
            take = order[:k]
            new_indices.extend(cols[take])
            new_data.extend(vals[take])
        new_indptr.append(len(new_indices))
    import scipy.sparse as sp
    return sp.csr_matrix((np.asarray(new_data), np.asarray(new_indices), np.asarray(new_indptr)), shape=mat.shape)

def build_hetero(train_df, add_two_hop=False, two_hop_min_count=1, two_hop_topk=50, add_uu=True, add_ii=True, hop_order=2):
    u_idx = pd.Index(train_df['user_id'].unique())
    i_idx = pd.Index(train_df['item_id'].unique())
    umap = {u: k for k, u in enumerate(u_idx)}
    imap = {i: k for k, i in enumerate(i_idx)}
    src = train_df['user_id'].map(umap).to_numpy()
    dst = train_df['item_id'].map(imap).to_numpy()

    ei = torch.tensor(np.vstack([src, dst]), dtype=torch.long)  # [2, E]
    base = HeteroData()
    base['user'].num_nodes = len(umap)
    base['item'].num_nodes = len(imap)
    base['user','interacts','item'].edge_index = ei
    base['item','rev_interacts','user'].edge_index = ei.flip(0)

    # Add rating-derived node features (precomputed, aligned to u_idx/i_idx)
    if 'rating' in train_df.columns:
        # Per-user/item mean rating and count from TRAIN interactions only
        u_stats = train_df.groupby('user_id')['rating'].agg(['mean','count'])
        i_stats = train_df.groupby('item_id')['rating'].agg(['mean','count'])

        u_mean = u_stats['mean'].reindex(u_idx, fill_value=u_stats['mean'].mean() if len(u_stats) else 0.0).to_numpy(np.float32)
        u_cnt  = u_stats['count'].reindex(u_idx, fill_value=0).to_numpy(np.float32)
        i_mean = i_stats['mean'].reindex(i_idx, fill_value=i_stats['mean'].mean() if len(i_stats) else 0.0).to_numpy(np.float32)
        i_cnt  = i_stats['count'].reindex(i_idx, fill_value=0).to_numpy(np.float32)

        # Z-score means, log1p counts
        u_mean_z = (u_mean - (u_mean.mean() if u_mean.size else 0.0)) / (u_mean.std() + 1e-8)
        i_mean_z = (i_mean - (i_mean.mean() if i_mean.size else 0.0)) / (i_mean.std() + 1e-8)
        u_cnt_l  = np.log1p(u_cnt)
        i_cnt_l  = np.log1p(i_cnt)

        base['user'].x = torch.from_numpy(np.stack([u_mean_z, u_cnt_l], axis=1)).float()
        base['item'].x = torch.from_numpy(np.stack([i_mean_z, i_cnt_l], axis=1)).float()
    else:
        # Fallback identity features if no ratings column exists
        base['user'].x = torch.arange(len(umap))[:, None].float()
        base['item'].x = torch.arange(len(imap))[:, None].float()

    if not add_two_hop:
        return base, u_idx, i_idx, umap, imap

    try:
        import scipy.sparse as sp
        U, I = len(umap), len(imap)
        B = sp.coo_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(U, I)).tocsr()

        # 2-hop co-occurrence graphs
        UU2 = (B @ B.T).tocsr()
        II2 = (B.T @ B).tocsr()
        UU2.setdiag(0); II2.setdiag(0)
        UU2.eliminate_zeros(); II2.eliminate_zeros()

        if two_hop_min_count > 1:
            UU2.data[UU2.data < two_hop_min_count] = 0
            II2.data[II2.data < two_hop_min_count] = 0
            UU2.eliminate_zeros(); II2.eliminate_zeros()

        # Prune 2-hop to top-k per row first to cap size
        UU2_k = _prune_csr_topk_rows(UU2, two_hop_topk) if add_uu else None
        II2_k = _prune_csr_topk_rows(II2, two_hop_topk) if add_ii else None

        if hop_order == 4:
            # Square the pruned 2-hop graph to get 4-hop, then prune again
            if add_uu and UU2_k is not None:
                UU4 = (UU2_k @ UU2_k).tocsr()
                UU4.setdiag(0); UU4.eliminate_zeros()
                UUk = _prune_csr_topk_rows(UU4, two_hop_topk)
            else:
                UUk = None
            if add_ii and II2_k is not None:
                II4 = (II2_k @ II2_k).tocsr()
                II4.setdiag(0); II4.eliminate_zeros()
                IIk = _prune_csr_topk_rows(II4, two_hop_topk)
            else:
                IIk = None
        else:
            # Use 2-hop directly
            UUk = UU2_k if add_uu else None
            IIk = II2_k if add_ii else None

        enriched = base.clone()
        if add_uu and UUk is not None and UUk.nnz > 0:
            uu_rows, uu_cols = UUk.nonzero()
            uu_w = UUk.data.astype(np.float32)
            enriched['user','similar','user'].edge_index  = torch.tensor([uu_rows, uu_cols], dtype=torch.long)
            enriched['user','similar','user'].edge_weight = torch.from_numpy(uu_w)
        if add_ii and IIk is not None and IIk.nnz > 0:
            ii_rows, ii_cols = IIk.nonzero()
            ii_w = IIk.data.astype(np.float32)
            enriched['item','similar','item'].edge_index  = torch.tensor([ii_rows, ii_cols], dtype=torch.long)
            enriched['item','similar','item'].edge_weight = torch.from_numpy(ii_w)

        return enriched, u_idx, i_idx, umap, imap
    except Exception:
        # Fallback torch (CPU), set weights=1 (no Jaccard)
        U, I = len(umap), len(imap)
        vals = torch.ones(ei.size(1), dtype=torch.float32)  # CPU
        B = torch.sparse_coo_tensor(ei, vals, (U, I))
        UU = torch.sparse.mm(B, B.transpose(0, 1)).to_dense()
        II = torch.sparse.mm(B.transpose(0, 1), B).to_dense()
        UU.fill_diagonal_(0); II.fill_diagonal_(0)
        uu_rows, uu_cols = (UU >= two_hop_min_count).nonzero(as_tuple=True)
        ii_rows, ii_cols = (II >= two_hop_min_count).nonzero(as_tuple=True)
        uu_rows = uu_rows.numpy(); uu_cols = uu_cols.numpy()
        ii_rows = ii_rows.numpy(); ii_cols = ii_cols.numpy()
        uu_jacc = np.ones_like(uu_rows, dtype=np.float32)
        ii_jacc = np.ones_like(ii_rows, dtype=np.float32)

    enriched = base.clone()
    # Top-k per source node (keeps graph sparse and bounded)
    if add_uu and len(uu_rows):
        ei_u = torch.tensor([uu_rows, uu_cols], dtype=torch.long)
        w_u = torch.from_numpy(uu_jacc)
        ei_u_k, w_u_k = prune_topk_with_weights(ei_u, w_u, k=two_hop_topk)
        enriched['user','similar','user'].edge_index  = ei_u_k
        enriched['user','similar','user'].edge_weight = w_u_k

    if add_ii and len(ii_rows):
        ei_i = torch.tensor([ii_rows, ii_cols], dtype=torch.long)
        w_i = torch.from_numpy(ii_jacc)
        ei_i_k, w_i_k = prune_topk_with_weights(ei_i, w_i, k=two_hop_topk)
        enriched['item','similar','item'].edge_index  = ei_i_k
        enriched['item','similar','item'].edge_weight = w_i_k

    return enriched, u_idx, i_idx, umap, imap

def prune_topk(edge_index, weights, k):
    # edge_index: [2,E], weights: [E]
    import numpy as np, torch
    src, dst = edge_index
    df = {}
    for s, d, w in zip(src.tolist(), dst.tolist(), weights.tolist()):
        df.setdefault(s, []).append((d, w))
    keep_src, keep_dst = [], []
    for s, lst in df.items():
        lst.sort(key=lambda x: x[1], reverse=True)
        for d, w in lst[:k]:
            keep_src.append(s); keep_dst.append(d)
    return torch.tensor([keep_src, keep_dst], dtype=torch.long)

def prune_topk_with_weights(edge_index, weights, k):
    # edge_index: [2,E] (CPU), weights: [E] (CPU)
    src, dst = edge_index
    buckets = {}
    for s, d, w in zip(src.tolist(), dst.tolist(), weights.tolist()):
        lst = buckets.setdefault(s, [])
        lst.append((d, w))
    keep_s, keep_d, keep_w = [], [], []
    for s, lst in buckets.items():
        lst.sort(key=lambda x: x[1], reverse=True)
        for d, w in lst[:k]:
            keep_s.append(s); keep_d.append(d); keep_w.append(w)
    return (torch.tensor([keep_s, keep_d], dtype=torch.long),
            torch.tensor(keep_w, dtype=torch.float32))

def _save_frame(df, out_dir, prefix):
    # Prefer Parquet; fall back to CSV if engine missing
    pq = os.path.join(out_dir, f"{prefix}_interactions.parquet")
    try:
        df.to_parquet(pq, index=False)
        return pq
    except Exception:
        csv = os.path.join(out_dir, f"{prefix}_interactions.csv")
        df.to_csv(csv, index=False)
        return csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["video_games","electronics","books", "all"], default="video_games")
    ap.add_argument("--category", default=None)
    ap.add_argument("--output-dir", default="data/loaded_data")
    ap.add_argument("--two-hop-min-count", type=int, default=1)
    ap.add_argument("--two-hop-topk", type=int, default=300, help="Keep top-k 2/4-hop neighbors per source node")
    ap.add_argument("--two-hop-relations", choices=["both","ii","uu"], default="ii",
                    help="Which k-hop relations to include")
    ap.add_argument("--hop-order", type=int, choices=[2,4], default=2, help="Use 2-hop (A^2) or 4-hop (A^4)")
    ap.add_argument("--disable-two-hop", action="store_true",
                    help="If set, do not compute 2-hop edges; A2_plus_A will be identical to A.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    def process_one(tag, filename):
        path = f"data/{filename}"
        print(path)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if args.category:
            df = df[df['category'].str.lower() == args.category.lower()]
        prefix = tag  # Video_Games / Electronics

        df_path = _save_frame(df, args.output_dir, prefix)

        train_df = df[df['split'] == 'train'][['user_id','item_id']]

        add_uu = args.two_hop_relations in ("both","uu")
        add_ii = args.two_hop_relations in ("both","ii")

        base_graph, u_idx, i_idx, umap, imap = build_hetero(train_df, add_two_hop=False)
        if args.disable_two_hop:
            enriched_graph = base_graph  # skip heavy 2-hop build
        else:
            enriched_graph, _, _, _, _ = build_hetero(
                train_df,
                add_two_hop=True,
                two_hop_min_count=args.two_hop_min_count,
                two_hop_topk=args.two_hop_topk,
                add_uu=add_uu,
                add_ii=add_ii,
                hop_order=args.hop_order
            )

        meta = {
            "prefix": prefix,
            "num_users": len(u_idx),
            "num_items": len(i_idx),
            "paths": {}
        }

        base_path = os.path.join(args.output_dir, f"{prefix}_graphsage_A.pt")
        print("[DEBUG] Saving base graph to: ", base_path)
        torch.save({"data": base_graph, "u_idx": list(u_idx), "i_idx": list(i_idx),
                    "umap": umap, "imap": imap}, base_path)
        meta["paths"]["A"] = base_path


        a2_path = os.path.join(args.output_dir, f"{prefix}_graphsage_A2_plus_A.pt")
        torch.save({"data": enriched_graph, "u_idx": list(u_idx), "i_idx": list(i_idx),
                    "umap": umap, "imap": imap}, a2_path)
        meta["paths"]["A2_plus_A"] = a2_path

        meta_path = os.path.join(args.output_dir, f"{prefix}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[{prefix}] saved dataframe: {df_path}")
        print(f"[{prefix}] saved A: {base_path}")
        print(f"[{prefix}] saved A^2 + A: {a2_path}")
        print(f"[{prefix}] meta: {meta_path}")

    if args.dataset == "all":
        process_one("Video_Games", "interactions.Video_Games.split.csv")
        process_one("Electronics", "interactions.Electronics.split.csv")
    elif args.dataset == "video_games":
        print("[DEBUG]: args.dataset is: ", args.dataset)
        process_one("Video_Games", "interactions.Video_Games.split.csv")
    elif args.dataset == "books":
        process_one("Books", "interactions.Books.split.csv")
    else:
        process_one("Electronics", "interactions.Electronics.split.csv")

if __name__ == "__main__":
    main()
