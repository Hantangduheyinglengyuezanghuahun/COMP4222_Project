# Hyperparameters: k the threhold for top-k pruning in two-hop edges

import argparse, os, json, pandas as pd, numpy as np, torch
from torch_geometric.data import HeteroData

def build_hetero(train_df, add_two_hop=False, two_hop_min_count=1):
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
    base['user'].x = torch.arange(len(umap))[:, None].float()
    base['item'].x = torch.arange(len(imap))[:, None].float()

    if not add_two_hop:
        return base, u_idx, i_idx, umap, imap

    # Build A^2 components (user-user & item-item) via sparse biadjacency
    try:
        import scipy.sparse as sp
        U, I = len(umap), len(imap)
        B = sp.coo_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(U, I))
        UU = (B @ B.T)  # user-user counts
        II = (B.T @ B)  # item-item counts
        UU.setdiag(0); II.setdiag(0)
        UU = UU.tocoo(); II = II.tocoo()
        uu_mask = UU.data >= two_hop_min_count
        ii_mask = II.data >= two_hop_min_count
        uu_rows = UU.row[uu_mask]; uu_cols = UU.col[uu_mask]
        ii_rows = II.row[ii_mask]; ii_cols = II.col[ii_mask]
    except Exception:
        # Fallback torch (CPU)
        U, I = len(umap), len(imap)
        vals = torch.ones(ei.size(1), dtype=torch.float32)  # CPU
        B = torch.sparse_coo_tensor(ei, vals, (U, I))
        UU = torch.sparse.mm(B, B.transpose(0, 1)).to_dense()
        II = torch.sparse.mm(B.transpose(0, 1), B).to_dense()
        UU.fill_diagonal_(0); II.fill_diagonal_(0)
        uu_rows, uu_cols = (UU >= two_hop_min_count).nonzero(as_tuple=True)
        ii_rows, ii_cols = (II >= two_hop_min_count).nonzero(as_tuple=True)

    enriched = base.clone()
    if len(uu_rows):
        weights = torch.as_tensor(UU.data, dtype=torch.float32)
        uu_idx = torch.tensor([UU.row, UU.col], dtype=torch.long)
        uu_idx = prune_topk(uu_idx, weights, k=50)
        enriched['user','similar','user'].edge_index = uu_idx
    if len(ii_rows):
        enriched['item','similar','item'].edge_index = torch.stack([
            torch.as_tensor(ii_rows, dtype=torch.long),
            torch.as_tensor(ii_cols, dtype=torch.long)
        ], dim=0)
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
    ap.add_argument("--dataset", choices=["video_games","electronics","all"], default="video_games")
    ap.add_argument("--category", default=None)
    ap.add_argument("--output-dir", default="data/loaded_data")
    ap.add_argument("--two-hop-min-count", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    def process_one(tag, filename):
        path = f"data/{filename}"
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if args.category:
            df = df[df['category'].str.lower() == args.category.lower()]
        prefix = tag  # Video_Games / Electronics

        df_path = _save_frame(df, args.output_dir, prefix)

        train_df = df[df.split == 'train'][['user_id','item_id']]

        base_graph, u_idx, i_idx, umap, imap = build_hetero(train_df, add_two_hop=False)
        enriched_graph, _, _, _, _ = build_hetero(train_df, add_two_hop=True,
                                                  two_hop_min_count=args.two_hop_min_count)

        meta = {
            "prefix": prefix,
            "num_users": len(u_idx),
            "num_items": len(i_idx),
            "paths": {}
        }

        base_path = os.path.join(args.output_dir, f"{prefix}_graphsage_A.pt")
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
        process_one("Video_Games", "interactions.Video_Games.split.csv")
    else:
        process_one("Electronics", "interactions.Electronics.split.csv")

if __name__ == "__main__":
    main()
