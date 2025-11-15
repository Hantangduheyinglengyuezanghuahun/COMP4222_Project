#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, time, numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch_geometric.data import HeteroData
from eval import build_ground_truth, evaluate

# -------------------------- Utils --------------------------
def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def load_preprocessed(dataset_dir, category=None, use_a2=False):
    """
    兼容已有预处理产物（若无则回退为就地构建）
    """
    prefix = (category if category else "all")
    fname = f"{prefix}_graphsage_{'A2_plus_A' if use_a2 else 'A'}.pt"
    path = os.path.join(dataset_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run dataset/pd.py first.")
    return torch.load(path)

# -------------------------- 行为边增强（LightGCN 邻接默认不并入 i2i） --------------------------
def augment_graph_with_behavioral_edges(
    data: HeteroData,
    df: pd.DataFrame,
    umap: dict,
    imap: dict,
    device,
    add_rating=True,
    add_history=True,
    rating_bins=((None, 3.0, 'r_low'), (3.0, 4.0, 'r_mid'), (4.0, None, 'r_high')),
    history_keep_last=10
):
    # rating 分桶：建立多关系（是否用于 LightGCN 邻接由 build_lightgcn_adj 的开关决定）
    if add_rating and ('rating' in df.columns):
        tr = df[df.split == 'train'][['user_id','item_id','rating']].dropna()
        for lo, hi, name in rating_bins:
            m = pd.Series(True, index=tr.index)
            if lo is not None: m &= (tr['rating'] >= lo)
            if hi is not None: m &= (tr['rating'] < hi)
            sub = tr[m]
            if len(sub) == 0: 
                continue
            su = sub['user_id'].map(umap).dropna().astype(int).to_numpy()
            si = sub['item_id'].map(imap).dropna().astype(int).to_numpy()
            if su.size == 0:
                continue
            ei = torch.tensor(np.vstack([su, si]), dtype=torch.long, device=device)
            data['user', name, 'item'].edge_index = ei
            data['item', f'rev_{name}', 'user'].edge_index = ei.flip(0)

    # 序列边：仅保留最近 history_keep_last 个的相邻对（默认不并入 LightGCN 邻接）
    if add_history and ('history' in df.columns):
        seqs = df.loc[df['history'].notna(), 'history'].astype(str)
        src, dst = [], []
        for h in seqs:
            toks = [t for t in h.split() if t in imap]
            if history_keep_last is not None and history_keep_last > 0:
                toks = toks[-history_keep_last:]
            if len(toks) < 2:
                continue
            ids = [imap[t] for t in toks]
            for a, b in zip(ids[:-1], ids[1:]):
                src.append(a); dst.append(b)
        if len(src) > 0:
            ei = torch.tensor([src, dst], dtype=torch.long, device=device)
            data['item', 'next', 'item'].edge_index = ei
            data['item', 'prev', 'item'].edge_index = ei.flip(0)

# -------------------------- LightGCN 模型 --------------------------
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, dim=64, layers=2, norm_emb=False):
        super().__init__()
        self.num_users  = num_users
        self.num_items  = num_items
        self.layers     = layers
        self.norm_emb   = norm_emb
        self.user_emb   = nn.Embedding(num_users, dim)
        self.item_emb   = nn.Embedding(num_items, dim)
        self.item_bias  = nn.Embedding(num_items, 1)  # learnable item bias（用于 logits & 重排）
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.zeros_(self.item_bias.weight)

    @torch.no_grad()
    def _split(self, E):
        return E[:self.num_users], E[self.num_users:]

    def propagate(self, A_norm: torch.Tensor):
        """
        LightGCN: E^(k+1) = Â E^(k), 输出为 0..K 的平均
        """
        E = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)  # [U+I, D]
        outs = [E]
        for _ in range(self.layers):
            E = torch.sparse.mm(A_norm, E)
            outs.append(E)
        E = torch.stack(outs, dim=0).mean(0)  # layer-wise average
        Ue, Ie = self._split(E)
        if self.norm_emb:
            Ue = F.normalize(Ue, dim=-1)
            Ie = F.normalize(Ie, dim=-1)
        return Ue, Ie

# -------------------------- 构建 LightGCN 归一化邻接 --------------------------
def build_lightgcn_adj(
    data: HeteroData,
    num_users: int,
    num_items: int,
    device,
    bipartite_only: bool = True,
    use_only_interacts: bool = True,
    include_item_item: bool = False
):
    """
    默认：严格二分 + 仅 interacts（推荐）。需要时可放开 include_item_item / use_only_interacts=False
    """
    U, I = num_users, num_items
    N = U + I
    rows, cols = [], []

    def add_u2i(ei):
        u = ei[0].to('cpu'); i = ei[1].to('cpu')
        rows.append(u);       cols.append(i + U)   # u -> i
        rows.append(i + U);   cols.append(u)       # i -> u

    def add_i2i(ei):
        i = ei[0].to('cpu') + U; j = ei[1].to('cpu') + U
        rows.append(i); cols.append(j)
        rows.append(j); cols.append(i)

    for (src, rel, dst) in data.edge_types:
        if src == 'user' and dst == 'item':
            if rel.startswith('rev_'):  # 反向关系不并入（我们已显式对称）
                continue
            if use_only_interacts and rel != 'interacts':
                continue
            ei = data[(src, rel, dst)].edge_index
            if ei.numel() > 0: add_u2i(ei)
        elif (not bipartite_only) and include_item_item and src == 'item' and dst == 'item':
            ei = data[(src, rel, dst)].edge_index
            if ei.numel() > 0: add_i2i(ei)
        else:
            # 其他一律忽略，确保严格二分
            pass

    if len(rows) == 0:
        idx = torch.empty((2,0), dtype=torch.long, device=device)
        val = torch.empty((0,), dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(idx, val, (N, N)).coalesce()

    rows = torch.cat(rows, dim=0)
    cols = torch.cat(cols, dim=0)
    idx  = torch.stack([rows, cols], dim=0)
    val  = torch.ones(idx.size(1), dtype=torch.float32)

    # CPU 构建，最后丢到 device
    A = torch.sparse_coo_tensor(idx, val, (N, N)).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1e-12)
    r, c = A.indices()
    v = A.values() / torch.sqrt(deg[r] * deg[c])
    A_norm = torch.sparse_coo_tensor(torch.stack([r, c], dim=0), v, (N, N)).coalesce().to(device)
    return A_norm

# -------------------------- 主流程 --------------------------
def main(args):
    torch.set_num_threads(max(1, args.threads))
    log("Start")

    # ---------- 读数据 ----------
    prefix = (args.category if args.category else "all")
    pq  = os.path.join(args.dataset_dir, f"{prefix}_interactions.parquet")
    csv = os.path.join(args.dataset_dir, f"{prefix}_interactions.csv")
    fallback_csv = r"C:\Users\Public\interactions.Video_Games.split.csv"

    t0 = time.time()
    if os.path.exists(pq):
        df = pd.read_parquet(pq); log(f"Loaded parquet: {pq} (rows={len(df)}) in {time.time()-t0:.2f}s")
    elif os.path.exists(csv):
        df = pd.read_csv(csv);    log(f"Loaded csv: {csv} (rows={len(df)}) in {time.time()-t0:.2f}s")
    elif os.path.exists(fallback_csv):
        df = pd.read_csv(fallback_csv); log(f"Loaded fallback csv: {fallback_csv} (rows={len(df)}) in {time.time()-t0:.2f}s")
    else:
        raise FileNotFoundError(f"Missing {pq} / {csv} and fallback {fallback_csv}. Provide one or run dataset/pd.py.")

    # 列名标准化
    col_aliases = {"user":"user_id","uid":"user_id","u":"user_id",
                   "item":"item_id","iid":"item_id","i":"item_id",
                   "label":"rating","score":"rating","ts":"timestamp","time":"timestamp"}
    rename_map = {src:dst for src,dst in col_aliases.items() if (src in df.columns and dst not in df.columns)}
    if rename_map: df = df.rename(columns=rename_map)
    if "split" not in df.columns: df["split"] = "train"
    needed = {"user_id","item_id","split"}
    missing = needed - set(df.columns)
    if missing: raise ValueError(f"Missing required columns: {missing}")

    # ---------- 构图 ----------
    try:
        loaded = load_preprocessed(args.dataset_dir, args.category, args.use_a2)
        data: HeteroData = loaded["data"]
        u_idx = pd.Index(loaded["u_idx"]); i_idx = pd.Index(loaded["i_idx"])
        umap  = dict(loaded["umap"]);      imap  = dict(loaded["imap"])
        log("Loaded preprocessed graph")
    except FileNotFoundError as e:
        log(f"WARN: {e} -> Build HeteroData on-the-fly")
        tr = df[df["split"]=="train"][["user_id","item_id"]].dropna().drop_duplicates()
        u_idx = pd.Index(df["user_id"].dropna().unique())
        i_idx = pd.Index(df["item_id"].dropna().unique())
        umap  = {u:k for k,u in enumerate(u_idx)}
        imap  = {i:k for k,i in enumerate(i_idx)}

        su = tr["user_id"].map(umap); si = tr["item_id"].map(imap)
        tr_ok = tr[su.notna() & si.notna()]
        su = tr_ok["user_id"].map(umap).astype(int).to_numpy()
        si = tr_ok["item_id"].map(imap).astype(int).to_numpy()
        edge = torch.tensor(np.vstack([su, si]), dtype=torch.long)

        data = HeteroData()
        data["user"].num_nodes = len(u_idx)
        data["item"].num_nodes = len(i_idx)
        if edge.numel() > 0:
            data["user","interacts","item"].edge_index = edge
            data["item","rev_interacts","user"].edge_index = edge.flip(0)
        else:
            data["user","interacts","item"].edge_index = torch.empty((2,0), dtype=torch.long)
            data["item","rev_interacts","user"].edge_index = torch.empty((2,0), dtype=torch.long)

    # 可选：增强行为边（默认开关来自命令行；但 LightGCN 邻接默认不并入 i2i）
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    data = data.to(device)
    if args.add_rating or args.add_history:
        augment_graph_with_behavioral_edges(
            data, df, umap, imap, device,
            add_rating=args.add_rating,
            add_history=args.add_history,
            history_keep_last=args.history_keep_last
        )

    # ---------- 训练样本 ----------
    train_cols = ["user_id","item_id"] + (["rating"] if "rating" in df.columns else [])
    pairs = df[df.split=="train"][train_cols].drop_duplicates()
    if (args.min_pos_rating is not None) and ("rating" in pairs.columns):
        pairs = pairs[pairs["rating"] >= args.min_pos_rating].drop(columns=["rating"])
    pu = pairs["user_id"].map(umap); pi = pairs["item_id"].map(imap)
    pairs = pairs[pu.notna() & pi.notna()]
    pos_u = torch.tensor(pairs["user_id"].map(umap).astype(int).to_numpy(), device=device, dtype=torch.long)
    pos_i = torch.tensor(pairs["item_id"].map(imap).astype(int).to_numpy(), device=device, dtype=torch.long)

    num_users = data['user'].num_nodes
    num_items = data['item'].num_nodes
    log(f"[INFO] Users={num_users}  Items={num_items}  Train pairs={pos_u.numel()}  Device={device}")

    # ---------- 构建 LightGCN 邻接（关键：默认严格二分 + 仅 interacts） ----------
    A_norm = build_lightgcn_adj(
        data, num_users, num_items, device,
        bipartite_only = True,
        use_only_interacts = True,
        include_item_item = False  # 保持 False，避免扩散困于 item→item
    )
    log(f"A_norm nnz={A_norm._nnz()}")

    # ---------- 模型 & 优化 ----------
    model = LightGCN(num_users, num_items, dim=args.dim, layers=args.layers, norm_emb=args.norm_emb).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # ---------- 采样权重（时间） ----------
    if (args.recency_alpha > 0.0) and ("timestamp" in df.columns):
        tmap = df[df.split=="train"][["user_id","item_id","timestamp"]].drop_duplicates()
        tmap["uid"] = tmap["user_id"].map(umap); tmap["iid"] = tmap["item_id"].map(imap)
        tmap = tmap.dropna(subset=["uid","iid"])
        if not tmap.empty:
            tmap[["uid","iid"]] = tmap[["uid","iid"]].astype(int)
            key_uid = pairs["user_id"].map(umap).astype("Int64")
            key_iid = pairs["item_id"].map(imap).astype("Int64")
            key = pd.MultiIndex.from_arrays([key_uid, key_iid])
            tser = pd.Series(tmap["timestamp"].to_numpy(),
                             index=pd.MultiIndex.from_arrays([tmap["uid"], tmap["iid"]]))
            ts_vals = pd.Series(tser.reindex(key).fillna(tser.min()).to_numpy()).to_numpy()
            ts = torch.tensor(ts_vals, device=device, dtype=torch.float)
            ts = (ts - ts.min()) / (ts.max() - ts.min() + 1e-9)
            sample_w = torch.exp(ts * args.recency_alpha)
            sample_w = sample_w / sample_w.sum()
        else:
            sample_w = None
    else:
        sample_w = None

    # ---------- 人气（训练：负采样；推理：重排） ----------
    train_df = df[df.split=="train"][["user_id","item_id"]]
    vc = train_df["item_id"].map(imap).dropna().astype(int).value_counts()
    pop_counts = torch.zeros(num_items, dtype=torch.float, device=device)
    if len(vc) > 0:
        pop_counts[torch.tensor(vc.index.values, device=device)] = torch.tensor(vc.values, dtype=torch.float, device=device)
    # 采样分布 p(i) ∝ (count+1)^alpha
    alpha = args.pop_ns_alpha
    pop_p = (pop_counts + 1.0) ** alpha
    pop_p = pop_p / pop_p.sum()
    # 推理的人气分：log(1+count)
    item_pop_score = torch.log1p(pop_counts)

    def sample_negs_with_pop(batch_size: int, num_negs: int):
        idx = torch.multinomial(pop_p, num_samples=batch_size * num_negs, replacement=True)
        return idx.view(batch_size, num_negs)

    # ---------- 训练 ----------
    bsz, negs = args.batch, max(1, args.negs)
    log(f"Train cfg: epochs={args.epochs} | layers={args.layers} | dim={args.dim} | "
        f"loss={args.loss} | inbatch={args.use_inbatch} | norm={args.norm_emb} | "
        f"tau={args.temperature} | batch={bsz} | negs={negs} | wd={args.wd} | bpr_reg={args.bpr_reg}")

    model.train()
    for epoch in range(1, args.epochs+1):
        if pos_u.numel() == 0:
            log("WARN: no train pairs; skip"); break

        # 抽样顺序
        if sample_w is None:
            order = torch.randperm(pos_u.size(0), device=device)
        else:
            order = torch.multinomial(sample_w, num_samples=pos_u.size(0), replacement=True)

        total = 0.0
        et0 = time.time()
        for s in range(0, order.numel(), bsz):
            idx = order[s:s+bsz]
            u = pos_u[idx]; i = pos_i[idx]

            # 取出当前 batch 的嵌入
            Ue, Ie = model.propagate(A_norm)   # [U,D], [I,D]
            ue = Ue[u]                         # [B,D]
            pe = Ie[i]                         # [B,D]

            if args.loss == "inbatch":
                # In-batch InfoNCE（每行对应一个 user，对角为正）
                logits = (ue @ pe.T) / max(1e-8, args.temperature)  # [B,B]
                ib = model.item_bias.weight[i].squeeze(-1)          # [B]
                logits = logits + ib.unsqueeze(0)
                target = torch.arange(ue.size(0), device=device)
                loss = F.cross_entropy(logits, target)

                with torch.no_grad():
                    top1 = (logits.argmax(dim=1) == target).float().mean().item()
                    auc = float('nan')

            else:
                # ===== BPR pairwise =====
                ni = sample_negs_with_pop(u.size(0), 1).squeeze(1)  # [B]
                ne = Ie[ni]                                         # [B,D]

                pos = (ue * pe).sum(-1)     # [B]
                neg = (ue * ne).sum(-1)     # [B]
                bpr = -torch.log(torch.sigmoid(pos - neg) + 1e-12).mean()

                # L2 正则（只正则 batch 涉及的行）
                if args.bpr_reg > 0:
                    reg = (ue.pow(2).sum(dim=1) + pe.pow(2).sum(dim=1) + ne.pow(2).sum(dim=1)).mean()
                    loss = bpr + args.bpr_reg * reg
                else:
                    loss = bpr

                with torch.no_grad():
                    top1 = float('nan')
                    auc = (pos > neg).float().mean().item()

            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss)

            # 打点日志
            step = (s // bsz) + 1
            if args.log_interval > 0 and (step % args.log_interval == 0):
                if args.loss == "inbatch":
                    log(f"epoch {epoch:03d} step {step:05d} loss={total/step:.4f} inbatch@top1={top1:.3f}")
                else:
                    log(f"epoch {epoch:03d} step {step:05d} loss={total/step:.4f} bpr@auc={auc:.3f}")

        if args.loss == "inbatch":
            log(f"[epoch {epoch:03d}] loss_sum={total:.3f} time={time.time()-et0:.2f}s")
        else:
            log(f"[epoch {epoch:03d}] loss_sum={total:.3f} time={time.time()-et0:.2f}s (BPR)")

    # ---------- 推理 ----------
    log("Start inference")
    model.eval()
    with torch.no_grad():
        Ue, Ie = model.propagate(A_norm)
        ibias = model.item_bias.weight.squeeze(-1)

    # 已见过滤
    seen = df[df.split=="train"][["user_id","item_id"]].groupby("user_id")["item_id"].apply(set).to_dict()

    # 排序
    test_users = df[df.split=="test"]["user_id"].unique()
    rankings = {}
    for u in test_users:
        if u not in umap: continue
        uid = umap[u]
        base = (Ue[uid:uid+1] @ Ie.T).squeeze(0)              # 模型分
        if args.lambda_pop > 0:
            base = base + args.lambda_pop * item_pop_score    # 人气混合
        base = base + ibias                                   # 学到的 item 偏置

        ban = seen.get(u, set())
        if ban:
            ban_idx = [imap[i] for i in ban if i in imap]
            if len(ban_idx) > 0:
                base[torch.tensor(ban_idx, device=base.device, dtype=torch.long)] = -1e9

        k = min(args.k, Ie.size(0))
        if k > 0:
            topk = torch.topk(base, k=k).indices.tolist()
            item_ids = [i_idx[i] for i in topk]
            rankings[u] = item_ids

    # 评测
    log("Evaluate")
    gt = build_ground_truth(df, phase="test")
    rep = evaluate(rankings, gt, k_prec=args.kp, k_rec=args.kr, k_ndcg=args.kn)
    print("== LightGCN (bipartite-only) + BPR/InfoNCE + Pop (mean/std) ==")
    print(rep)

    # 健康检查
    test_items = set(df[df.split=="test"]["item_id"].unique())
    covered = (sum(1 for t in test_items if t in imap) / max(1, len(test_items))) if len(test_items)>0 else 0.0
    print("Test item coverage:", round(covered, 4))
    if pos_u.numel() > 0 and Ie.size(0) > 0:
        with torch.no_grad():
            idx = torch.randint(0, pos_u.numel(), (min(8192, pos_u.numel()),), device=Ue.device)
            posm = (Ue[pos_u[idx]] * Ie[pos_i[idx]]).sum(-1).mean().item()
            negm = (Ue[pos_u[idx]].unsqueeze(1) * Ie[torch.randint(0, Ie.size(0), (idx.numel(), 64), device=Ue.device)]).sum(-1).mean().item()
            print(f"Mean pos logit {posm:.3f} vs neg {negm:.3f}")

# -------------------------- CLI --------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None)
    ap.add_argument("--dataset-dir", default="data/loaded_data")
    ap.add_argument("--use-a2", action="store_true")

    # 模型
    ap.add_argument("--dim",       type=int, default=128, help="Embedding dim")
    ap.add_argument("--layers",    type=int, default=3,   help="LightGCN propagation layers K")
    ap.add_argument("--norm-emb",  action="store_true",   help="L2-normalize embeddings after propagation")

    # 训练
    ap.add_argument("--epochs",    type=int, default=40)
    ap.add_argument("--batch",     type=int, default=4096)
    ap.add_argument("--negs",      type=int, default=1)     # BPR 通常 1 个负样本足够
    ap.add_argument("--lr",        type=float, default=5e-4)
    ap.add_argument("--wd",        type=float, default=0.0) # 用 bpr_reg 做正则，避免双重正则
    ap.add_argument("--cpu",       action="store_true")
    ap.add_argument("--threads",   type=int, default=4)

    # 损失 / 对比学习 / BPR
    ap.add_argument("--loss", choices=["inbatch", "bpr"], default="bpr",
                    help="Training objective: in-batch InfoNCE or BPR pairwise.")
    ap.add_argument("--use-inbatch",  action="store_true", help="(兼容开关) 若 loss=inbatch 则生效")
    ap.add_argument("--temperature",  type=float, default=0.07, help="Softmax temperature for InfoNCE")
    ap.add_argument("--bpr-reg",      type=float, default=1e-4, help="L2 reg coeff for BPR on used rows")

    # 负采样（不开 in-batch 时生效，BPR 也使用）
    ap.add_argument("--pop-ns-alpha", type=float, default=0.75, help="Popularity^alpha for negative sampling")

    # 推荐与评测
    ap.add_argument("--k",  type=int, default=200)
    ap.add_argument("--kp", type=int, default=10)
    ap.add_argument("--kr", type=int, default=20)
    ap.add_argument("--kn", type=int, default=10)

    # 行为边增强（仅用于构图存储；LightGCN 邻接默认不并入 i2i，见 build_lightgcn_adj）
    ap.add_argument("--add-rating",      action="store_true")
    ap.add_argument("--add-history",     action="store_true")
    ap.add_argument("--history-keep-last", type=int, default=10)
    ap.add_argument("--min-pos-rating",  type=float, default=3.5)
    ap.add_argument("--recency-alpha",   type=float, default=0.0)

    # 推理重排
    ap.add_argument("--lambda-pop",  type=float, default=0.15, help="Mix weight for popularity at inference")

    # 日志
    ap.add_argument("--log-interval", type=int, default=20)

    args = ap.parse_args()
    if args.min_pos_rating is not None and args.min_pos_rating < 0:
        args.min_pos_rating = None
    print("The args:", args)
    main(args)
