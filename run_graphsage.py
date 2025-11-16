import argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from eval import build_ground_truth, evaluate
import os
import torch.nn.functional as F
import datetime

def load_preprocessed(dataset_dir, category=None, use_a2=False):
    prefix = (category if category else "all")
    fname = f"{prefix}_graphsage_{'A2_plus_A' if use_a2 else 'A'}.pt"
    path = os.path.join(dataset_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run dataset/pd.py first.")
    return torch.load(path)

def load_node2vec(dataset_dir, category, dim):
    path = os.path.join(dataset_dir, f"{category}_node2vec_{dim}.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu")



def load_text_features(text_emb_path: str, text_index_path: str, i_idx: pd.Index, device: torch.device):
    # Load precomputed item text embeddings and align to i_idx ordering
    if not (os.path.exists(text_emb_path) and os.path.exists(text_index_path)):
        raise FileNotFoundError(f"Missing text features: {text_emb_path} or {text_index_path}")
    emb = np.load(text_emb_path)  # shape [N_items_text, D_text], float32
    idx_df = pd.read_csv(text_index_path)  # columns: item_id, idx
    if "item_id" not in idx_df.columns or "idx" not in idx_df.columns:
        raise ValueError("item_text_index.csv must have columns: item_id, idx")
    id2row = dict(zip(idx_df["item_id"].astype(str), idx_df["idx"].astype(int)))
    D = emb.shape[1]
    # Build aligned matrix [I, D]
    aligned = np.zeros((len(i_idx), D), dtype=np.float32)
    missing = 0
    for j, item in enumerate(i_idx):
        r = id2row.get(str(item))
        if r is None or r < 0 or r >= emb.shape[0]:
            missing += 1
            continue
        aligned[j] = emb[r]
    if missing:
        print(f"[TEXT] Missing {missing}/{len(i_idx)} items in text features; filled with zeros.")
    print(f"[DEBUG] Loaded text features with shape: {aligned.shape}")
    return torch.from_numpy(aligned).to(device=device, dtype=torch.float32)

class SAGEEncoder(nn.Module):
    def __init__(self, hidden=64, out=64, drop_out=0.2):
        super().__init__()
        self.user_emb = nn.Embedding(1, hidden)  # 仅占位；真正特征在 to_hetero 时替换
        self.item_emb = nn.Embedding(1, hidden)
        self.conv1 = SAGEConv((-1, -1), hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.dropout = nn.Dropout(p=drop_out)
        self.conv2 = SAGEConv((-1, -1), out)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.mlp(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        # x = F.normalize(x, p=2.0, dim=-1)
        return x

class ItemRatingFusion(nn.Module):
    def __init__(self, n2v_dim=64, rating_dim=2, out_dim=64, hid_dim=None):
        super().__init__()
        # Avoid bottleneck: default hidden = max(n2v_dim+rating_dim, 2*out_dim)
        hid = hid_dim if hid_dim is not None else max(n2v_dim + rating_dim, out_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(n2v_dim + rating_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, out_dim)
        )
    def forward(self, item_n2v, rating_feat):
        return self.net(torch.cat([item_n2v, rating_feat], dim=1))

class UserCommentFusion(nn.Module):
    def __init__(self, base_dim, comment_dim, out_dim, hid_dim=None):
        super().__init__()
        # Avoid bottleneck: hidden >= base_dim+comment_dim
        hid = hid_dim if hid_dim is not None else max(base_dim + comment_dim, out_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(base_dim + comment_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, out_dim)
        )
    def forward(self, base_user_feat, user_comment_feat):
        return self.net(torch.cat([base_user_feat, user_comment_feat], dim=1))

class ItemCommentFusion(nn.Module):
    def __init__(self, base_dim, comment_dim, out_dim, hid_dim=None):
        super().__init__()
        # Avoid bottleneck: hidden >= base_dim+comment_dim
        hid = hid_dim if hid_dim is not None else max(base_dim + comment_dim, out_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(base_dim + comment_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, out_dim)
        )
    def forward(self, base_item_feat, item_comment_feat):
        return self.net(torch.cat([base_item_feat, item_comment_feat], dim=1))

def negative_sampling(pos_src, num_items, num_negs=1):
    neg_dst = torch.randint(0, num_items, (pos_src.numel()*num_negs,), device=pos_src.device)
    return neg_dst

def main(args):
    prefix = (args.category if args.category else "all")
    print("[DEBUG]: The prefix is:", prefix)
    pq = os.path.join(args.dataset_dir, f"{prefix}_interactions.parquet")
    print("[DEBUG]: The parquet path is:", pq)
    print("[DEBUG]: The path exists:", os.path.exists(pq))
    csv = os.path.join(args.dataset_dir, f"{prefix}_interactions.csv")
    if os.path.exists(pq):
        df = pd.read_parquet(pq)
    elif os.path.exists(csv):
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError(f"Missing {pq} / {csv}. Run dataset/pd.py.")
    loaded = load_preprocessed(args.dataset_dir, args.category, args.use_a2)
    data = loaded['data']
    u_idx = pd.Index(loaded['u_idx'])
    i_idx = pd.Index(loaded['i_idx'])
    umap = loaded['umap']; imap = loaded['imap']
    train = df[df.split=='train'][['user_id','item_id']]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # Load Node2Vec features with decoupled dim
    n2v = load_node2vec(args.dataset_dir, (args.category if args.category else "all"), args.n2v_dim)
    if n2v is None:
        raise FileNotFoundError(f"Node2Vec features not found. Run dataset/node2vec.py first.")
    user_x = n2v['user_emb'].to(device=device, dtype=torch.float32)  # [U, n2v_dim]
    item_x = n2v['item_emb'].to(device=device, dtype=torch.float32)  # [I, n2v_dim]
    data['user'].x = user_x
    data['item'].x = item_x
    print(f"Loaded Node2Vec features: user_dim={data['user'].x.size(-1)}, item_dim={data['item'].x.size(-1)}")

    # Decide fusion dims and avoid squeezing in SAGE
    fusion_out = args.fusion_out if args.fusion_out is not None else int(data['item'].x.size(1))
    fusion_hid = args.fusion_hid  # can be None -> auto from inputs
    if args.expand_hidden and args.hidden < fusion_out:
        print(f"[DIM] expand_hidden: bump hidden {args.hidden} -> {fusion_out}")
        args.hidden = fusion_out

    # Build per-item rating features (mean_z, log_count) aligned to i_idx (optional)
    item_rating_feat = None
    if args.use_rating and 'rating' in df.columns:
        train_rating = df[df.split == 'train'][['item_id', 'rating']].copy()
        istats = train_rating.groupby('item_id')['rating'].agg(['mean', 'count'])
        global_mean = float(istats['mean'].mean()) if len(istats) else 0.0
        i_mean = istats['mean'].reindex(pd.Index(loaded['i_idx']), fill_value=global_mean).to_numpy(dtype=np.float32)
        i_cnt  = istats['count'].reindex(pd.Index(loaded['i_idx']), fill_value=0).to_numpy(dtype=np.float32)
        mean_z = (i_mean - float(i_mean.mean())) / (float(i_mean.std()) + 1e-8)
        cnt_l  = np.log1p(i_cnt)
        item_rating_feat = torch.from_numpy(np.stack([mean_z, cnt_l], axis=1)).to(device=device, dtype=torch.float32)
        print("[DEBUG] item_rating_feat:", tuple(item_rating_feat.shape))
    elif args.use_rating:
        print("[WARN] rating column not found; skipping rating fusion.")
    else:
        print("[INFO] --use-rating disabled; using pure Node2Vec item features.")

    # Initialize fusion modules
    rating_fuse = None
    user_comment_fuse = None
    item_comment_fuse = None
    user_comment_feat = None
    item_comment_feat = None

    # Apply rating fusion first (if enabled)
    if args.use_rating and item_rating_feat is not None:
        rating_fuse = ItemRatingFusion(n2v_dim=item_x.size(1),
                                       rating_dim=item_rating_feat.size(1),
                                       out_dim=fusion_out,
                                       hid_dim=fusion_hid).to(device)
        with torch.no_grad():
            data['item'].x = rating_fuse(data['item'].x, item_rating_feat)
        print(f"[RATING] Fused item features: dim={data['item'].x.size(-1)}")
    else:
        print(f"[RATING] Disabled; item dim={data['item'].x.size(-1)}")

    if args.use_comment_user or args.use_comment_item:
        if not (os.path.exists(args.comment_emb) and os.path.exists(args.comment_index)):
            print(f"[COMMENT][WARN] Missing {args.comment_emb} or {args.comment_index}; skipping comment fusion.")
        else:
            try:
                c_emb = np.load(args.comment_emb)  # [N_comments, Dc]
                c_idx = pd.read_csv(args.comment_index)
                if not {"row","user_id","item_id"}.issubset(c_idx.columns):
                    raise ValueError("comment_text_index.csv must have columns: row,user_id,item_id")
                Dc = c_emb.shape[1]
                if args.use_comment_user:
                    user_comment_sum = torch.zeros((len(u_idx), Dc), dtype=torch.float32)
                    user_comment_cnt = torch.zeros((len(u_idx),), dtype=torch.float32)
                if args.use_comment_item:
                    item_comment_sum = torch.zeros((len(i_idx), Dc), dtype=torch.float32)
                    item_comment_cnt = torch.zeros((len(i_idx),), dtype=torch.float32)
                for r in c_idx.itertuples(index=False):
                    crow = r.row
                    if crow < 0 or crow >= c_emb.shape[0]:
                        continue
                    vec = torch.from_numpy(c_emb[crow]).float()
                    uid_raw = str(r.user_id)
                    iid_raw = str(r.item_id)
                    uind = umap.get(uid_raw)
                    iind = imap.get(iid_raw)
                    if args.use_comment_user and uind is not None:
                        user_comment_sum[uind] += vec; user_comment_cnt[uind] += 1
                    if args.use_comment_item and iind is not None:
                        item_comment_sum[iind] += vec; item_comment_cnt[iind] += 1
                if args.use_comment_user:
                    user_comment_cnt[user_comment_cnt == 0] = 1
                    user_comment_feat = (user_comment_sum / user_comment_cnt.unsqueeze(1)).to(device)
                    print(f"[COMMENT][USER] shape={tuple(user_comment_feat.shape)}")
                if args.use_comment_item:
                    item_comment_cnt[item_comment_cnt == 0] = 1
                    item_comment_feat = (item_comment_sum / item_comment_cnt.unsqueeze(1)).to(device)
                    print(f"[COMMENT][ITEM] shape={tuple(item_comment_feat.shape)}")
            except Exception as e:
                print(f"[COMMENT][WARN] Failed aggregation: {e}")
                user_comment_feat = None; item_comment_feat = None
    else:
        print("[COMMENT] Both user/item comment fusion disabled.")

    # ================= Apply user comment fusion =================
    if args.use_comment_user and user_comment_feat is not None:
        base_user_dim = data['user'].x.size(1)
        user_comment_fuse = UserCommentFusion(base_dim=base_user_dim,
                                              comment_dim=user_comment_feat.size(1),
                                              out_dim=fusion_out,
                                              hid_dim=fusion_hid).to(device)
        with torch.no_grad():
            data['user'].x = user_comment_fuse(data['user'].x, user_comment_feat)
        print(f"[COMMENT][USER] Fused user dim={data['user'].x.size(-1)}")
    elif args.use_comment_user:
        print("[COMMENT][USER] Requested but feature missing; skipped.")

    # ================= Apply item comment fusion =================
    if args.use_comment_item and item_comment_feat is not None:
        base_item_dim_for_comment = data['item'].x.size(1)  # after rating fusion
        item_comment_fuse = ItemCommentFusion(base_dim=base_item_dim_for_comment,
                                              comment_dim=item_comment_feat.size(1),
                                              out_dim=fusion_out,
                                              hid_dim=fusion_hid).to(device)
        with torch.no_grad():
            data['item'].x = item_comment_fuse(data['item'].x, item_comment_feat)
        print(f"[COMMENT][ITEM] Fused item dim={data['item'].x.size(-1)}")
    elif args.use_comment_item:
        print("[COMMENT][ITEM] Requested but feature missing; skipped.")

    class LinkPredictor(nn.Module):
        def __init__(self, hidden=64, out=64, dropout = args.dropout):
            super().__init__()
            self.user_id_emb = nn.Embedding(data['user'].num_nodes, hidden)
            self.item_id_emb = nn.Embedding(data['item'].num_nodes, hidden)
            # Optionally init with Node2Vec (not required for fusion to work)
            n2v_local = load_node2vec(args.dataset_dir, (args.category if args.category else "all"), hidden)
            if n2v_local is not None:
                with torch.no_grad():
                    self.user_id_emb.weight.copy_(n2v_local['user_emb'].to(device=device, dtype=torch.float32))
                    self.item_id_emb.weight.copy_(n2v_local['item_emb'].to(device=device, dtype=torch.float32))
            self.encoder = to_hetero(SAGEEncoder(hidden, out, dropout), data.metadata())

        def forward(self, data):
            # data['item'].x already fused to 64-d tabove
            return self.encoder(data.x_dict, data.edge_index_dict)

        def score(self, u_emb, i_emb):
            return (u_emb * i_emb).sum(dim=-1)

    model = LinkPredictor(args.hidden, args.out).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # CHECKPOINT UTILS
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    def save_ckpt(epoch, final=False):
        tag = f"final_{epoch}" if final else f"epoch{epoch+1}"
        use_a2_tag = "A2_plus_A" if args.use_a2 else "A"
        rate_use_tag = "withRating" if args.use_rating else "noRating"
        text_user_tag = "withTextUsers" if args.use_comment_user else "noTextUsers"
        text_item_tag = "withTextItems" if args.use_comment_item else "noTextItems"
        dropout_tag = f"drop{args.dropout:.2f}".replace('.','p')
        node2vec_tag = f"n2v{args.n2v_dim}"
        fusion_tag = f"fusionOut{fusion_out}_fusionHid{(fusion_hid if fusion_hid is not None else 'auto')}" if (rating_fuse is not None or user_comment_fuse is not None or item_comment_fuse is not None) else "noFusion"
        hidden_tag = f"hidden{args.hidden}"
        out_tag = f"out{args.out}"
        prefix_tags = [node2vec_tag, fusion_tag, dropout_tag, hidden_tag, out_tag]
        prefix = "_".join(prefix_tags)     
        text_use_tag = f"{text_user_tag}_{text_item_tag}"
        fname = f"{args.category}_graphsage_{use_a2_tag}_{rate_use_tag}_{text_use_tag}_{prefix}_{tag}.pt"
        path = os.path.join(args.checkpoint_dir, fname)
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'opt_state': opt.state_dict(),
            'args': vars(args),
            # rating fusion
            'fuse_state': (rating_fuse.state_dict() if (rating_fuse is not None) else None),
            'fuse_meta': {
                'enabled': bool(args.use_rating and item_rating_feat is not None),
                'n2v_dim': int(item_x.size(1)),
                'rating_dim': (int(item_rating_feat.size(1)) if item_rating_feat is not None else 0),
                'out_dim': int(fusion_out),
                'hid_dim': (fusion_hid if fusion_hid is not None else int(item_x.size(1) + (item_rating_feat.size(1) if item_rating_feat is not None else 0)))
            },
            # user comment fusion
            'user_comment_fuse_state': (user_comment_fuse.state_dict() if (user_comment_fuse is not None) else None),
            'user_comment_meta': {
                'enabled': bool(args.use_comment_user and (user_comment_feat is not None)),
                'base_dim': int(user_x.size(1)),
                'comment_dim': (int(user_comment_feat.size(1)) if user_comment_feat is not None else 0),
                'out_dim': int(fusion_out),
                'hid_dim': (fusion_hid if fusion_hid is not None else int(user_x.size(1) + (user_comment_feat.size(1) if user_comment_feat is not None else 0)))
            },
            # item comment fusion
            'item_comment_fuse_state': (item_comment_fuse.state_dict() if (item_comment_fuse is not None) else None),
            'item_comment_meta': {
                'enabled': bool(args.use_comment_item and (item_comment_feat is not None)),
                'base_dim': int(item_x.size(1)),
                'comment_dim': (int(item_comment_feat.size(1)) if item_comment_feat is not None else 0),
                'out_dim': int(fusion_out),
                'hid_dim': (fusion_hid if fusion_hid is not None else int(item_x.size(1) + (item_comment_feat.size(1) if item_comment_feat is not None else 0)))
            },
            'timestamp': datetime.datetime.utcnow().isoformat()
        }, path)
        print(f"[CKPT] Saved: {path}")

    start_epoch = 0
    if args.resume_checkpoint:
        ck = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ck['model_state'])
        # Restore rating fusion
        ck_fuse = ck.get('fuse_state'); ck_meta = ck.get('fuse_meta', {}) or {}
        if ck_fuse is not None and ck_meta.get('enabled', False) and (item_rating_feat is not None) and args.use_rating:
            rating_fuse = ItemRatingFusion(
                n2v_dim=ck_meta.get('n2v_dim', item_x.size(1)),
                rating_dim=ck_meta.get('rating_dim', item_rating_feat.size(1) if item_rating_feat is not None else 0),
                out_dim=ck_meta.get('out_dim', fusion_out),
                hid_dim=ck_meta.get('hid_dim', fusion_hid)
            ).to(device)
            rating_fuse.load_state_dict(ck_fuse)
            with torch.no_grad():
                data['item'].x = rating_fuse(item_x, item_rating_feat)
            print("[CKPT] Restored rating fusion MLP.")
        # Restore user comment fusion
        uc_state = ck.get('user_comment_fuse_state'); uc_meta = ck.get('user_comment_meta', {}) or {}
        if args.use_comment_user and uc_state is not None and uc_meta.get('enabled', False) and user_comment_feat is not None:
            user_comment_fuse = UserCommentFusion(
                base_dim=uc_meta.get('base_dim', data['user'].x.size(1)),
                comment_dim=uc_meta.get('comment_dim', user_comment_feat.size(1)),
                out_dim=uc_meta.get('out_dim', fusion_out),
                hid_dim=uc_meta.get('hid_dim', fusion_hid)
            ).to(device)
            user_comment_fuse.load_state_dict(uc_state)
            with torch.no_grad():
                data['user'].x = user_comment_fuse(user_x.to(device), user_comment_feat)
            print("[CKPT] Restored user comment fusion MLP.")
        # Restore item comment fusion
        ic_state = ck.get('item_comment_fuse_state'); ic_meta = ck.get('item_comment_meta', {}) or {}
        if args.use_comment_item and ic_state is not None and ic_meta.get('enabled', False) and item_comment_feat is not None:
            item_comment_fuse = ItemCommentFusion(
                base_dim=ic_meta.get('base_dim', data['item'].x.size(1)),
                comment_dim=ic_meta.get('comment_dim', item_comment_feat.size(1)),
                out_dim=ic_meta.get('out_dim', fusion_out),
                hid_dim=ic_meta.get('hid_dim', fusion_hid)
            ).to(device)
            item_comment_fuse.load_state_dict(ic_state)
            with torch.no_grad():
                data['item'].x = item_comment_fuse(item_x.to(device), item_comment_feat)
            print("[CKPT] Restored item comment fusion MLP.")
        if not args.eval_only:
            opt.load_state_dict(ck['opt_state'])
            start_epoch = ck['epoch'] + 1
        print(f"[CKPT] Loaded checkpoint from {args.resume_checkpoint} (epoch {ck['epoch']})")

    # If eval-only requested, skip training
    eidx = data['user','interacts','item'].edge_index
    pos_u = eidx[0]; pos_i = eidx[1]
    bsz = args.batch
    loss_fn = nn.BCEWithLogitsLoss()

    if not args.eval_only:
        model.train()
        for epoch in range(start_epoch, args.epochs):
            perm = torch.randperm(pos_u.size(0), device=device)
            total = 0.0
            for s in range(0, pos_u.size(0), bsz):
                idx = perm[s:s+bsz]
                u = pos_u[idx]; i = pos_i[idx]
                neg_i = negative_sampling(u, data['item'].num_nodes, num_negs=args.negs)
                out = model(data)
                u_emb = out['user'][u]
                pi_emb = out['item'][i]
                ni_emb = out['item'][neg_i]
                pos_logit = (u_emb * pi_emb).sum(dim=-1)
                neg_logit = (u_emb.repeat_interleave(args.negs,0) * ni_emb).sum(dim=-1)
                y = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)])
                logits = torch.cat([pos_logit, neg_logit])
                loss = loss_fn(logits, y)
                opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss)
            if (epoch+1) % 2 == 0:
                print(f"epoch {epoch+1} loss {total:.3f}")
            if (epoch+1) % args.save_every == 0:
                save_ckpt(epoch)
        save_ckpt(args.epochs-1, final=True)
    else:
        print("[INFO] --eval-only set; skipping training.")

    # Inference
    model.eval()
    with torch.no_grad():
        out = model(data)
        U, I = out['user'], out['item']

    # 已见过滤
    seen = train.groupby('user_id')['item_id'].apply(set).to_dict()
    test_users = df[df.split=='test']['user_id'].unique()
    rankings = {}
    all_items = torch.arange(I.size(0), device=device)
    item_emb = I / (I.norm(dim=1, keepdim=True)+1e-12)

    for u in test_users:
        if u not in umap: 
            continue
        uid = umap[u]
        ue = U[uid:uid+1]
        ue = ue / (ue.norm(dim=1, keepdim=True)+1e-12)
        scores = (ue @ item_emb.T).squeeze(0)          # 余弦≈点积（已归一）
        # 过滤训练看过的
        ban = seen.get(u, set())
        if ban:
            ban_idx = torch.tensor([imap[i] for i in ban if i in imap], device=device)
            scores[ban_idx] = -1e9
        topk = torch.topk(scores, k=args.k).indices.tolist()
        rankings[u] = [i_idx[i] for i in topk]
        if args.eval_only:
            total = len(test_users)
            progress = (np.where(test_users == u)[0][0]+1)/total
            if (total >= 10) and (int(progress*10)*10 % 10 == 0):
                print(f"\r[INFO] Inference progress: {progress:.1%}", end='')

    gt = build_ground_truth(df, phase='test')
    rep = evaluate(rankings, gt, k_prec=args.kp, k_rec=args.kr, k_ndcg=args.kn)
    print("== GraphSAGE (mean/std) =="); print(rep)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Video_Games", help="Category to train on; default Video_Games.")
    ap.add_argument("--dataset-dir", default="data/loaded_data", help="Directory with preprocessed artifacts.")
    ap.add_argument("--use-a2", action="store_true", help="Load A^2 + A enriched graph.")
    ap.add_argument("--use-rating", action="store_true", help="Fuse item rating stats via MLP into item embeddings.")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--out", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--negs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--kp", type=int, default=10)
    ap.add_argument("--kr", type=int, default=20)
    ap.add_argument("--kn", type=int, default=10)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--checkpoint-dir", default="checkpoints", help="Directory to store training checkpoints.")
    ap.add_argument("--resume-checkpoint", default=None, help="Path to checkpoint to resume from.")
    ap.add_argument("--eval-only", action="store_true", help="Skip training; just evaluate loaded checkpoint/model.")
    ap.add_argument("--save-every", type=int, default=100, help="Save checkpoint every N epochs.")
    ap.add_argument("--use-comment-user", action="store_true",
                    help="Fuse aggregated comment embeddings into user features.")
    ap.add_argument("--use-comment-item", action="store_true",
                    help="Fuse aggregated comment embeddings into item features.")
    ap.add_argument("--comment-emb", default="data/textfeat/comment_text_emb.npy",
                    help="Per-comment embedding file (from build_text_features --save-comment-emb).")
    ap.add_argument("--comment-index", default="data/textfeat/comment_text_index.csv",
                    help="Comment index mapping file (row,user_id,item_id).")
    ap.add_argument("--n2v-dim", type=int, default=128, help="Dimension of precomputed Node2Vec to load.")
    ap.add_argument("--fusion-out", type=int, default=1024, help="Output dim of rating/comment fusion MLPs (default: keep input dim).")
    ap.add_argument("--fusion-hid", type=int, default=1024, help="Hidden dim of fusion MLPs. Default: base_dim+extra_dim.")
    ap.add_argument("--expand-hidden", action="store_true", help="If set, bump --hidden up to fusion-out to avoid early compression.")
    args = ap.parse_args()
    print("The args:", args)    
    main(args)
