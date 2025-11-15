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
        return x

class ItemRatingFusion(nn.Module):
    def __init__(self, n2v_dim=64, rating_dim=2, out_dim=64):
        super().__init__()
        hid = max(128, out_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(n2v_dim + rating_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, out_dim)
        )

    def forward(self, item_n2v, rating_feat):
        return self.net(torch.cat([item_n2v, rating_feat], dim=1))

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

    # Load Node2Vec features
    n2v = load_node2vec(args.dataset_dir, (args.category if args.category else "all"), args.hidden)
    if n2v is None:
        raise FileNotFoundError(f"Node2Vec features not found. Run dataset/node2vec.py first.")
    user_x = n2v['user_emb'].to(device=device, dtype=torch.float32)  # [U,64]
    item_x = n2v['item_emb'].to(device=device, dtype=torch.float32)  # [I,64]
    data['user'].x = user_x
    data['item'].x = item_x
    print(f"Loaded Node2Vec features: user_dim={data['user'].x.size(-1)}, item_dim={data['item'].x.size(-1)}")

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

    # Fuse ratings into item embeddings only if enabled
    if args.use_rating and item_rating_feat is not None:
        fuse = ItemRatingFusion(n2v_dim=item_x.size(1), rating_dim=item_rating_feat.size(1), out_dim=args.hidden).to(device)
        with torch.no_grad():
            data['item'].x = fuse(data['item'].x, item_rating_feat)  # keep 64-d
        print(f"Fused item features: item_dim={data['item'].x.size(-1)}")
    else:
        # Keep original Node2Vec item embeddings
        print(f"Item features unchanged (dim={data['item'].x.size(-1)})")

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

    model = LinkPredictor(args.hidden, args.out).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # CHECKPOINT UTILS
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    def save_ckpt(epoch, final=False):
        tag = f"final" if final else f"epoch{epoch+1}"
        use_a2_tag = "A2_plus_A" if args.use_a2 else "A"
        fname = f"{prefix}_graphsage_{use_a2_tag}_{tag}.pt"
        path = os.path.join(args.checkpoint_dir, fname)
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'opt_state': opt.state_dict(),
            'args': vars(args),
            'timestamp': datetime.datetime.utcnow().isoformat()
        }, path)
        print(f"[CKPT] Saved: {path}")

    start_epoch = 0
    if args.resume_checkpoint:
        ck = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ck['model_state'])
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
        total = len(test_users)
        progress = (np.where(test_users == u)[0][0]+1)/total
        if (total >= 10) and (int(progress*10)*10 % 10 == 0):
            print(f"\r[INFO] Inference progress: {progress:.1%}", end='')

    gt = build_ground_truth(df, phase='test')
    rep = evaluate(rankings, gt, k_prec=args.kp, k_rec=args.kr, k_ndcg=args.kn)
    print("== GraphSAGE (mean/std) =="); print(rep)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None)
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
    args = ap.parse_args()
    print("The args:", args)    
    main(args)
