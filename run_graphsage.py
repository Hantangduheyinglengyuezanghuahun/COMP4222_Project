import argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from eval import build_ground_truth, evaluate
import os

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
    def __init__(self, hidden=64, out=64):
        super().__init__()
        self.user_emb = nn.Embedding(1, hidden)  # 仅占位；真正特征在 to_hetero 时替换
        self.item_emb = nn.Embedding(1, hidden)
        self.conv1 = SAGEConv((-1, -1), hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = SAGEConv((-1, -1), out)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.mlp(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

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
    print(f"Data loaded with {data['user'].num_nodes} users and {data['item'].num_nodes} items.")

    n2v = load_node2vec(args.dataset_dir, prefix, args.hidden)
    if n2v is not None:
        # ensure features live on the same device/dtype as the graph
        user_x = n2v['user_emb'].to(device=device, dtype=torch.float32)
        item_x = n2v['item_emb'].to(device=device, dtype=torch.float32)
        data['user'].x = user_x
        data['item'].x = item_x
        print(f"Loaded Node2Vec features: dim={data['user'].x.size(-1)} (on {data['user'].x.device})")

    # build degrees (already on `device` because ei is on device)
    ei = data['user','interacts','item'].edge_index
    U = data['user'].num_nodes; I = data['item'].num_nodes
    user_deg = torch.bincount(ei[0], minlength=U).float().unsqueeze(1)
    item_deg = torch.bincount(ei[1], minlength=I).float().unsqueeze(1)
    user_deg = (user_deg + 1.0).log()
    item_deg = (item_deg + 1.0).log()

    # concat on the same device
    data['user'].x = torch.cat([data['user'].x, user_deg], dim=1)
    data['item'].x = torch.cat([data['item'].x, item_deg], dim=1)

    class LinkPredictor(nn.Module):
        def __init__(self, hidden=64, out=64):
            super().__init__()
            self.user_id_emb = nn.Embedding(data['user'].num_nodes, hidden)
            self.item_id_emb = nn.Embedding(data['item'].num_nodes, hidden)
            n2v = load_node2vec(args.dataset_dir, prefix, hidden)
            if n2v is not None:
                with torch.no_grad():
                    self.user_id_emb.weight.copy_(n2v['user_emb'])
                    self.item_id_emb.weight.copy_(n2v['item_emb'])
            self.encoder = to_hetero(SAGEEncoder(hidden, out), data.metadata())

        def forward(self, data):
            x_dict = self.encoder(data.x_dict, data.edge_index_dict)
            return x_dict

        def score(self, u_emb, i_emb):
            return (u_emb * i_emb).sum(dim=-1)  # 点积

    model = LinkPredictor(args.hidden, args.out).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # 构造训练正样本
    eidx = data['user','interacts','item'].edge_index
    pos_u = eidx[0]; pos_i = eidx[1]
    bsz = args.batch

    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(args.epochs):
        perm = torch.randperm(pos_u.size(0), device=device)
        print(device)
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

    # 推理并推荐
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

    gt = build_ground_truth(df, phase='test')
    rep = evaluate(rankings, gt, k_prec=args.kp, k_rec=args.kr, k_ndcg=args.kn)
    print("== GraphSAGE (mean/std) =="); print(rep)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None)
    ap.add_argument("--dataset-dir", default="data/loaded_data", help="Directory with preprocessed artifacts.")
    ap.add_argument("--use-a2", action="store_true", help="Load A^2 + A enriched graph.")
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
    args = ap.parse_args()
    print("The args:", args)    
    main(args)
