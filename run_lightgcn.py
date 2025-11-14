import sys, os, datetime, pickle, torch, torch.nn.functional as F
import numpy as np
from eval import build_ground_truth, evaluate

repo_code = os.path.join(os.path.dirname(__file__), "LightGCN_PyTorch", "code")
sys.path.insert(0, repo_code)
from world import args            # now includes category/dataset_dir/use_a2
from model import LightGCN

def load_preprocessed(dataset_dir, category, use_a2=False):
    if category is None:
        raise ValueError("Provide --category (Video_Games or Electronics).")
    prefix = category
    fname = f"{prefix}_graphsage_A2_plus_A.pt" if use_a2 else f"{prefix}_graphsage_A.pt"
    path = os.path.join(dataset_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run dataset/pd.py first.")
    return torch.load(path, map_location="cpu")

def save_ckpt(path, epoch, model, opt):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'opt_state': opt.state_dict(),
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'cli_args': vars(args)
    }, path)
    print(f"[CKPT] Saved {path}")

def load_ckpt(path, model, opt, eval_only=False):
    ck = torch.load(path, map_location='cpu')
    model.load_state_dict(ck['model_state'])
    if not eval_only:
        opt.load_state_dict(ck['opt_state'])
    print(f"[CKPT] Loaded {path} (epoch {ck['epoch']})")
    return ck['epoch'] + 1

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded = load_preprocessed(args.dataset_dir, args.category, bool(args.use_a2))
    data = loaded['data'].to(device)
    umap = loaded['umap']; imap = loaded['imap']
    u_idx = loaded['u_idx']; i_idx = loaded['i_idx']
    df_path = os.path.join("data", f"interactions.{args.category}.split.csv")
    import pandas as pd
    df = pd.read_csv(df_path)
    train = df[df.split=='train'][['user_id','item_id']]
    test_users = df[df.split=='test']['user_id'].unique()

    # Build normalized adjacency
    eidx = data['user','interacts','item'].edge_index.to(device)
    U = data['user'].num_nodes; I = data['item'].num_nodes
    u = eidx[0]; i = eidx[1] + U
    row = torch.cat([u, i]); col = torch.cat([i, u])
    vals = torch.ones(row.size(0), device=device)
    A = torch.sparse_coo_tensor(torch.stack([row, col]), vals, (U+I, U+I)).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5); deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
    norm_vals = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    A_norm = torch.sparse_coo_tensor(torch.stack([row, col]), norm_vals, (U+I, U+I)).coalesce()

    # Adapter dataset fulfilling model expectation
    class Adapter:
        def __init__(self):
            self.n_users = U; self.m_items = I; self.Graph = A_norm
            pos_lists = [[] for _ in range(U)]
            for _, r in train.iterrows():
                uu = umap.get(r['user_id']); ii = imap.get(r['item_id'])
                if uu is not None and ii is not None:
                    pos_lists[uu].append(ii)
            self._allPos = pos_lists
            test = df[df.split=='test'][['user_id','item_id']]
            test_map = {}
            for _, r in test.iterrows():
                uu = umap.get(r['user_id']); ii = imap.get(r['item_id'])
                if uu is not None and ii is not None:
                    test_map.setdefault(uu, set()).add(ii)
            self._testDict = test_map
        def getSparseGraph(self): return self.Graph
        @property
        def allPos(self): return self._allPos
        @property
        def testDict(self): return self._testDict
        @property
        def trainDataSize(self): return sum(len(l) for l in self._allPos)

    adapter = Adapter()
    config = {
        'latent_dim_rec': args.recdim,
        'lightGCN_n_layers': args.layer,
        'keep_prob': args.keepprob,
        'A_split': False,
        'pretrain': args.pretrain,
        'dropout': args.dropout,
        'lr': args.lr,
        'decay': args.decay
    }
    model = LightGCN(config, adapter).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    start_epoch = 0
    if args.resume_checkpoint:
        start_epoch = load_ckpt(args.resume_checkpoint, model, opt, eval_only=bool(args.eval_only))

    # Positives
    pos_u = eidx[0]; pos_i = eidx[1]
    bsz = args.bpr_batch
    if not args.eval_only:
        model.train()
        for epoch in range(start_epoch, args.epochs):
            perm = torch.randperm(pos_u.size(0), device=device)
            total = 0.0
            for s in range(0, pos_u.size(0), bsz):
                idx = perm[s:s+bsz]
                u_b = pos_u[idx]; p_b = pos_i[idx]

                if args.negs <= 1:
                    neg = torch.randint(0, I, (u_b.size(0),), device=device)
                    loss, reg = model.bpr_loss(u_b, p_b, neg)
                else:
                    # repeat users/positives so lengths match flattened negatives
                    u_rep = u_b.repeat_interleave(args.negs)
                    p_rep = p_b.repeat_interleave(args.negs)
                    neg = torch.randint(0, I, (u_rep.size(0),), device=device)
                    loss, reg = model.bpr_loss(u_rep, p_rep, neg)

                loss = loss + args.decay * reg
                opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss)
            with torch.no_grad():
                model.embedding_user.weight.data = F.normalize(model.embedding_user.weight.data, p=2, dim=1)
                model.embedding_item.weight.data = F.normalize(model.embedding_item.weight.data, p=2, dim=1)
            if (epoch+1) % 2 == 0:
                print(f"[Train] epoch {epoch+1} loss {total:.4f}")
            if (epoch+1) % args.save_every == 0:
                ck = os.path.join(args.path, f"lgcn_epoch{epoch+1}.pt")
                os.makedirs(args.path, exist_ok=True)
                save_ckpt(ck, epoch, model, opt)
        final = os.path.join(args.path, "lgcn_final.pt")
        save_ckpt(final, args.epochs-1, model, opt)
    else:
        print("[INFO] eval_only: skipping training")

    # Evaluation
    model.eval()
    with torch.no_grad():
        Ue, Ie = model.computer()
        Ie = Ie / (Ie.norm(dim=1, keepdim=True)+1e-12)
        seen = train.groupby('user_id')['item_id'].apply(set).to_dict()
        rankings = {}
        for uraw in test_users:
            if uraw not in umap: continue
            uid = umap[uraw]
            ue = Ue[uid:uid+1]
            ue = ue / (ue.norm(dim=1, keepdim=True)+1e-12)
            scores = (ue @ Ie.T).squeeze(0)
            ban = seen.get(uraw, set())
            if ban:
                bidx = [imap[i] for i in ban if i in imap]
                if bidx:
                    scores[torch.tensor(bidx, device=device)] = -1e9
            topk = torch.topk(scores, k=args.k).indices.tolist()
            rankings[uraw] = [i_idx[i] for i in topk]
    import pandas as pd
    gt = build_ground_truth(pd.read_csv(df_path), phase='test')
    rep = evaluate(rankings, gt, k_prec=args.kp, k_rec=args.kr, k_ndcg=args.kn)
    print("== LightGCN (mean/std) ==")
    print(rep)

if __name__ == "__main__":
    main()