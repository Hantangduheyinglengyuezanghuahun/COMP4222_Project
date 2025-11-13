# Hyperparameter: epochs, dim
import os, argparse, torch
from torch_geometric.nn.models import Node2Vec

def to_homogeneous(edge_index_ui, num_users, num_items):
    # user (0..U-1), item (U..U+I-1)
    src_u = edge_index_ui[0]
    dst_i = edge_index_ui[1] + num_users
    e1 = torch.stack([src_u, dst_i], dim=0)
    e2 = torch.stack([dst_i, src_u], dim=0)  # undirected
    return torch.cat([e1, e2], dim=1), num_users + num_items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="data/loaded_data")
    ap.add_argument("--category", default="Video_Games")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--walk-length", type=int, default=20)
    ap.add_argument("--context-size", type=int, default=10)
    ap.add_argument("--walks-per-node", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.01)
    args = ap.parse_args()

    prefix = args.category
    pack = torch.load(os.path.join(args.dataset_dir, f"{prefix}_graphsage_A.pt"), map_location="cpu")
    data = pack["data"]
    U = data["user"].num_nodes
    I = data["item"].num_nodes
    ei_ui = data["user","interacts","item"].edge_index.cpu()

    edge_index_h, N = to_homogeneous(ei_ui, U, I)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n2v = Node2Vec(
        edge_index_h, embedding_dim=args.dim, walk_length=args.walk_length,
        context_size=args.context_size, walks_per_node=args.walks_per_node,
        p=1.0, q=1.0, num_nodes=N, sparse=True
    ).to(device)

    loader = n2v.loader(batch_size=128, shuffle=True, num_workers=0)
    opt = torch.optim.SparseAdam(list(n2v.parameters()), lr=args.lr)

    n2v.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for pos_rw, neg_rw in loader:
            opt.zero_grad()
            loss = n2v.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            opt.step()
            total += float(loss)
        print(f"epoch {epoch} n2v loss {total/len(loader):.4f}")

    with torch.no_grad():
        emb = n2v.embedding.weight.detach().cpu()  # [U+I, D]
        user_emb = emb[:U].contiguous()
        item_emb = emb[U:].contiguous()

    out = os.path.join(args.dataset_dir, f"{prefix}_node2vec_{args.dim}.pt")
    torch.save({"user_emb": user_emb, "item_emb": item_emb}, out)
    print(f"Saved Node2Vec embeddings to {out}")

if __name__ == "__main__":
    main()