#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_text_features.py
把 Video_Games.jsonl 里的 (title + text) 文本编码为每个 item 的向量，并保存：
  - item_text_emb.npy:   (#items, dim) float32
  - item_text_index.csv: item_id -> 行号 的映射

支持：
  - 健壮 JSONL 读取（修复常见错误；跨行拼接；可跳过坏行并记录）
  - Sentence-BERT 或 TF-IDF+SVD 两种文本编码
  - 基于 helpful_vote / rating 的加权聚合
  - 全程详细进度日志

示例：
  python build_text_features.py ^
    --reviews "C:\\Users\\hzz20\\Downloads\\Video_Games.jsonl" ^
    --out-dir data\\textfeat ^
    --encoder sbert ^
    --sbert-model all-MiniLM-L6-v2 ^
    --use-weights ^
    --skip-bad --error-log bad_lines.log
"""

import argparse, os, sys, re, json, time
import gzip
import numpy as np
import pandas as pd

# ========================= 基础日志 =========================
def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ========================= 健壮 JSONL 读取 =========================
def read_jsonl_with_progress(path,
                             log_every: int = 100_000,
                             tolerant: bool = True,
                             skip_bad: bool = False,
                             error_log: str | None = None,
                             max_errors: int = 1000,
                             join_multiline: bool = True,
                             join_limit: int = 50) -> pd.DataFrame:
    """
    更健壮的 JSONL 读取：
      - tolerant: 常见修复（去尾逗号、True/False/None -> JSON 合法、去 BOM、NaN/Inf -> null）
      - join_multiline: 解析失败时向下拼接多行再试（处理跨行记录）
      - skip_bad: 仍失败则跳过并记录到 error_log
    """

    def _open(p):
        return gzip.open(p, "rt", encoding="utf-8", errors="replace") if str(p).endswith(".gz") \
               else open(p, "r", encoding="utf-8", errors="replace")

    def count_lines(p):
        c = 0
        with _open(p) as f:
            for _ in f:
                c += 1
        return c

    def _fix_common(s: str) -> str:
        s = s.lstrip("\ufeff").strip()  # 去 BOM

        # 去对象/数组收尾前的尾逗号：  ,  }
        s = re.sub(r',(\s*[}\]])', r'\1', s)

        # 仅当冒号后直接出现这些“Python 风格字面量”时再替换；
        # 使用固定宽度回溯 (?<=:)，把可能的空格放到回溯外
        s = re.sub(r'(?<=:)\s*True\b',  ' true',  s)
        s = re.sub(r'(?<=:)\s*False\b', ' false', s)
        s = re.sub(r'(?<=:)\s*None\b',  ' null',  s)

        # 处理 NaN / Infinity
        s = re.sub(r'(?<=:)\s*(NaN|Infinity|-Infinity)\b', ' null', s)

        return s


    def _loads_one(raw: str):
        if not tolerant:
            return json.loads(raw)
        return json.loads(_fix_common(raw))

    total = count_lines(path)
    if total == 0:
        log(f"Reading JSONL: {path} (empty)")
        return pd.DataFrame([])

    elog = open(error_log, "w", encoding="utf-8") if error_log else None
    rows, bad, i = [], 0, 0
    t0 = time.perf_counter(); last = t0

    log(f"Reading JSONL: {path} (lines={total:,})")
    with _open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            i += 1
            raw = line.rstrip("\r\n")
            if not raw:
                continue
            try:
                obj = _loads_one(raw)
                rows.append(obj)
            except json.JSONDecodeError:
                # 尝试跨行拼接
                joined, ok = raw, False
                if join_multiline:
                    for _k in range(join_limit):
                        nxt = f.readline()
                        if not nxt:
                            break
                        i += 1
                        joined += "\n" + nxt.rstrip("\r\n")
                        try:
                            obj = _loads_one(joined)
                            rows.append(obj)
                            ok = True
                            break
                        except json.JSONDecodeError:
                            continue
                if not ok:
                    bad += 1
                    if elog:
                        elog.write(f"--- line {i} ---\n{joined}\n")
                    snippet = (raw[:200] + ("..." if len(raw) > 200 else ""))
                    log(f"!! JSON decode failed at line {i} (bad={bad}/{max_errors}). Snippet: {snippet}")
                    if not skip_bad or bad >= max_errors:
                        if elog:
                            elog.close()
                        raise

            if (i % max(1, log_every)) == 0:
                t1 = time.perf_counter()
                speed = log_every / max(1e-9, (t1 - last))
                pct = 100.0 * i / max(1, total)
                log(f"  parsed {i:,}/{total:,} ({pct:5.1f}%)  ~{speed:,.0f} lines/s")
                last = t1

    if elog:
        elog.close()

    t1 = time.perf_counter()
    log(f"Finished reading: ok={len(rows):,}, bad={bad:,}, time={t1-t0:.2f}s")
    return pd.DataFrame(rows)

# ========================= 文本拼接与编码 =========================
def concat_text(row) -> str:
    # Fallbacks for Amazon JSON: summary/reviewText
    t1 = str(row.get("title") or row.get("summary") or "")
    t2 = str(row.get("text") or row.get("reviewText") or "")
    return (t1 + " " + t2).strip()

def pick_device(user_choice: str = "auto") -> str:
    """
    SentenceTransformer 支持 device='cpu' 或 'cuda'。
    """
    if user_choice.lower() == "cpu":
        return "cpu"
    if user_choice.lower() == "cuda":
        return "cuda"
    # auto
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def encode_sbert(texts, model_name="all-MiniLM-L6-v2", batch=64, normalize=True, device="auto"):
    from sentence_transformers import SentenceTransformer
    device = pick_device(device)
    log(f"Loading SBERT model: {model_name} (device={device})")
    model = SentenceTransformer(model_name, device=device)
    log(f"Encoding {len(texts):,} texts with SBERT (batch={batch}) ...")
    embs = model.encode(
        texts,
        batch_size=batch,
        show_progress_bar=True,
        normalize_embeddings=normalize
    )
    Z = np.asarray(embs, dtype=np.float32)
    log(f"SBERT done: shape={Z.shape}, dtype={Z.dtype}")
    return Z

def encode_tfidf(texts, dim=512, max_features=50_000):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    log(f"TF-IDF vectorizing {len(texts):,} docs (max_features={max_features}, ngram=(1,2)) ...")
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = vec.fit_transform(texts)  # sparse [N, V]
    log(f"SVD reducing to {dim} dims ...")
    svd = TruncatedSVD(n_components=dim, random_state=0)
    Z = svd.fit_transform(X).astype(np.float32)  # [N, dim]
    # L2 normalize
    n = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    Z = Z / n
    log(f"TF-IDF+SVD done: shape={Z.shape}, dtype={Z.dtype}")
    return Z

# ========================= 按 item 聚合 =========================
def aggregate_item_embeddings(df: pd.DataFrame,
                              Z: np.ndarray,
                              item_key: str,
                              weight_col: str = "_w",
                              group_log_every: int = 10_000):
    """
    按 item 对评论向量做加权平均，得到每个 item 一个向量
    """
    assert len(df) == Z.shape[0], "Z 与 df 行数需要一致"
    df = df.copy()
    df["_idx"] = np.arange(len(df))
    groups = df.groupby(item_key, sort=False)

    item_ids = groups.size().index.tolist()
    id2row = {it: i for i, it in enumerate(item_ids)}
    D = Z.shape[1]
    item_emb = np.zeros((len(item_ids), D), dtype=np.float32)

    t0 = time.perf_counter()
    for gi, (it, sub) in enumerate(groups, start=1):
        idxs = sub["_idx"].to_numpy()
        ws = sub[weight_col].to_numpy(dtype=np.float32).reshape(-1, 1)
        v = (Z[idxs] * ws).sum(axis=0) / (ws.sum() + 1e-12)
        item_emb[id2row[it]] = v

        if (gi % max(1, group_log_every)) == 0:
            dt = time.perf_counter() - t0
            speed = group_log_every / max(1e-9, dt)
            log(f"  grouped {gi:,}/{len(item_ids):,} items  ~{speed:,.0f} items/s")
            t0 = time.perf_counter()

    return item_ids, item_emb

# ========================= 主流程 =========================
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # 读取 JSONL
    df = read_jsonl_with_progress(
        args.reviews,
        log_every=args.log_interval,
        tolerant=True,
        skip_bad=args.skip_bad,
        error_log=args.error_log,
        max_errors=args.max_errors,
        join_multiline=True,
        join_limit=args.join_limit
    )
    if df.empty:
        log("Input is empty. Exit.")
        return

    # 选择 item 键
    item_key = "parent_asin" if ("parent_asin" in df.columns) else "asin"
    if item_key not in df.columns:
        print("需要字段 parent_asin 或 asin", file=sys.stderr)
        sys.exit(1)

    # 先过滤缺少 item_id 的行，再做编码，确保 df 与 Z 同步
    before = len(df)
    df = df[df[item_key].notna()].copy()
    if len(df) != before:
        log(f"Drop rows without {item_key}: {before - len(df):,}")

    # 文本拼接
    log("Concatenating title + text -> text_all ...")
    df["text_all"] = df.apply(concat_text, axis=1)
    empty_texts = (df["text_all"].str.strip() == "").sum()
    log(f"text rows: {len(df):,} | empty texts: {empty_texts:,}")

    # 权重
    if args.use_weights:
        log("Computing per-row weights (helpful_vote * rating factor) ...")
        w = np.ones(len(df), dtype=np.float32)
        if "helpful_vote" in df.columns:
            w *= (1.0 + df["helpful_vote"].fillna(0).to_numpy(dtype=np.float32))
        if "rating" in df.columns:
            w *= (df["rating"].fillna(0).to_numpy(dtype=np.float32) / 5.0 + 0.5)  # 0.5~1.5
        df["_w"] = w
    else:
        df["_w"] = 1.0

    # 编码
    texts = df["text_all"].fillna("").astype(str).tolist()
    if args.encoder == "sbert":
        Z = encode_sbert(
            texts,
            model_name=args.sbert_model,
            batch=args.batch,
            normalize=True,
            device=args.device
        )
    else:
        Z = encode_tfidf(
            texts,
            dim=args.dim,
            max_features=args.max_features
        )

    # 聚合到 item
    log(f"Grouping to items by '{item_key}' and aggregating (weighted mean) ...")
    item_ids, item_emb = aggregate_item_embeddings(
        df=df, Z=Z, item_key=item_key, weight_col="_w",
        group_log_every=args.group_log_interval
    )

    # 保存
    emb_path = os.path.join(args.out_dir, "item_text_emb.npy")
    idx_path = os.path.join(args.out_dir, "item_text_index.csv")
    np.save(emb_path, item_emb)
    pd.DataFrame({"item_id": item_ids, "idx": np.arange(len(item_ids))}) \
        .to_csv(idx_path, index=False)

    log(f"[OK] items={len(item_ids):,} dim={item_emb.shape[1]}")
    log(f"saved: {emb_path}")
    log(f"saved: {idx_path}")

    # 打印若干样例
    if args.show_examples > 0:
        k = min(args.show_examples, len(item_ids))
        log(f"Examples (first {k} items; first 8 dims):")
        for i in range(k):
            it = item_ids[i]
            vec8 = " ".join([f"{x:.4f}" for x in item_emb[i, :8]])
            print(f"  {i:05d}  item_id={it}  emb[:8]=[{vec8}]")

# ========================= CLI =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # 输入/输出
    ap.add_argument("--reviews", default="data/Video_Games.jsonl",
                    help="JSONL 或 .jsonl.gz 路径，每行一个评论对象")
    ap.add_argument("--out-dir", default="data/textfeat",
                    help="输出目录（会自动创建）")

    # 编码选择
    ap.add_argument("--encoder", choices=["sbert", "tfidf"], default="sbert",
                    help="文本编码器：sbert 或 tfidf")
    ap.add_argument("--sbert-model", default="all-MiniLM-L6-v2",
                    help="Sentence-BERT 模型名")
    ap.add_argument("--batch", type=int, default=64,
                    help="SBERT 编码 batch size")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="SBERT 运行设备选择")
    # TF-IDF 兜底
    ap.add_argument("--dim", type=int, default=512,
                    help="TF-IDF+SVD 的输出维度")
    ap.add_argument("--max-features", type=int, default=50_000,
                    help="TF-IDF 最大特征数")

    # 权重
    ap.add_argument("--use-weights", action="store_true",
                    help="使用 (1+helpful_vote)*(rating/5+0.5) 作为聚合权重")

    # 进度与健壮读取参数
    ap.add_argument("--log-interval", type=int, default=100_000,
                    help="读取 JSONL 时每 N 行打印一次进度")
    ap.add_argument("--group-log-interval", type=int, default=10_000,
                    help="按 item 聚合时每 N 组打印一次进度")
    ap.add_argument("--skip-bad", action="store_true",
                    help="遇到无法修复的坏行时跳过并记录到 --error-log")
    ap.add_argument("--error-log", default="bad_lines.log",
                    help="保存坏行原文以便排查")
    ap.add_argument("--max-errors", type=int, default=1000,
                    help="坏行累计到此数目即停止并报错")
    ap.add_argument("--join-limit", type=int, default=50,
                    help="解析失败时向下拼接最多 N 行再试（处理跨行记录）")

    # 输出示例
    ap.add_argument("--show-examples", type=int, default=3,
                    help="保存后打印前 N 个 item 的向量前 8 维，0 表示不打印")

    args = ap.parse_args()

    # 兜底默认，防止 AttributeError（即使有人删了 CLI 也能跑）
    args.skip_bad = getattr(args, "skip_bad", False)
    args.error_log = getattr(args, "error_log", "bad_lines.log")
    args.max_errors = getattr(args, "max_errors", 1000)
    args.join_limit = getattr(args, "join_limit", 50)
    args.log_interval = getattr(args, "log_interval", 100_000)
    args.group_log_interval = getattr(args, "group_log_interval", 10_000)
    args.show_examples = getattr(args, "show_examples", 3)
    args.device = getattr(args, "device", "auto")

    print("The args:", args)
    main(args)
