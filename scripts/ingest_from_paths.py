# project/scripts/ingest_from_paths.py
import re, sys
from pathlib import Path
import pandas as pd

PAT = re.compile(r'(?P<cat>[^/\\]+)\.(?P<split>train|valid|test)\.csv$', re.I)

def read_one(path: Path):
    m = PAT.search(path.name)
    if not m:
        raise ValueError(f"Filename must look like <Category>.<train|valid|test>.csv: {path}")
    cat = m.group('cat')
    split = m.group('split').lower()
    split = {'train':'train','valid':'val','test':'test'}[split]

    df = pd.read_csv(path)
    cols = set(c.lower() for c in df.columns)

    # 标准化列名
    rename = {}
    # user_id
    for cand in ['user_id','user','uid']:
        if cand in cols: 
            rename[[c for c in df.columns if c.lower()==cand][0]] = 'user_id'; break
    # item_id (优先 parent_asin)
    if 'parent_asin' in cols:
        rename[[c for c in df.columns if c.lower()=='parent_asin'][0]] = 'item_id'
    elif 'item_id' in cols:
        rename[[c for c in df.columns if c.lower()=='item_id'][0]] = 'item_id'
    elif 'asin' in cols:
        rename[[c for c in df.columns if c.lower()=='asin'][0]] = 'item_id'
    else:
        raise ValueError(f"Cannot find item id column in {path}. Expect parent_asin / item_id / asin.")

    # rating
    if 'rating' in cols:
        rename[[c for c in df.columns if c.lower()=='rating'][0]] = 'rating'
    else:
        df['rating'] = 1.0  # 若无评分，设为隐式反馈

    # timestamp（可选）
    if 'timestamp' in cols:
        rename[[c for c in df.columns if c.lower()=='timestamp'][0]] = 'timestamp'
    else:
        df['timestamp'] = 0

    df = df.rename(columns=rename)
    keep = ['user_id','item_id','rating','timestamp']
    df = df[keep].copy()
    df['split'] = split
    df['category'] = cat
    return df

def main(paths):
    frames = []
    for p in paths:
        p = Path(p).expanduser()
        frames.append(read_one(p))
        print(f"[OK] Loaded {p}")
    all_df = pd.concat(frames, ignore_index=True)

    # 去重（某些发布会含重叠行）
    all_df = all_df.drop_duplicates(subset=['user_id','item_id','timestamp','split','category'])

    out_dir = Path("project/data"); out_dir.mkdir(parents=True, exist_ok=True)
    # 写总表
    all_path = out_dir / "interactions.split.csv"
    all_df.to_csv(all_path, index=False)
    print(f"[SAVE] {all_path}  rows={len(all_df)}")

    # 各类别单独文件
    for cat, g in all_df.groupby('category'):
        p = out_dir / f"interactions.{cat}.split.csv"
        g.to_csv(p, index=False)
        print(f"[SAVE] {p}  rows={len(g)}")

    # 小结
    print("\n== Summary ==")
    print(all_df.groupby(['category','split']).size().unstack(fill_value=0))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_from_paths.py <csv1> <csv2> ...")
        sys.exit(1)
    main(sys.argv[1:])
