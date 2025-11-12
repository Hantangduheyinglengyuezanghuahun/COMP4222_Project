# COMP4222 Project: Recommender Baselines (Popularity / PPR / ALS / GraphSAGE)

## Overview
- Popularity baseline: eval.py
- Personalized PageRank: run_ppr.py
- ALS: run_als.py
- GraphSAGE: run_graphsage.py
  - Uses either A (user–item bipartite) or A^2 + A (adds 2-hop user–user and item–item edges)
- Data are preprocessed once and reused from data/loaded_data

## Directory highlights
- dataset/pd.py: preprocess CSV → Parquet/CSV + GraphSAGE graph artifacts (A and A^2 + A)
- dataset/precompute_node2vec.py: optional Node2Vec features for nodes
- run_ppr.sh, run_graphsage.sh: SLURM scripts

## Environment setup (Linux)

### Option A: Conda (recommended)
```bash
conda create -n comp4222 python=3.10 -y
conda activate comp4222

# PyTorch (CUDA 12.1) – adjust if you need CPU-only
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# PyG core (no compiled ops)
pip install torch-geometric==2.5.0

# Scientific stack
pip install pandas==2.2.2 numpy==1.26.4 scipy==1.10.1 scikit-learn==1.3.2 implicit==0.7.2

# Optional for Parquet
pip install pyarrow==14.0.2 fastparquet==2023.8.0

# Optional (only if you want Node2Vec precompute)
pip install pyg-lib torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

### Option B: Using virtualenv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

## Data Preparation

Raw split CSVs expected at:
- data/interactions.Video_Games.split.csv
- data/interactions.Electronics.split.csv

Preprocess and generate artifacts (Parquet + GraphSAGE graphs) into data/loaded_data:

Process Video Games only:
python dataset/pd.py --dataset video_games

Process Electronics only:
python dataset/pd.py --dataset electronics

Process both (Video_Games first then Electronics):
python dataset/pd.py --dataset all

Artifacts created per prefix (e.g. Video_Games):
- data/loaded_data/Video_Games_interactions.parquet (or .csv fallback)
- data/loaded_data/Video_Games_graphsage_A.pt
- data/loaded_data/Video_Games_graphsage_A2_plus_A.pt
- data/loaded_data/Video_Games_meta.json

Repeat for Electronics.

## Running Models

Popularity:
python eval.py --category Video_Games

PPR:
python run_ppr.py --category Video_Games --dataset-dir data/loaded_data

ALS:
python run_als.py --category Video_Games --dataset-dir data/loaded_data

GraphSAGE (base A):
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data

GraphSAGE (enriched A^2 + A):
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --use-a2

Change category to Electronics as needed.

## SLURM (Examples)

GraphSAGE:
sbatch run_graphsage.sh

PPR:
sbatch run_ppr.sh

Each script:
1. Prints environment info (Python / PyTorch / CUDA).
2. Optionally preprocess (add a call to dataset/pd.py if not already processed).
3. Launches the model.

## Notes
- Ensure preprocessing is run before any training script; otherwise run_graphsage.py will raise FileNotFoundError.
- A^2 + A currently adds user-user and item-item 2-hop similarity edges (co-interaction); no new cross-type edges are added.
- If Parquet engines are missing, pd.py will fall back to CSV.
- For large graphs consider increasing two-hop minimum count to reduce dense similarity edges.

## Repro / Extensibility
- Adjust hidden/out dimensions and training hyperparameters via run_graphsage.py arguments.
- Add new datasets by extending pd.py with additional filename patterns.
- To experiment with different neighbor aggregators replace SAGEEncoder or implement a custom MessagePassing layer.

## Troubleshooting
Missing Parquet engine: install pyarrow or fastparquet.
OOM on GPUs: reduce batch size or number of negative samples.
Poor metrics: try --use-a2 or tune learning rate / epochs.
