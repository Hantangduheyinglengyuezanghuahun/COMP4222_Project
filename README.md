# Reco Baselines (Popularity / PPR / ALS / GraphSAGE)

## Overview
This project provides baseline recommender models:
- Popularity (eval.py)
- Personalized PageRank (run_ppr.py)
- ALS (run_als.py)
- GraphSAGE (run_graphsage.py) with selectable adjacency:
  - A (user–item bipartite edges)
  - A^2 + A (adds user–user & item–item 2-hop similarity edges)

Data preprocessing now produces reusable graph artifacts so training does not rebuild graphs each run.

## Environment Setup

### Using virtualenv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

### Using conda
conda create -n project python==3.9
conda activate project
pip install -r requirements.txt   # (or conda equivalents)

### Optional (Parquet support)
If you want Parquet output (preferred):
pip install pyarrow fastparquet

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
