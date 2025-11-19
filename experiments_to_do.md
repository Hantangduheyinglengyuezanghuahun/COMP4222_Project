# Graphsage
## (1.1) Base (No rating, no a2 hop, no comment embedding, n2v 128)
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3 --hidden 128 --out 64
          P@10      R@20   NDCG@10
mean  0.007351  0.073510  0.024458
std   0.026097  0.260974  0.122620
## (1.2) Use a2-hop
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3 --use-a2
## (1.3) use 64 out dim for n2v
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3 --use-a2 --n2v-dim 64 
## (1.4) Use rating
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3  --use-rating --fusion-out 256 --fusion-hidden 256
## (1.5) change hidden and out
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3 --hidden 128 --out 128
## (1.6) use user comment embedding only
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3  --use-rating --use-comment-user
## (1.7) use itme comment embedding only
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3  --use-rating --use-comment-item
## (1.8) use both user and item
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3  --use-rating --use-comment-item
--use-comment-user
## (1.9) use a2-hop with rating
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3 --use-a2 --use rating --fusion-out 256 --fusion-hid 256
## (1.10) train for 1000 epoch with lr to be 1e-3
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 1000 --lr 1e-3 --hidden 128 --out 64
## (1.11) train for 1000 epoch with lr to be 5e-4
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 1000 --lr 5e-4 --hidden 128 --out 64
## (1.12) use both user and item comment with 1000 epochs (ZHANG Xingjian)
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 1000 --lr 1e-4  --use-rating --use-comment-item --use-comment-user --batch 512 --hidden 256 --out 256 --fusion-out 256 --fusion-hid 256

# Ablation Study
## Run the pipeline and evaluate every 10 epochs. (Codes need to be changed slightly)
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 300 --lr 1e-3 --hidden 128 --out 64 --eval-during-training --eval-every 10

# Try different hidden and out
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3 --hidden 128 --out 128

python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3 --hidden 64 --out 64

python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3 --hidden 64 --out 128

# Try different dropout from [0.1, 0.2, 0.3, 0.4, 0.5]
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3 --hidden 128 --out 64 --dropout <your_drop_out_rate>

# Try different dimensions of node2vec embedding
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3 --hidden 128 --out 64 --n2v-dim 64
(The default is 128, run dataset/precompute_node2vec.py first)

# Try different numbers of negs, save evaluate every 10 epoch
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 300 --lr 1e-3 --hidden 128 --out 64 --negs 1 --eval-during-training --eval-every 10

python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 300 --lr 1e-3 --hidden 128 --out 64 --negs 10 --eval-during-training --eval-every 10

python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 300 --lr 1e-3 --hidden 128 --out 64 --negs 10 --eval-during-training --eval-every 10

# Try different alpha and iters for ppr
python run_ppr_sparse.py --alpha <your-alpha> --iters <your-iter>

for alpha in [0.1, 0.15, 0.2, 0.5, 0.8] and iter in [40, 50]. Change one variable at once, with the other one using default. 7 experiments in total. 

Results:

|  \alpha |  0.1  |  0.15  |  0.2  |  0.5  |  0.8  |
| --------| ----- |  ----- |------ | ----- | ----- |
|P@10||||||
|R@20||||||
|ndcg@20||||||

|  iter|  40  |  50 |
| --------| ----- |  ----- |
|P@10||
|R@20||
|ndcg@20||



