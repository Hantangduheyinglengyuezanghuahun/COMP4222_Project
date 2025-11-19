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
