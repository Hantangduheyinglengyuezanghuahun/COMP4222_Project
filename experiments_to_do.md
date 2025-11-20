# Graphsage
## (1.1) Base (No rating, no a2 hop, no comment embedding, n2v 128)
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3 --hidden 128 --out 64

== GraphSAGE (mean/std) ==
          P@10      R@20   NDCG@10
mean  0.004711  0.073363  0.024682
std   0.021187  0.260732  0.123383

## (1.2) Use a2-hop
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3 --use-a2

== GraphSAGE (mean/std) ==
          P@10      R@20   NDCG@10
mean  0.004562  0.070893  0.023601
std   0.020866  0.256648  0.119724

## (1.3) use 64 out dim for n2v
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3 --use-a2 --n2v-dim 64 

== GraphSAGE (mean/std) ==
          P@10      R@20   NDCG@10
mean  0.004148  0.06753  0.021108
std   0.019941  0.250952  0.112595

## (1.4) Use rating
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3  --use-rating --fusion-out 256 --fusion-hidden 256

== GraphSAGE (mean/std) ==
          P@10      R@20   NDCG@10
mean  0.004736  0.074154  0.024545
std   0.021241  0.262023  0.12226

## (1.5) change hidden and out
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3 --hidden 128 --out 128

== GraphSAGE (mean/std) ==
          P@10      R@20   NDCG@10
mean  0.004748  0.073722  0.025326
std   0.021266  0.261319  0.126101

## (1.6) use user comment embedding only
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3  --use-rating --use-comment-user

== GraphSAGE (mean/std) ==
          P@10      R@20   NDCG@10
mean  0.004116  0.064868  0.021352
std   0.019865  0.246294  0.114591

## (1.7) use itme comment embedding only
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3  --use-rating --use-comment-item


== GraphSAGE (mean/std) ==
          P@10      R@20   NDCG@10
mean  0.004694  0.073859  0.024520
std   0.021151  0.261542  0.122596


## (1.8) use both user and item
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3  --use-rating --use-comment-item
--use-comment-user
## (1.9) use a2-hop with rating
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3 --use-a2 --use-rating --fusion-out 256 --fusion-hid 256

== GraphSAGE (mean/std) ==
          P@10      R@20   NDCG@10
mean  0.004619  0.073869  0.023401
std   0.02099  0.261559  0.122620

## (1.10) train for 1000 epoch with lr to be 1e-3
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 1000 --lr 1e-3 --hidden 128 --out 64

== GraphSAGE (mean/std) ==
          P@10      R@20   NDCG@10
mean  0.004882  0.075737  0.025559
std   0.021549  0.264578  0.125298

## (1.11) train for 1000 epoch with lr to be 5e-4
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 1000 --lr 5e-4 --hidden 128 --out 64

== GraphSAGE (mean/std) ==
          P@10      R@20   NDCG@10
mean  0.004732  0.073078  0.024508
std   0.021232  0.260266  0.122052

## (1.12) use both user and item comment with 1000 epochs (ZHANG Xingjian)
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 1000 --lr 1e-4  --use-rating --use-comment-item --use-comment-user --batch 512 --hidden 256 --out 256 --fusion-out 256 --fusion-hid 256

# Ablation Study
## Run the pipeline and evaluate every 10 epochs. (Codes need to be changed slightly)
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 300 --lr 1e-3 --hidden 128 --out 64 --eval-during-training --eval-every 10

# Try different hidden and out
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3 --hidden 128 --out 128

== GraphSAGE (mean/std) ==
          P@10      R@20   NDCG@10
mean  0.004793  0.074260  0.025079
std   0.021362  0.262194  0.124225

python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3 --hidden 64 --out 64

== GraphSAGE (mean/std) ==
          P@10      R@20   NDCG@10
mean  0.004377  0.069078  0.022748
std   0.020459  0.253589  0.118003

python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3 --hidden 64 --out 128

== GraphSAGE (mean/std) ==
          P@10      R@20   NDCG@10
mean  0.004375  0.069321  0.022717
std   0.020454  0.254001  0.117759

# Try different dropout from [0.1, 0.2, 0.3, 0.4, 0.5]
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3 --hidden 128 --out 64 --dropout <your_drop_out_rate>

== GraphSAGE (mean/std) ==
          P@10      R@20   NDCG@10
mean  0.004298  0.067189  0.022039
std   0.020282  0.250351  0.115321

== GraphSAGE (mean/std) ==
          P@10      R@20   NDCG@10
mean  0.004724  0.073194  0.024777
std   0.021216  0.260456  0.123834

== GraphSAGE (mean/std) ==
          P@10      R@20   NDCG@10
mean  0.004752  0.073605  0.025010
std   0.021275  0.261129  0.124338

== GraphSAGE (mean/std) ==
          P@10      R@20   NDCG@10
mean  0.004638  0.071115  0.024427
std   0.021031  0.257018  0.123237

== GraphSAGE (mean/std) ==
          P@10      R@20   NDCG@10
mean  0.004679  0.072339  0.024605
std   0.021119  0.259050  0.123536

# Try different dimensions of node2vec embedding
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 120 --lr 1e-3 --hidden 128 --out 64 --n2v-dim 64
(The default is 128, run dataset/precompute_node2vec.py first)

== GraphSAGE (mean/std) ==
          P@10      R@20   NDCG@10
mean  0.004776  0.074576  0.025004
std   0.021326  0.262708  0.124272

# Try different numbers of negs, save evaluate every 10 epoch
python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 300 --lr 1e-3 --hidden 128 --out 64 --negs 1 --eval-during-training --eval-every 10

== GraphSAGE (mean/std) ==
          P@10      R@20   NDCG@10
mean  0.004742  0.074028  0.024615
std   0.021255  0.261817  0.122466


python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 300 --lr 1e-3 --hidden 128 --out 64 --negs 10 --eval-during-training --eval-every 10

python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 300 --lr 1e-3 --hidden 128 --out 64 --negs 10 --eval-during-training --eval-every 10

# Try different alpha and iters for ppr 

python run_ppr_sparse.py --alpha <your-alpha> --iters <your-iter>

for alpha in [0.1, 0.15, 0.2, 0.5, 0.8] and iter in [40, 50, 100]. Change one variable at once, with the other one using default. 7 experiments in total. 

Results:

|  \alpha |  0.1  |  0.15  |  0.2  |  0.5  |  0.8  |
| --------| ----- |  ----- |------ | ----- | ----- |
|P@10||||||
|R@20||||||
|ndcg@20||||||

|  iter|  40  |  50 | 100 |
| --------| ----- |  ----- | ------|
|P@10|||
|R@20|||
|ndcg@20|||

# Run ppr with the combination of the best resul above

# fuse graphsage with the ppr (The checkpoints can be found in google drive)
run fuse_graphsage_ppr.py --category Video_Games --gs-ckpt checkpoints/Video_Games_graphsage_A_withRating_noTextUsers_noTextItems_n2v128_fusionOut256_fusionHid256_drop0p20_hidden128_out64_negs5_final_3999.pt --ppr-scores <path-to-your-ppr-checkpoints> --ppr-meta <path-to-your-ppr-meta> --use-rating --norm minmax --filter-seen --gamma-step 0.1




