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
== Periodic Eval (epoch 10) ==
          P@10      R@20   NDCG@10
mean  0.004244  0.067073  0.021791
std   0.020160  0.250150  0.114756
== Periodic Eval (epoch 20) ==
          P@10      R@20   NDCG@10
mean  0.004481  0.070872  0.023196
std   0.020688  0.256613  0.118881
== Periodic Eval (epoch 30) ==
          P@10      R@20   NDCG@10
mean  0.004755  0.074270  0.024886
std   0.021281  0.262212  0.123832
== Periodic Eval (epoch 40) ==
          P@10     R@20   NDCG@10
mean  0.004755  0.07464  0.024924
std   0.021281  0.26281  0.123842
== Periodic Eval (epoch 50) ==
          P@10      R@20   NDCG@10
mean  0.004851  0.075399  0.025507
std   0.021484  0.264036  0.125411
== Periodic Eval (epoch 60) ==
          P@10      R@20   NDCG@10
mean  0.004948  0.076634  0.025992
std   0.021687  0.266011  0.126663
== Periodic Eval (epoch 70) ==
          P@10      R@20   NDCG@10
mean  0.004950  0.075980  0.025904
std   0.021692  0.264967  0.126232
== Periodic Eval (epoch 80) ==
          P@10      R@20   NDCG@10
mean  0.004966  0.076697  0.026127
std   0.021725  0.266112  0.127155
== Periodic Eval (epoch 90) ==
          P@10      R@20   NDCG@10
mean  0.004961  0.076096  0.026018
std   0.021714  0.265153  0.126582
== Periodic Eval (epoch 100) ==
          P@10      R@20   NDCG@10
mean  0.004926  0.076096  0.026041
std   0.021641  0.265153  0.127171
== Periodic Eval (epoch 110) ==
          P@10      R@20   NDCG@10
mean  0.004874  0.074619  0.025639
std   0.021533  0.262776  0.125753
== Periodic Eval (epoch 120) ==
          P@10      R@20   NDCG@10
mean  0.004999  0.076381  0.026200
std   0.021792  0.265608  0.127064
== Periodic Eval (epoch 130) ==
          P@10      R@20   NDCG@10
mean  0.004861  0.075188  0.025514
std   0.021504  0.263696  0.125412
== Periodic Eval (epoch 140) ==
          P@10      R@20   NDCG@10
mean  0.005005  0.077489  0.026389
std   0.021805  0.267367  0.128017
== Periodic Eval (epoch 150) ==
          P@10      R@20   NDCG@10
mean  0.004902  0.076033  0.025724
std   0.021591  0.265052  0.126025
== Periodic Eval (epoch 160) ==
          P@10      R@20   NDCG@10
mean  0.004955  0.076391  0.026476
std   0.021700  0.265625  0.129033
== Periodic Eval (epoch 170) ==
          P@10      R@20   NDCG@10
mean  0.004979  0.077193  0.026396
std   0.021751  0.266899  0.128237
== Periodic Eval (epoch 180) ==
          P@10      R@20   NDCG@10
mean  0.004909  0.075843  0.026095
std   0.021606  0.264748  0.127844
== Periodic Eval (epoch 190) ==
          P@10      R@20   NDCG@10
mean  0.004943  0.076782  0.026262
std   0.021676  0.266246  0.128208
== Periodic Eval (epoch 200) ==
          P@10      R@20   NDCG@10
mean  0.005046  0.077436  0.026692
std   0.021890  0.267284  0.129062
== Periodic Eval (epoch 210) ==
          P@10      R@20   NDCG@10
mean  0.004915  0.076486  0.026072
std   0.021619  0.265776  0.127638
== Periodic Eval (epoch 220) ==
          P@10      R@20   NDCG@10
mean  0.004841  0.075568  0.025649
std   0.021462  0.264307  0.126340
== Periodic Eval (epoch 230) ==
          P@10      R@20   NDCG@10
mean  0.004896  0.075421  0.026014
std   0.021580  0.264070  0.127580
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
mean  0.004448  0.069479  0.023064
std   0.020616  0.254269  0.118823

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

== GraphSAGE (mean/std) ==
          P@10      R@20   NDCG@10
mean  0.004915  0.075779  0.026078
std   0.021619  0.264646  0.127370

python run_graphsage.py --category Video_Games --dataset-dir data/loaded_data --epoch 300 --lr 1e-3 --hidden 128 --out 64 --negs 100 --eval-during-training --eval-every 10

== GraphSAGE (mean/std) ==
          P@10      R@20   NDCG@10
mean  0.004413  0.068118  0.023354
std   0.020539  0.251950  0.121043

# Try different alpha and iters for ppr 

python run_ppr_sparse.py --alpha <your-alpha> --iters <your-iter>

for alpha in [0.1, 0.15, 0.2, 0.5, 0.8] and iter in [40, 50, 100]. Change one variable at once, with the other one using default. 7 experiments in total. 

Results:

|  \alpha |  0.1  |  0.15  |  0.2  |  0.5  |  0.8  |
| --------| ----- |  ----- |------ | ----- | ----- |
|P@10|0.005233|0.005484|0.005516|0.005386|0.005337|
|R@20|0.078639|0.081773|0.082744|0.080602|0.079230|
|ndcg@20|0.028585|0.029892|0.030161|0.029579|0.029213|

|  iter|  40  |  50 | 100 |
| --------| ----- |  ----- | ------|
|P@10|0.005483|0.005481|0.005479|
|R@20|0.081626|0.081657|0.081647|
|ndcg@20|0.029883|0.029876|0.029869|

# Run ppr with the combination of the best resul above

# fuse graphsage with the ppr (The checkpoints can be found in google drive)
run fuse_graphsage_ppr.py --category Video_Games --gs-ckpt checkpoints/Video_Games_graphsage_A_withRating_noTextUsers_noTextItems_n2v128_fusionOut256_fusionHid256_drop0p20_hidden128_out64_negs5_final_3999.pt --ppr-scores <path-to-your-ppr-checkpoints> --ppr-meta <path-to-your-ppr-meta> --use-rating --norm minmax --filter-seen --gamma-step 0.1




