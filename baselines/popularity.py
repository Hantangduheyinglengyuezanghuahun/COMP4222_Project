# project/baselines/popularity.py
import pandas as pd

def fit_popularity(train_df):
    pop_list = train_df['item_id'].value_counts().index.tolist()
    seen = train_df.groupby('user_id')['item_id'].apply(set).to_dict()
    return pop_list, seen

def recommend_popularity(pop, seen, users, K=200):
    rankings = {}
    for u in users:
        ban = seen.get(u, set())
        # 过滤训练看过的
        lst = [i for i in pop if i not in ban]
        rankings[u] = lst[:K]
    return rankings
