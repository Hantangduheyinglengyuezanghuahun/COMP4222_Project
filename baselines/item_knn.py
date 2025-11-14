import numpy as np, pandas as pd

def fit_knn(data, sims=[], lmbd=10, alpha=0.5, n_sims=5000, min_cooccurrence=3):
    seen = data.groupby('user_id')['item_id'].apply(set).to_dict()

    itemids = data['item_id'].unique()
    item_to_idx = {item: idx for idx, item in enumerate(itemids)}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    n = len(itemids)
    
    userids = data['user_id'].unique()
    user_to_idx = {user: idx for idx, user in enumerate(userids)}
    
    data = data.copy()
    data['item_idx'] = data['item_id'].map(item_to_idx)
    data['user_idx'] = data['user_id'].map(user_to_idx)
    
    supp = data.groupby('user_idx').size()
    user_offsets = np.zeros(len(supp) + 1, dtype=np.int32)
    user_offsets[1:] = supp.cumsum()
    index_by_user = data.sort_values('user_idx').index.values
    
    item_popularity = data.groupby('item_idx').size()
    item_offsets = np.zeros(n + 1, dtype=np.int32)
    item_offsets[1:] = item_popularity.cumsum()
    index_by_items = data.sort_values('item_idx').index.values
    
    sims = {}
    
    for i in range(n):
        if item_popularity[i] < 2:  
            continue
            
        iarray = np.zeros(n)
        start = item_offsets[i]
        end = item_offsets[i + 1]
        
        users_of_i = set()
        for e in index_by_items[start:end]:
            uidx = data.loc[e, 'user_idx']
            users_of_i.add(uidx)
        
        for uidx in users_of_i:
            ustart = user_offsets[uidx]
            uend = user_offsets[uidx + 1]
            user_events = index_by_user[ustart:uend]
            for event_idx in user_events:
                j = data.loc[event_idx, 'item_idx']
                iarray[j] += 1
        
        iarray[i] = 0  
        
        iarray[iarray < min_cooccurrence] = 0
        
        union_popularity = item_popularity[i] + item_popularity.values - iarray
        union_popularity[union_popularity == 0] = 1  
        
        cosine_sim = iarray / np.sqrt(item_popularity[i] * item_popularity.values)
        jaccard_sim = iarray / union_popularity
        combined_sim = 0.7 * cosine_sim + 0.3 * jaccard_sim
        
        norm = np.power((item_popularity[i] + lmbd), alpha) * np.power((item_popularity.values + lmbd), (1.0 - alpha))
        norm[norm == 0] = 1
        iarray = combined_sim / norm
        
        indices = np.argsort(iarray)[-1:-1-n_sims:-1]
        valid_indices = indices[iarray[indices] > 0] 
        
        if len(valid_indices) > 0:
            original_item = idx_to_item[i]
            similar_items = [idx_to_item[idx] for idx in valid_indices]
            similar_scores = iarray[valid_indices]
            sims[original_item] = pd.Series(data=similar_scores, index=similar_items)
    
    return sims, seen

def recommend_knn(sim,seen,user_ids,K=200):
    ranking = {}
    
    for user_id in user_ids:
        if user_id not in seen:
            ranking[user_id] = []
            continue
            
        user_seen = seen[user_id]
        candidate_scores = {}
        
        for seen_item in user_seen:
            if seen_item not in sim:
                continue
                
            similar_items = sim[seen_item]
            for item_id, score in similar_items.items():
                if item_id in user_seen:
                    continue
                    
                if item_id in candidate_scores:
                    candidate_scores[item_id] += score
                else:
                    candidate_scores[item_id] = score
        
        if candidate_scores:
            top_items = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:K]
            ranking[user_id] = [item_id for item_id, score in top_items]
        else:
            ranking[user_id] = []
    
    return ranking
