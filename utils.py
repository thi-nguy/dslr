def select_high_corr_feature(df, threshold=0.7):
    corr_matrix = df.corr()
    high_corr = []
    n = len(corr_matrix.columns)

    for i in range(n):
        for j in range(i+1, n):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                high_corr.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    return high_corr

def get_feature_set(high_corr):
    from collections import Counter
    
    # Đếm số lần xuất hiện của mỗi feature
    feature_count = Counter()
    for feature_1, feature_2, corr_value in high_corr:
        feature_count[feature_1] += 1
        feature_count[feature_2] += 1
    
    # Từ mỗi cặp, chọn feature xuất hiện nhiều hơn
    selected = set()
    processed_pairs = set()
    
    for feature_1, feature_2, corr_value in high_corr:
        pair = tuple(sorted([feature_1, feature_2]))
        if pair not in processed_pairs:
            # Chọn feature có count cao hơn
            if feature_count[feature_1] >= feature_count[feature_2]:
                selected.add(feature_1)
            else:
                selected.add(feature_2)
            processed_pairs.add(pair)
    
    return selected