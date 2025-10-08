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