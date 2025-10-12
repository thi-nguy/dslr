import pandas as pd
from LogisticReg import LogisticRegression, select_data


SELECTED_FEATURES = ["Flying", "Muggle Studies", "Charms", "Herbology", "Ancient Runes", "Astronomy", "Divination"]

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

def get_high_corr_feature_set(high_corr):
    feature_set = set()
    for feature_1, feature_2, _ in high_corr:
        feature_set.add(feature_1)
        feature_set.add(feature_2)
    return feature_set

def prepare_model():
    data = pd.read_csv("dataset_train.csv", index_col = "Index")
    X, y = select_data(data, SELECTED_FEATURES)
    model = LogisticRegression()
    model.scaling(X)
    model.set_houses(y)
    return model