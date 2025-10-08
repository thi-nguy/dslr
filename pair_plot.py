import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sys
from utils import select_high_corr_feature, get_high_corr_feature_set

def plot_pair_plot(df, selected_feature):
    selected_df = df[list(selected_feature)]
    sns.pairplot(selected_df, height=1)
    plt.tight_layout()

    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nPair-plot is closed")
        plt.close('all')


def main():
    if len(sys.argv) > 1:
        print("This program does not need arguments")
        print("It draws pair plot for dataset_train.csv")
    else:
        try: # ! To finalize
            df = pd.read_csv('dataset_train.csv', index_col='Index')
            numeric_df = df.select_dtypes(include=[np.number])

            features = set(numeric_df.columns.tolist())
            kept_high_corr_features = set(['Astronomy', 'Herbology', 'Muggle Studies', 'Flying'])
            total_high_corr_features = get_high_corr_feature_set(select_high_corr_feature(numeric_df))
            
            feature_to_remove = total_high_corr_features - kept_high_corr_features
            feature_to_keep = features - feature_to_remove

            plot_pair_plot(numeric_df, feature_to_keep)

        except FileNotFoundError:
            print(f'File dataset_train.csv does not exist')
    

if __name__ == '__main__':
    main()