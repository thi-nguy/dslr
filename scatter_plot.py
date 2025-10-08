from DataHandle import DataHandle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sys
from utils import select_high_corr_feature, get_feature_set

def plot_high_correlation_features(df, threshold=0.7):
    high_corr = select_high_corr_feature(df)
    feature_set = get_feature_set(high_corr)
    print("High correlation features: ", feature_set)
    n_pairs = len(high_corr)

    
    if n_pairs == 0:
        print(f"No correlation |correlation| >= {threshold}")
        return
    
    n_rows = 2
    n_cols = (n_pairs + n_rows - 1) // n_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15,10))
    axes = axes.flatten() if n_pairs > 1 else [axes]

    for index, (feature_1, feature_2, corr_value) in enumerate(high_corr):
        axes[index].scatter(df[feature_1], df[feature_2], alpha=0.5)
        axes[index].set_xlabel(feature_1)
        axes[index].set_ylabel(feature_2)
        axes[index].set_title(f'{feature_1} vs {feature_2}\nCorr: {corr_value:.3f}')
        axes[index].grid(True, alpha=0.3)
    
    for index in range(n_pairs, len(axes)):
        axes[index].set_visible(False)
    
    fig.suptitle(f'High Correlation Pairs (|r| >= {threshold})', fontsize=16, y=1.00)
    fig.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nScatter plots are closed")
        plt.close('all')

def plot_heat_map(df):
    correlation_matrix = df.corr()
    fig, axes = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, square=True, linewidth=1)
    axes.set_title('Correlation Heatmap')
    fig.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nHeatmap is closed")
        plt.close('all')



def main():
    df = pd.read_csv('dataset_train.csv', index_col='Index') # To Sophie 6: In contrast with histogram, we run the data set directly here...
    numeric_df = df.select_dtypes(include=[np.number])

    plot_heat_map(numeric_df)
    plot_high_correlation_features(numeric_df) # To Sophie 7: we don't plot all 13 features scatter_plot because it's huge.
    

if __name__ == '__main__':
    main()