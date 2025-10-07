from DataHandle import DataHandle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sys

def plot_high_correlation_features(df, threshold=0.7):
    corr_matrix = df.corr()
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                high_corr.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    n_pairs = len(high_corr)
    
    if n_pairs == 0:
        print(f"No correlation |correlation| >= {threshold}")
        return
    
    n_rows = 2
    n_cols = (n_pairs + n_rows - 1) // n_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15,10))
    axes = axes.flatten() if n_pairs > 1 else [axes]
    
    for idx, (feat1, feat2, corr) in enumerate(high_corr):
        axes[idx].scatter(df[feat1], df[feat2], alpha=0.5)
        axes[idx].set_xlabel(feat1)
        axes[idx].set_ylabel(feat2)
        axes[idx].set_title(f'{feat1} vs {feat2}\nCorr: {corr:.3f}')
        axes[idx].grid(True, alpha=0.3)
    
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'High Correlation Pairs (|r| >= {threshold})', fontsize=16, y=1.00)
    fig.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nScatter plots are closed")
        plt.close('all')


def main():
    df = pd.read_csv('./datasets/dataset_train.csv', index_col='Index') # To Sophie 6: In contrast with histogram, we run the data set directly here...
    numeric_df = df.select_dtypes(include=[np.number])

    correlation_matrix = numeric_df.corr()
    fig, axes = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, square=True, linewidth=1)
    axes.set_title('Correlation Heatmap')
    fig.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nHeatmap is closed")
        plt.close('all')

    plot_high_correlation_features(numeric_df) # To Sophie 7: we don't plot all 13 features scatter_plot because it's huge.


if __name__ == '__main__':
    main()