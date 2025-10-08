import seaborn as sns
from utils import select_high_corr_feature, get_feature_set

def plot_pair_plot(df, selected_feature):
    selected_df = np.array(df[selected_features])
    sns.pairplot(df)

    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nPair-plot is closed")
        plt.close('all')




def main():
    df = pd.read_csv('dataset_train.csv', index_col='Index')
    numeric_df = df.select_dtypes(include=[np.number])

    high_corr = select_high_corr_feature(df)
    feature_set = get_feature_set(high_corr)

    plot_pair_plot(numeric_df, feature_set)


if __name__ == '__main__':
    main()