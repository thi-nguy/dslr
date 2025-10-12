import sys
import pandas as pd
from LogisticReg import LogisticRegression, select_data


def main():
    if len(sys.argv) > 1:
        print("This program does not need arguments")
        print("It trains data from dataset_train.csv")
    else:
        try:
            data = pd.read_csv("dataset_train.csv", index_col = "Index")
            selected_features = ["Flying", "Muggle Studies", "Charms", "Herbology", "Ancient Runes", "Astronomy", "Divination"] # To be confirmed by the two previous parts
            X, y = select_data(data, selected_features)
            model = LogisticRegression()
            weights = model.fit(X, y)
            model.plot_loss()
            model.save_weights('weights.csv', selected_features)
        except FileNotFoundError:
            print("dataset_train.csv not found.")


if __name__ == "__main__":
    main()