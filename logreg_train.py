import sys
import pandas as pd
from LogisticReg import LogisticRegression
from utils import prepare_model, SELECTED_FEATURES


def main():
    if len(sys.argv) > 1:
        print("This program does not need arguments")
        print("It trains data from dataset_train.csv")
    else:
        try:
            model, X, y = prepare_model()
            model.fit(X, y)
            model.plot_loss()
            model.save_weights('weights.csv', SELECTED_FEATURES)
        except FileNotFoundError:
            print("dataset_train.csv not found.")


if __name__ == "__main__":
    main()