import sys
import numpy as np
import pandas as pd
from LogisticReg import LogisticRegression, select_data
from utils import prepare_model, SELECTED_FEATURES


def get_y_predict(X, log_model):
    if log_model.scaler is None or log_model.houses is None:
        raise ValueError("Model must be preparee before prediction")
    
    X.dropna()
    X_scaled = log_model.scaler.transform(X)
    X_scaled = np.insert(X_scaled, 0, 1, axis=1)
    y_predicted = {}
    for house_name in log_model.houses:
        w = log_model.weights[house_name]
        z = X_scaled @ w
        y_predicted[house_name] = log_model._sigmoid(z)
    proba_array = np.hstack([y_predicted[house] for house in log_model.houses])
    return proba_array

def predict(X, log_model):
    probabilities = get_y_predict(X, log_model)
    predicted_indices = np.argmax(probabilities, axis=1)
    predictions = np.array([log_model.houses[idx] for idx in predicted_indices])
    
    return predictions


def main():
    if len(sys.argv) < 3:
        print("How to use this program: python3 logreg_predict.py <test_file> <weights_file>")
        print("Example: -----python3 logreg_predict.py dataset_test.csv weights.csv-----")
    else:
        try:
            model = prepare_model()
            test_file_path = sys.argv[1]
            weights_path = sys.argv[2]

            test_data = pd.read_csv(test_file_path, index_col = "Index")
           
            df = pd.read_csv(weights_path)
            for house_name in model.houses:
                model.weights[house_name] = df[house_name].values.reshape(-1, 1)

            X_test = test_data[SELECTED_FEATURES]

            predictions = predict(X_test, model)

            if 'Index' in test_data.columns:
                indices = test_data['Index']
            else:
                indices = test_data.index

            result_df = pd.DataFrame({
                'Index': indices,
                'Hogwarts House': predictions
            })
            result_df.to_csv('houses.csv', index=False)

        except FileNotFoundError:
            print("dataset_test.csv not found.")


if __name__ == "__main__":
    main()