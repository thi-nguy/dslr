import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler


class LogisticRegression(object):
    def __init__(self, weights=[], learning_rate=0.0005, epochs=300):
        self.learning_rate = learning_rate
        self.number_of_epochs = epochs
        self.weight_array = weights

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _loss_function(self, weights, X, y):
        y_pred = self._sigmoid(X @ weights)
        cost = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return cost

    def _select_features(self, original_data):
        data = original_data.dropna()
        selected_features = ["Charms", "Herbology", "Ancient Runes", "Astronomy", "Divination"] # To be confirmed by the two previous parts
        features = np.array(data[selected_features])
        labels = np.array(data.loc[:,"Hogwarts House"])
        
        return features, labels

    def fit(self, data):
        X, y = self._select_features(data)
        # scaler = RobustScaler()
        # X_scaled = scaler.fit_transform(X)
        X_scaled = X
        X_scaled = np.insert(X_scaled, 0, 1, axis=1)


        m, n = X_scaled.shape
        print(m, n)

        epoch_array = [i for i in range(0, self.number_of_epochs)]

        for house_name in np.unique(y):
            y_binary = np.where(y == house_name, 1, 0) #syntax_scaled: np.where(condition, value_if_condition_true, value_if_condition_false)
            w = np.zeros(n) # Create an array of 0 for initial weights
            cost_array_over_epochs = []
            for i in range(self.number_of_epochs):
                # ŷ = σ(X_scaledw)
                z = X_scaled @ w
                y_predict = self._sigmoid(z)
                # ∇J(w) = 1/m × X_scaled^T(ŷ - y)
                gradient = (1/m) * X_scaled.T @ (y_predict - y_binary)
                # Update Rule: w := w - α × ∇J(w)
                w = w - self.learning_rate * gradient

                cost = self._loss_function(w, X_scaled, y_binary)
                cost_array_over_epochs.append(cost)
                # if i % 100 == 0:
                #     print(f"Iteration {i}: Loss = {loss:.4f}")
            plt.plot(epoch_array, cost_array_over_epochs, label=house_name)
            plt.xlabel("epoch")
            plt.ylabel("cost")
            plt.legend()
            plt.show()
            # print(house_name, w)

        return w



    
    def _predict_one(self, x_scaled):
        return max((x.dot(w), c) for w, c in self.w)[1]


    def predict(self, X):
        return [self._predict_one(i) for i in np.insert(X, 0, 1, axis=1)]
    
    
    def score(self,data):
        X, y = self._use_chosen_features(data)
        return sum(self.predict(X) == y) / len(y)   

    
if __name__ == "__main__":
    try:
        data = pd.read_csv("dataset_train.csv", index_col = "Index")
    except:
        print("dataset_train.csv not found.")
    weights = LogisticRegression().fit(data)
    # print(weights)
    # np.save("weights", weights)
    # print("Accuracy score:", LogisticRegression().score(data))