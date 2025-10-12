import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

def select_data(original_df, selected_features):
    data = original_df.dropna()
    features = data[selected_features]
    labels = np.array(data.loc[:,"Hogwarts House"])
    
    return features, labels

class LogisticRegression(object):
    def __init__(self, learning_rate=0.05, n_iterations=2000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = {}
        self.losses = {}
        self.scaler = None
        self.houses = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _compute_loss(self, y, y_pred):
        m = len(y)
        cost = (-1/m) * (y.T @ np.log(y_pred) + (1 - y).T @ np.log(1 - y_pred))
        return cost

    def _compute_gradient(self, X, y, y_pred):
        m = len(y)
        gradient = (1/m) * (X.T @ (y_pred - y))
        return gradient

    def scaling(self, x):
        self.scaler = RobustScaler()
        x_scaled = self.scaler.fit_transform(x)
        return x_scaled
    
    def set_houses(self, y):
        self.houses = np.unique(y)

    def fit(self, X, y):
        X_scaled = scaling(X)
        X_scaled = np.insert(X_scaled, 0, 1, axis=1)

        m, n = X_scaled.shape
        self.set_houses(y)

        for house_name in self.houses:
            y_binary = np.where(y == house_name, 1, 0).reshape(-1, 1)
            w = np.zeros((n, 1)) # Create an array of 0 for initial weights
            self.losses[house_name] = []
            for i in range(self.n_iterations):
                # ŷ = σ(X_scaledw)
                z = X_scaled @ w
                y_predict = self._sigmoid(z)
                # ∇J(w) = 1/m × X_scaled^T(ŷ - y)
                loss = self._compute_loss(y_binary, y_predict)
                if isinstance(loss, np.ndarray):
                    loss = loss.item()
                self.losses[house_name].append(loss)

                gradient = self._compute_gradient(X_scaled, y_binary, y_predict)

                # Update Rule: w := w - α × ∇J(w)
                w -= self.learning_rate * gradient

            self.weights[house_name] = w
        return self.weights

    def plot_loss(self):
        plt.figure(figsize=(12, 6))
        
        for class_name, loss_history in self.losses.items():
            steps = range(len(loss_history))
            plt.plot(steps, loss_history, label=class_name, linewidth=2)
        
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss over Iterations', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nTraining Loss plot is closed")
            plt.close('all')

    def save_weights(self, filename='weights.csv', feature_names=None):
        feature_names = ['Bias'] + feature_names
        weight_dict = {'Feature': feature_names}
        for house_name, weight in self.weights.items():
            weight_dict[house_name] = weight.flatten()
        
        df = pd.DataFrame(weight_dict)
                
        df.to_csv(filename, index=False)
        print(f'Weights saved to {filename}')
