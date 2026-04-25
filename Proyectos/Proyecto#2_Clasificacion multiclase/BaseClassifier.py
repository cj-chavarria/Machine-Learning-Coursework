from sklearn.base import ClassifierMixin, BaseEstimator
import numpy as np

class BaseClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("Invalid input shapes")
        
        self.num_features = X.shape[1]
        self.classes_ = np.unique(y)
        
    def predict(self, X: np.ndarray):
        if X.shape[1] != self.num_features:
            raise ValueError(f'Expected {self.num_features} features, but got {X.shape[1]}')