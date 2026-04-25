import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.linear_model import LogisticRegression, SGDClassifier
from itertools import combinations
from scipy.stats import mode
from sklearn.utils import check_random_state
from BaseClassifier import BaseClassifier
from sklearn.preprocessing import add_dummy_feature

class OneVsAll(BaseEstimator, ClassifierMixin):
    def __init__(self, bin_clf: str = 'logistic_reg') -> None:
        if bin_clf not in ('logistic_reg', 'sgd'):
            raise ValueError("bin_clf must be either 'logistic_reg' or 'sgd'")
        self.bin_clf: str = bin_clf


    def fit(self, X_fit: np.ndarray, y_fit: np.ndarray):
        
        if X_fit.ndim != 2 or y_fit.ndim != 1:
            raise ValueError('Error with the dimensions of the provided data')
        
        self.num_features = X_fit.shape[1]
        self.models: list = []
        self.classes: np.ndarray = np.unique(y_fit)
        
        for y_val in self.classes:
            bin_model = LogisticRegression(max_iter=1000) if self.bin_clf == 'logistic_reg' else SGDClassifier(loss='log_loss')
            y_bin: np.ndarray = np.where(y_fit == y_val, 1, 0)
            
            self.models.append(bin_model.fit(X_fit, y_bin))
        
        return self
    
    
    def predict(self, X_pred: np.ndarray) -> np.ndarray:
        if not self.models:
            raise ValueError('The model needs to be trained first. Use fit(X, y)')
        
        if X_pred.shape[1] != self.num_features:
            raise ValueError(f'Expected {self.num_features} features, but got {X_pred.shape[1]}')
        
        models_probas: list = [m.predict_proba  (X_pred)[:,1] for m in self.models]
        y_pred: np.ndarray = self.classes[np.argmax(a=models_probas, axis=0)]
        
        return y_pred




class OneVsOne(BaseEstimator, ClassifierMixin):
    def __init__(self, bin_clf: str = 'logistic_reg'):
        if bin_clf not in ('logistic_reg', 'sgd'):
            raise ValueError("bin_clf must be either 'logistic_reg' or 'sgd'")
        
        self.bin_clf = bin_clf
    
    
    def fit(self, X_fit: np.ndarray, y_fit: np.ndarray):
        if X_fit.ndim != 2 or y_fit.ndim != 1:
            raise ValueError('Error with the dimensions of the provided data')
        
        self.num_features = X_fit.shape[1]
        classes: np.ndarray = np.unique(y_fit)
        self.models: list = []
        combos = list(combinations(classes, 2))
        
        for combo in combos:
            bin_model = LogisticRegression(max_iter=1000) if self.bin_clf == 'logistic_reg' else SGDClassifier(loss='log_loss')
            class_1 = combo[0]
            class_2 = combo[1]
            mask = (y_fit == class_1) | (y_fit == class_2)
            
            self.models.append(bin_model.fit(X_fit[mask], y_fit[mask]))
        
        return self
    
    
    def predict(self, X_pred: np.ndarray) -> np.ndarray:
        if not self.models:
            raise ValueError('The model needs to be trained first. Use fit(X, y)')
        
        if X_pred.shape[1] != self.num_features:
            raise ValueError(f'Expected {self.num_features} features, but got {X_pred.shape[1]}')
        
        predicts = [m.predict(X_pred) for m in self.models]
        y_pred: np.ndarray = mode(predicts)[0]
        
        return y_pred



class SoftMaxClassifier(BaseClassifier):
    def __init__(self, gradient_descent="batch", epochs=100, lr=0.01, random_state=None, batch_size=32):
        self.gradient_descent = gradient_descent
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state
        self.rng = check_random_state(self.random_state)
        self.batch_size = batch_size
    
    def softmax(self, logits):
        Z = logits - np.max(logits, axis=1, keepdims=True)
        exp_Z = np.exp(Z)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def one_hot(self, y: np.ndarray):
        m, = y.shape
        c = len(self.classes_)
        y_hot = np.zeros(shape=(m, c))
        y_hot[np.arange(m), y] = 1
        return y_hot
      
    def fit(self, X: np.ndarray, y: np.ndarray):
        super().fit(X, y)
        y_hot = self.one_hot(y)
        X_dummy = add_dummy_feature(X)
        
        K = len(self.classes_)
        self.W_ = self.rng.randn(K, X_dummy.shape[1]) * 0.01
        self.J_ = []
        
        if self.gradient_descent == "batch":
            self.train_batch(X_dummy, y_hot)
            
        if self.gradient_descent == "stochastic":
            self.train_stochastic(X_dummy, y_hot)
        
        if self.gradient_descent == "mini":
            self.train_mini_batch(X_dummy, y_hot)
        
        return self
    
    def predict_proba(self, X:np.ndarray):
        super().predict(X)
        X_dummy = add_dummy_feature(X)
        logits = X_dummy @self.W_.T
        p_hat = self.softmax(logits)
        return p_hat
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        p_hat = self.predict_proba(X)
        return self.classes_[np.argmax(p_hat, axis=1)]
    
    def train_batch(self, X: np.ndarray, y: np.ndarray):
        for i in range(self.epochs):
            self.train_epoch(X, y)
   
    def train_stochastic(self, X: np.ndarray, y: np.ndarray):
        m = X.shape[0]
        for i in range(self.epochs):
            idx = self.rng.choice(m)
            X_sample = X[idx].reshape(1,-1)
            y_sample = y[idx].reshape(1,-1)
            self.train_epoch(X_sample, y_sample)
    
    def train_mini_batch(self, X: np.ndarray, y: np.ndarray):
        m = X.shape[0]
        for i in range(self.epochs):
            idxs = self.rng.choice(m, size=self.batch_size, replace=False)
            X_sample = X[idxs]
            y_sample = y[idxs]
            self.train_epoch(X_sample, y_sample)
            
    def train_epoch(self, X: np.ndarray, y: np.ndarray):
        # Calculate logits
        logits = X @ self.W_.T
        logits -= np.max(logits, axis=1, keepdims=True)
        # Probability scores
        p_hat = self.softmax(logits)
        # cross entropy cost function
        J = - 1 / X.shape[0] * np.sum(np.sum( y * np.log(p_hat + 1e-15), axis=1))
        self.J_.append(J)
        # Cross entropy gradient vector
        dW = 1 / X.shape[0] * (p_hat - y).T @ X
        self.W_ -= self.lr * dW