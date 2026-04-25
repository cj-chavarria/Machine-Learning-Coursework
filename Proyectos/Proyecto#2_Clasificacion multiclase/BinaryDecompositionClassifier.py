from sklearn.linear_model import LogisticRegression
from .BaseClassifier import BaseClassifier

class BinaryDecompositionClassifier(BaseClassifier):
    def __init__(self, binary_classifier=None, n_jobs=-1):
        self.binary_classifier = binary_classifier
        self.n_jobs = n_jobs
    
    def fit(self, X, y):
        super().fit(X, y)
        
        if self.binary_classifier is None:
            self.binary_classifier = LogisticRegression(max_iter=1000)
            
    def predict(self, X):
        super().predict(X)
        
        if not hasattr(self, "models"):
            raise ValueError('The model needs to be trained first. Use fit(X, y)')