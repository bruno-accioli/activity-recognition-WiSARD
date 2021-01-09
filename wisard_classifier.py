from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from wisardpkg import Wisard
import numpy as np


class WisardClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, address_size=10, bleaching_activated=True,
                 ignore_zero=False, complete_addressing=True,
                 verbose=False):
        self.address_size = address_size
        self.bleaching_activated = bleaching_activated
        self.ignore_zero = ignore_zero
        self.complete_addressing = complete_addressing
        self.verbose = verbose
    
    def fit(self, X, y):
        self.base_model_ = Wisard(self.address_size, 
                                  bleachingActivated=self.bleaching_activated,
                                  ignoreZero=self.ignore_zero,
                                  completeAddressing=self.complete_addressing,
                                  verbose=self.verbose,
                                  returnClassesDegrees=True)
        X = self._transform_X(X)
        y = self._transform_y(y)
        self.classes_ = unique_labels(y)
        self.base_model_.train(X, y)
        return self
    
    def predict(self, X):
        X = self._transform_X(X)
        
        check_is_fitted(self, 'base_model_')
        
        X = self._transform_X(X)
        predictions = self.base_model_.classify(X)
        predictions = np.array([p['class'] for p in predictions]).reshape(-1, 1)
        return predictions
    
    def predict_proba(self, X):
        X = self._transform_X(X)
        
        check_is_fitted(self, 'base_model_')
        
        predictions = self.base_model_.classify(X)
        probas = []
        for c in self.classes_:
            pred_class = [p_class['degree'] for p in predictions for p_class in p['classesDegrees'] if p_class['class'] == c]
            pred_class = np.array(pred_class).reshape(-1,1)
            probas.append(pred_class)
        probas = np.hstack(probas)
        return probas
    
    def score(self, X, y, sample_weight=None):
        y = self._transform_y(y)
        return super().score(X, y, sample_weight)
    
    def _transform_X(self, X):
        if isinstance(X, list):
            pass
        elif isinstance(X, np.ndarray):
            X = X.tolist()
        else:
            X = np.array(X).tolist()
        return X
    
    def _transform_y(self, y):
        if isinstance(y, list):
            pass
        elif isinstance(y, np.ndarray):
            y = y.astype(str).reshape(-1,).tolist()
        else:
            y = np.array(y).astype(str).reshape(-1,).tolist()
        return y