from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional, Union
import numpy as np

class ThermometerEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns : List[int], n_bits : int, min_value : Optional[Union[List[float], float]] = None, 
                 max_value : Optional[Union[List[float], float]] = None):
        self.columns = columns
        self.n_bits = n_bits
        self.min_value = min_value
        self.max_value = max_value
        self._bins = {}
    
    @property
    def min_value(self):
        return self._min_value

    @min_value.setter
    def min_value(self, value):
        if isinstance(value, list) or value is None:
             self._min_value = value
        else:
            self._min_value = [value] * len(self.columns)
    
    @property
    def max_value(self):
        return self._max_value

    @max_value.setter
    def max_value(self, value):
        if isinstance(value, list) or value is None:
             self._max_value = value
        else:
            self._max_value = [value] * len(self.columns)
    
    
    def fit(self, x, y=None):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        
        if not self.min_value:
            self.min_value = []
            for c in self.columns:
                self.min_value.append(np.min(x[:, c]))
        
        if not self.max_value:
            self.max_value = []
            for c in self.columns:
                self.max_value.append(np.max(x[:, c]))
        
        for i, c in enumerate(self.columns):            
            self._bins[c] =  np.histogram_bin_edges([], bins=self.n_bits, range=(self.min_value[i], self.max_value[i]))
        
        return self
                
    
    def transform(self, x):
        x_encoded = np.array([])
        n_columns = x.shape[1]
        for c in range(n_columns):
            if c in self.columns:
                column_encoded = np.zeros(shape=(x.shape[0], self.n_bits))
                for i in range(self.n_bits):
                    mask = (x[:, c] > self._bins[c][i])
                    column_encoded[mask, i] = 1
            else:
                column_encoded = x[:, c].reshape(-1, 1)
            
            if x_encoded.size == 0:                
                x_encoded = column_encoded
            else:
                x_encoded = np.hstack((x_encoded, column_encoded))
        
        return x_encoded      
