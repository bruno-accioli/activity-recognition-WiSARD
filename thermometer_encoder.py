from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional, Union
import numpy as np

class ThermometerEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns : List[int], n_bits : int, quantile_based : bool = False,
                 min_value : Optional[Union[List[float], float]] = None, 
                 max_value : Optional[Union[List[float], float]] = None):
        self.columns = columns
        self.n_bits = n_bits
        self.quantile_based = quantile_based
        self.min_value = min_value
        self.max_value = max_value
        self._bins = dict()
    
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
    
    
    def fit(self, x: Union[list, np.ndarray], y=None):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        
        if not self.min_value:
            self.min_value = []
            for c in self.columns:
                self.min_value.append(np.min(x[:, c]))
        
        if not self.max_value:
            self.max_value = []
            for c in self.columns:
                self.max_value.append(np.max(x[:, c]))
        
        for i, c in enumerate(self.columns):
            if self.quantile_based:
                self._bins[c] = np.unique(np.quantile(x, np.linspace(0, 1, self.n_bits+1), 
                                                      interpolation='higher'))
            else:
                self._bins[c] =  np.histogram_bin_edges([], bins=self.n_bits, 
                                                        range=(self.min_value[i], 
                                                               self.max_value[i]))
            self._possible_values = [((i)*[1] + (self.n_bits-i)*[0]) for i in range(self.n_bits+1)]
            self._possible_values = np.array(self._possible_values)
        return self
                
    
    def transform(self, x: Union[list, np.ndarray]):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
            
        x_encoded = []
        encoded_features = []
        n_columns = x.shape[1]

        for c in range(n_columns):
            column = x[:, c]
            if c in self.columns:
                levels = np.digitize(column, bins=self._bins[c][:-1], 
                                     right=True)
                column_encoded = self._possible_values[levels]
                encoded_features.append(column_encoded)
            else:
                encoded_features.append(column)
        self.encoded_features = encoded_features
        
        x_encoded = np.hstack(encoded_features)
        
        return x_encoded
 