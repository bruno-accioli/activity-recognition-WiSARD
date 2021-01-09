from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from typing import List, Optional, Union
import numpy as np

class ThermometerEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns : List[int] = None, 
                 n_bits : Union[int, List[int]] = 10, 
                 quantile_based : bool = False,
                 min_value : Optional[Union[List[float], float]] = None, 
                 max_value : Optional[Union[List[float], float]] = None):
        self.columns = columns
        self.n_bits = n_bits
        self.quantile_based = quantile_based
        self.min_value = min_value
        self.max_value = max_value

    
    def fit(self, x: Union[list, np.ndarray], y=None):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        
        # parameter validation
        if not self.columns:
            self.columns = list(range(x.shape[1]))
        
        self.min_value_by_column_ = self._validate_value(self.min_value)
        self.max_value_by_column_ = self._validate_value(self.max_value)
        
        if isinstance(self.n_bits, list):
            self.n_bits_by_column_ = self.n_bits
        else:
            self.n_bits_by_column_ = [self.n_bits] * x.shape[1]
        
        # fitting
        if not self.min_value_by_column_:
            self.min_value_by_column_ = []
            for c in self.columns:
                self.min_value_by_column_.append(np.min(x[:, c]))
        
        if not self.max_value_by_column_:
            self.max_value_by_column_ = []
            for c in self.columns:
                self.max_value_by_column_.append(np.max(x[:, c]))
        
        self.possible_values_ = dict()
        self.bins_ = dict()
        for i, c in enumerate(self.columns):
            if self.quantile_based:
                self.bins_[c] = np.unique(np.quantile(x[:, c], np.linspace(0, 1, self.n_bits_by_column_[c]+1), 
                                                      interpolation='higher'))
            else:
                self.bins_[c] =  np.histogram_bin_edges([], bins=self.n_bits_by_column_[c], 
                                                        range=(self.min_value_by_column_[i], 
                                                               self.max_value_by_column_[i]))
            self.possible_values_[c] = [((i)*[1] + (self.n_bits_by_column_[c]-i)*[0]) for i in range(self.n_bits_by_column_[c]+1)]
            self.possible_values_[c] = np.array(self.possible_values_[c])
        return self
                
    
    def transform(self, x: Union[list, np.ndarray]):
        check_is_fitted(self, 'bins_')
        
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
                levels = np.digitize(column, bins=self.bins_[c][:-1], 
                                     right=True)
                column_encoded = self.possible_values_[c][levels]
                encoded_features.append(column_encoded)
            else:
                encoded_features.append(column)
        self.encoded_features = encoded_features
        
        x_encoded = np.hstack(encoded_features)
        
        return x_encoded
    
    def _validate_value(self, value):
        if isinstance(value, list) or value is None:
             return value
        else:
            return [value] * len(self.columns)
 