import numpy as np
import pandas as pd

from scipy.stats import zscore

'''
Time Series Data Module for managing training with PytorchLightning

Data Format:

(N,C,T)

N: Number of samples
C: Number of channels (univariate = 1, multivariate > 1)
T: Number of timepoints

'''

class ZNormalizer():
    def __init__(self,mean=0,std=1):
        self.mean = mean
        self.std = std

    def transform(self,X):

        self.mean = np.mean(X,axis=2, keepdims=True)
        self.std = np.std(X,axis=2, keepdims=True)

        # self.mean = np.mean(X,axis=(0,2), keepdims=True)
        # self.std = np.std(X,axis=(0,2), keepdims=True)

        z = (X - self.mean) / self.std
        z = np.nan_to_num(z)
        return z
        
    def fit(self,X):
        self.mean = np.mean(X,axis=2, keepdims=True)
        self.std = np.std(X,axis=2, keepdims=True)
        # self.mean = np.mean(X,axis=(0,2), keepdims=True)
        # self.std = np.std(X,axis=(0,2), keepdims=True)

        return self
    
    def fit_transform(self,X):
        self.fit(X)
        z = self.transform(X)
        z = np.nan_to_num(z)
        return z 

class MinMaxNormalizer():
    def __init__(self,min=0,max=0):
        self.min = min
        self.max = max
    def transform(self,X):
        z = (X -self.min) / (self.max - self.min)
        return z
    def fit(self,X):
        self.min = np.min(X,axis=2,keepdims=True)
        self.max = np.max(X,axis=2,keepdims=True)
        
        return self

    def fit_transform(self,X):
        self.fit(X)
        z = self.transform(X)
        return z 

class MeanNormalizer():
    def __init__(self,mean=0,min=0,max=0):
        self.mean = mean
        self.min = min
        self.mean = mean
    def transform(self,X):
        z = (X - self.mean) / (self.max - self.min)
        return z

    def fit(self,X):
        self.mean = np.mean(X,axis=2,keepdims=True)
        self.min = np.min(X,axis=2,keepdims=True)
        self.max = np.max(X,axis=2,keepdims=True)

        return self

    def fit_transform(self,X):
        self.fit(X)
        z = self.transform(X)
        return z 

class MedianNormalizer():
    def __init__(self,median=0):
        self.median=median
    def transform(self,X):
        z = X / self.median
        return z

    def fit(self,X):
        self.median = np.median(X,axis=2,keepdims=True)

    def fit_transform(self,X):
        self.fit(X)
        z = self.transform(X)
        return z 
    
class UnitNormalizer():
    def __init__(self):
        pass
    def transform(self,X):
        z = X / np.linalg.norm(X,axis=2,keepdims=True)
        return z

    def fit(self,X):
        pass

    def fit_transform(self,X):
        self.fit(X)
        z = self.transform(X)
        return z

def sigmoid(x):
    z = 1/(1 + np.exp(-x))
    return z

class SigmoidNormalizer():
    def __init__(self):
        pass
    def transform(self,X):
        z = sigmoid(X)
        return z

    def fit(self,X):
        pass

    def fit_transform(self,X):
        self.fit(X)
        z = self.transform(X)
        return z

def tanh(x):
    z = (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
    return z

class TanhNormalizer():
    def __init__(self):
        pass
    def transform(self,X):
        z = np.tanh(X)
        return z

    def fit(self,X):
        pass

    def fit_transform(self,X):
        self.fit(X)
        z = self.transform(X)
        return z

normalization_methods = {
    'zscore':ZNormalizer,
    'minmax':MinMaxNormalizer,
    'median':MedianNormalizer,
    'mean':MeanNormalizer,
    'unit':UnitNormalizer,
    'sigmoid':SigmoidNormalizer,
    'tanh':TanhNormalizer
}
    
def create_normalizer(name='zscore',X=None):

    # if name == 'zscore':
    #     norm = ZNormalizer()
    # elif name == 'minmax':
    #     norm = MinMaxNormalizer()
    # elif name == 'mean':
    #     norm = MeanNormalizer()
    # elif name == 'median':
    #     norm = MedianNormalizer()
    # elif name == 'unit':
    #     norm = UnitNormalizer()
    # elif name == 'sigmoid':
    #     norm = SigmoidNormalizer()
    # elif name == 'tanh':
    #     norm = TanhNormalizer()

    norm = normalization_methods[name]()

    if X is not None:
        X_transform = norm.fit_transform(X)
        return norm, X_transform
    else:
        return norm



# if __name__ == "__main__":

#     X = np.array([2,2,2,2,2,2]).reshape(-1,1)
#     print(zscore(X, axis=-1))