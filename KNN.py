import numpy as np

class KNN:
    def __init__(self,k):
        self.k = k
    
    def transform(self,data):
        data = data.reshape(data.shape[0],-1) # Flattening images, (60k,28,28) -> (60k,784)
        data[data > 0] = 1 # Boolean Indexing
        return data

    def fit(self,data,labels):
        self.data = data
        self.labels = labels