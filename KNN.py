import numpy as np

class KNN:
    def __init__(self,k):
        self.k = k
    
    def fit(self,data,labels):
        self.data = data
        self.labels = labels
    
    