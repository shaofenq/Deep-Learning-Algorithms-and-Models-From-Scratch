import numpy as np


class Identity:
    
    def forward(self, Z):
    
        self.A = Z
        
        return self.A
    
    def backward(self):
    
        dAdZ = np.ones(self.A.shape, dtype="f")
        
        return dAdZ


class Sigmoid:
    
    def forward(self, Z):
    
        self.A = (1+np.exp(-Z))**(-1) # TODO
        
        return self.A
    
    def backward(self):
    
        dAdZ = self.A - (self.A)**2 # TODO
        
        return dAdZ


class Tanh:
    
    def forward(self, Z):
    
        self.A = np.tanh(Z) # TODO
        
        return self.A
    
    def backward(self):
    
        dAdZ =  1 - (self.A)**2
        
        return dAdZ


class ReLU:
    
    def forward(self, Z):
    
        self.A =  np.maximum(Z,0)
        
        return self.A
    
    def backward(self):
        copy = self.A.copy()
        copy[np.where(copy > 0)] = 1
        copy[np.where(copy <= 0)] = 0
    
        dAdZ = copy # TODO
        
        return dAdZ
        
        
