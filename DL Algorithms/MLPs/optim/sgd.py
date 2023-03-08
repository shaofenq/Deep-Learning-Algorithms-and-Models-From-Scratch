import numpy as np

class SGD:

    def __init__(self, model, lr=0.1, momentum=0):
    
        self.l   = model.layers
        self.L   = len(model.layers)
        self.lr  = lr
        self.mu  = momentum
        self.v_W = [np.zeros(self.l[i].W.shape, dtype="f") for i in range(self.L)]
        self.v_b = [np.zeros(self.l[i].b.shape, dtype="f") for i in range(self.L)]
    
    def step(self):
    
        for i in range(self.L):
        
            if self.mu == 0:
        
                self.l[i].W = self.l[i].W - self.lr*self.l[i].dLdW# TODO
                self.l[i].b = self.l[i].b - self.lr*self.l[i].dLdb# TODO
                
            else:
        
                self.v_W[i] = self.mu*self.v_W[i] + self.l[i].dLdW # TODO
                self.v_b[i] = self.mu*self.v_b[i] + self.l[i].dLdb # TODO
                self.l[i].W = self.l[i].W - self.lr*self.v_W[i] # TODO
                self.l[i].b = self.l[i].b - self.lr*self.v_b[i] # TODO
    
        return None
    
    
    
    
"""class PseudoModel:
    def __init__(self):
        self.layers = [ mytorch.nn.Linear(3,2) ]
        self.f = [ mytorch.nn.ReLU() ]
    def forward(self, A):
        return NotImplemented
    def backward(self):
        return NotImplemented

# Create Example Model
pseudo_model = PseudoModel()
pseudo_model.layers[0].W = np.ones((3,2))
pseudo_model.layers[0].dLdW = np.ones((3,2))/10
pseudo_model.layers[0].b = np.ones((3,1))
pseudo_model.layers[0].dLdb = np.ones((3,1))/10
print("W\n\n", pseudo_model.layers[0].W)
print("W\n\n", pseudo_model.layers[0].b)
# Test Example Models
optimizer = SGD(pseudo_model, lr=1)
optimizer.step()
print("W\n\n", pseudo_model.layers[0].W)
print("W\n\n", pseudo_model.layers[0].b)"""