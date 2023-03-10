# GRU--LSTM
import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h):
        return self.forward(x, h)

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx

        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh

        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx

        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def forward(self, x, h):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        
        # This code should not take more than 10 lines.
        self.r = np.matmul(self.Wrx, x) + self.brx + np.matmul(self.Wrh, h) + self.brh
        self.r = self.r_act.forward(self.r)

        self.z = np.matmul(self.Wzx, x) + self.bzx + np.matmul(self.Wzh, h) + self.bzh
        self.z = self.z_act.forward(self.z)

        self.n = np.matmul(self.Wnx, x) + self.bnx + self.r * (np.matmul(self.Wnh, h) + self.bnh)
        self.n = self.h_act.forward(self.n)

        h_t = (1 - self.z) * self.n + self.z * h
        

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.
        
    
        
        return h_t
        #raise NotImplementedError

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs match the
        # initalized shapes accordingly
        
        # This code should not take more than 25 lines.
        # delta --> dLdht
        dLdzt = np.ravel(delta * (self.hidden - self.n))# flatten
        dLdzt = dLdzt * self.z_act.derivative()
        
        dLdnt = np.ravel((delta * (1 - self.z).T).T)
        dLdnt = dLdnt * self.h_act.derivative(self.n)

        
        dLdrt = dLdnt * (np.matmul(self.Wnh, self.hidden) + self.bnh).T
        dLdrt = dLdrt * self.r_act.derivative()

        

        # Transpose all calculated dWs
        self.dWrx = np.outer(self.x, dLdrt).T
        self.dWzx = np.outer(self.x, dLdzt).T
        self.dWnx = np.outer(self.x, dLdnt).T

        self.dWrh = np.outer(self.hidden, dLdrt).T
        self.dWzh = np.outer(self.hidden, dLdzt).T
        self.dWnh = np.outer(self.hidden, dLdnt * self.r).T

        self.dbrx = dLdrt
        self.dbzx = dLdzt
        self.dbnx = dLdnt

        self.dbrh = dLdrt
        self.dbzh = dLdzt
        self.dbnh = dLdnt * self.r

        dx = np.matmul(self.Wzx.T, dLdzt) + np.matmul(self.Wrx.T, dLdrt) + np.matmul(dLdnt, self.Wnx) #step6
        # reshape to dx(input)
        dx = np.reshape(dx, (1, self.d))

        dh = np.matmul(dLdzt, self.Wzh) + np.matmul(dLdrt, self.Wrh) + np.ravel(delta) * self.z + np.matmul(dLdnt * self.r, self.Wnh) # TODO
        dh = np.reshape(dh, (1, self.h))
        

        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        return dx, dh
        #raise NotImplementedError

