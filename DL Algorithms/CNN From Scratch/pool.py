import numpy as np
from resampling import *

class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        batch_size, in_channels, self.input_width, self.input_height = A.shape
        output_width = (self.input_width - self.kernel)//1 + 1
        output_height = (self.input_height- self.kernel)//1 + 1
        Z = np.zeros((batch_size, in_channels, output_width, output_height))
        self.max_index = {}
        # np.zeros((batch_size, in_channels, output_width, output_height))
        for batch in range(batch_size):
            for inchannel in range(in_channels):
                for i in range(output_width):
                    for j in range(output_height):
                        Z[batch,inchannel, i, j] = (A[batch,inchannel, i:i+self.kernel, j:j+self.kernel]).max()
                        index = np.argmax(A[batch,inchannel, i:i+self.kernel, j:j+self.kernel])
                        row, col = index//self.kernel, index % self.kernel
                        self.max_index[batch, inchannel, i, j] = (batch, inchannel, i+row, j+col)
                
        
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # in_channels = out_channels
        batch_size, out_channels, output_width, output_height = dLdZ.shape
        in_channels = out_channels
        dLdA = np.zeros((batch_size, in_channels, self.input_width, self.input_height))
        for batch in range(batch_size):
            for inchannel in range(in_channels):
                for i in range(output_width):
                    for j in range(output_height):
                        pos = self.max_index[batch, inchannel, i, j]
                        dLdA[pos] += dLdZ[batch,inchannel,i,j]
        return dLdA

class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        batch_size, in_channels, self.input_width, self.input_height = A.shape
        output_width = (self.input_width - self.kernel)//1 + 1
        output_height = (self.input_height- self.kernel)//1 + 1
        Z = np.zeros((batch_size, in_channels, output_width, output_height))
        for batch in range(batch_size):
            for inchannel in range(in_channels):
                for i in range(output_width):
                    for j in range(output_height):
                        Z[batch,inchannel, i, j] = (A[batch,inchannel, i:i+self.kernel, j:j+self.kernel]).mean()
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # in_channels = out_channels
        batch_size, out_channels, output_width, output_height = dLdZ.shape
        in_channels = out_channels
        dLdA = np.zeros((batch_size, in_channels, self.input_width, self.input_height))
        for batch in range(batch_size):
            for inchannel in range(in_channels):
                for i in range(output_width):
                    for j in range(output_height):
                        dLdA[batch, inchannel, i:i+self.kernel, j:j+self.kernel] += (1/(self.kernel)**2)*dLdZ[batch, inchannel, i, j]

        return dLdA

class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        
        #Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 =  MaxPool2d_stride1(kernel)#TODO
        self.downsample2d = Downsample2d(stride) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        pooled = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(pooled)
        return Z
        
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        delta_out = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(delta_out)
       
        return dLdA

class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        #Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel) #TODO
        self.downsample2d = Downsample2d(stride) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        pooled = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(pooled)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        delta_out = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(delta_out)
       
        return dLdA
