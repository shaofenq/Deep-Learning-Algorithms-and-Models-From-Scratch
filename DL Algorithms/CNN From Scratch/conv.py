# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *

class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        batch_size, in_channels, input_size = A.shape
        self.A = A
        output_size = (input_size - self.kernel_size)//1 + 1
        Z = np.zeros((batch_size, self.out_channels, output_size))

        for i in range(output_size):
            A_block = A[:, :, i: i + self.kernel_size]
            Z[:, :, i] = np.tensordot(A_block, self.W, ([1, 2], [1, 2]))
            # Z = (batch_size,out_channels, output_size)
            
        # add bias to each output chanel
        # for each output chanel, a single number is add to all nunmbers in the output_size 1-D array
        Z = np.transpose(Z, axes = [0,2,1])
        Z = np.transpose(Z + self.b, axes = [0,2,1])
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        batch_size, out_channels, output_size = dLdZ.shape
        batch_size, in_channels, input_size = (self.A).shape
        dLdA = np.zeros((batch_size, self.in_channels, input_size))
        # A (np.array): (batch_size, in_channels, input_size)
        # dLdZ (np.array): (batch_size, out_channels, output_size)
        # convolution for each outputchanel
        
        for batch in range(batch_size):
            for outchannel in range(self.out_channels):
                for inchannel in range(self.in_channels):
                    for i in range(self.kernel_size):
                        for out in range(output_size):
                            self.dLdW[outchannel, inchannel, i] += self.A[batch, inchannel, i + out] * dLdZ[batch, outchannel, out]
        
        
        
        
        
        #sum over all output_size and obatin C1 size vector for each sample
        # averaging over all samples in the batchsize(np.average(axis = 0)
        self.dLdb = np.sum(dLdZ, axis = (0,2))
        
        dLdA = np.zeros((batch_size, in_channels, input_size))
        # broadcast dLdZ input chanels times and pad each side with kernelsize -1 zeros
        for batch in range(batch_size):
            for outchannel in range(self.out_channels):
                dLdZ_broadcasted = np.tile(dLdZ[batch, outchannel, :], (in_channels,1))
                dLdZ_padded = np.pad(dLdZ_broadcasted, ((0,0),(self.kernel_size - 1, self.kernel_size - 1)), 'constant', constant_values=0)
                # flipp the corresponding weight matrix(outchanel)
                kernel_flipped = np.flip(self.W[outchannel, :, :], axis = 1)
                # and do the convolution
                # dLdZ_padded: 2D, rows->in_channels
                # kernel_flipped: in_channels, kernel_size
                for inchannel in range(self.in_channels):
                    for i in range(input_size):
                        for j in range(self.kernel_size):
                            dLdA[batch,inchannel,i] += dLdZ_padded[inchannel,i+j]*kernel_flipped[inchannel,j]

        return dLdA

class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
    
        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn) # TODO
        self.downsample1d = Downsample1d(stride) # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1
        stride_1_Z = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(stride_1_Z) # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        # TODO
        upsampled_dLdZ = self.downsample1d.backward(dLdZ)

        

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(upsampled_dLdZ)

        return dLdA


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, input_width, input_height = A.shape
        output_width = (input_width - self.kernel_size)//1 + 1
        output_height = (input_height - self.kernel_size)//1 + 1
        Z = np.zeros((batch_size, self.out_channels, output_width, output_height))

        for i in range(output_width):
            for j in range(output_height):
                A_block = A[:, :, i: i + self.kernel_size, j: j + self.kernel_size]
                Z[:, :, i, j] = np.tensordot(A_block, self.W, ([1, 2, 3], [1, 2, 3]))
            # Z = (batch_size,out_channels, output_width, output_height)
            
        # add bias to each output chanel
        # for each output chanel, a single number is add to all nunmbers in the output_size 2-D array
        for k in range(self.out_channels):
            Z[:,k,:,:] += (self.b)[k]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, out_channels, output_width, output_height = dLdZ.shape
        batch_size, in_channels, input_width, input_height = (self.A).shape
        
        # A (np.array): (batch_size, in_channels, input_width, input_height)
        # convolution for each outputchanel
        # dLdW: out_channels, in_channels, kernel_size, kernel_size
        for batch in range(batch_size):
            for outchannel in range(self.out_channels):
                for inchannel in range(self.in_channels):
                    for i in range(self.kernel_size):
                        for j in range(self.kernel_size):
                            self.dLdW[outchannel, inchannel, i, j] += np.sum(self.A[batch, inchannel, i:i + output_width, j:j+output_height] * dLdZ[batch, outchannel, :,:])

       
        self.dLdb = np.sum(dLdZ, axis = (0,2,3)) # TODO
    
        dLdA = np.zeros((batch_size, in_channels, input_width, input_height))
        # broadcast dLdZ input chanels times and pad each side with kernelsize -1 zeros
        for batch in range(batch_size):
            for outchannel in range(self.out_channels):
                dLdZ_padded = np.pad(dLdZ[batch, outchannel, :, :], ((self.kernel_size - 1,self.kernel_size - 1),(self.kernel_size - 1, self.kernel_size - 1)), 'constant', constant_values=0)
                # flipp the corresponding weight matrix(outchanel)
                # and do the convolution
                # dLdZ_padded: 2D, rows->in_channels
                # kernel_flipped: in_channels, kernel_size
                for inchannel in range(self.in_channels):
                    kernel_flipped = np.flip(self.W[outchannel, inchannel, :, :])
                    for i in range(input_width):
                        for j in range(input_height):
                            dLdA[batch,inchannel,i, j] += np.sum(dLdZ_padded[i:i + self.kernel_size, j:j+self.kernel_size]*kernel_flipped)

        return dLdA

class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels,kernel_size, weight_init_fn, bias_init_fn) # TODO
        self.downsample2d = Downsample2d(stride) # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call Conv2d_stride1
        # TODO
        stride_1_Z = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(stride_1_Z) # TODO


        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # Call downsample1d backward
        # TODO
        upsampled_dLdZ = self.downsample2d.backward(dLdZ)

        

        # Call Conv2d_stride1 backward
        dLdA = self.conv2d_stride1.backward(upsampled_dLdZ)

        return dLdA

class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv1d stride 1 and upsample1d isntance
        #TODO
        self.upsample1d = Upsample1d(upsampling_factor) #TODO
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        #TODO
        # upsample
        A_upsampled = self.upsample1d.forward(A) #TODO

        # Call Conv1d_stride1()
        Z = self.conv1d_stride1.forward(A_upsampled) #TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #TODO

        #Call backward in the correct order
        delta_out = self.conv1d_stride1.backward(dLdZ) #TODO

        dLdA = self.upsample1d.backward(delta_out)  #TODO

        return dLdA

class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels,kernel_size, weight_init_fn, bias_init_fn) #TODO
        self.upsample2d = Upsample2d(upsampling_factor)  #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # upsample
        A_upsampled = self.upsample2d.forward(A) #TODO

        # Call Conv2d_stride1()
        Z =  self.conv2d_stride1.forward(A_upsampled)#TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #Call backward in correct order
        delta_out = self.conv2d_stride1.backward(dLdZ) #TODO

        dLdA = self.upsample2d.backward(delta_out)#TODO

        return dLdA

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        self.batch_size,self.in_channels, self.in_width = A.shape

        Z =  A.reshape((self.batch_size, self.in_channels * self.in_width))# TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        dLdA = dLdZ.reshape((self.batch_size, self.in_channels, self.in_width)) #TODO

        return dLdA


