import numpy as np

class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        batch_size, in_channels, input_width = A.shape
        output_width = input_width * self.upsampling_factor - (self.upsampling_factor - 1)
        Z = np.zeros((batch_size, in_channels, output_width)) # TODO
        for i in range(0, input_width):
            Z[:,:,(i)*self.upsampling_factor] = A[:,:,i]

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        batch_size, in_channels, output_width = dLdZ.shape
        input_width = (output_width - 1)//self.upsampling_factor + 1
        dLdA = np.zeros((batch_size, in_channels, input_width)) # TODO
        m = 0
        for i in range(0, output_width, self.upsampling_factor):
            dLdA[:,:,m] = dLdZ[:,:,i]
            m += 1

        return dLdA

class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        batch_size, in_channels, input_width = A.shape
        output_width = (input_width - 1)//self.downsampling_factor + 1
        Z = np.zeros((batch_size, in_channels, output_width)) # TODO
        m = 0
        for i in range(0, input_width, self.downsampling_factor):
            Z[:,:,m] = A[:,:,i]
            m += 1
        self.shape = A.shape[2]
        print("A shape is", A.shape)
        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        # need to check if outputwidth is even
        # if so we need to pad anotehr zero at the end
        # if not, keep as usual
        #print(dLdZ.shape)
        print(self.downsampling_factor)
        batch_size, in_channels, output_width = dLdZ.shape
        input_width = output_width * self.downsampling_factor - (self.downsampling_factor - 1)

        dLdA = np.zeros((batch_size, in_channels, self.shape)) # TODO
        for i in range(0, output_width):
            dLdA[:,:,(i)*self.downsampling_factor] = dLdZ[:,:,i]
        print("dLdA", dLdA.shape)
        return dLdA

class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        batch_size, in_channels, input_width, input_height = A.shape
        output_height = input_height * self.upsampling_factor - (self.upsampling_factor - 1)
        output_width = input_width * self.upsampling_factor - (self.upsampling_factor - 1)
        Z = np.zeros((batch_size, in_channels, output_width, output_height)) # TODO
        for i in range(0,input_width):
            for j in range(0, input_height):
                Z[:,:,(i)*self.upsampling_factor,(j)*self.upsampling_factor ] = A[:,:,i, j]
        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, in_channels, output_width, output_height = dLdZ.shape
        input_width =  (output_width - 1)//self.upsampling_factor + 1
        input_height =  (output_height - 1)//self.upsampling_factor + 1
        dLdA = np.zeros((batch_size, in_channels, input_width, input_height)) # TODO
        m = 0
        for i in range(0, output_width, self.upsampling_factor):
            n = 0
            for j in range(0, output_height, self.upsampling_factor):
                dLdA[:,:,m, n] = dLdZ[:,:,i, j]
                n += 1
            m += 1
        return dLdA

class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        batch_size, in_channels, input_width, input_height = A.shape
        output_width =  (input_width - 1)//self.downsampling_factor + 1
        output_height =  (input_height - 1)//self.downsampling_factor + 1
        Z = np.zeros((batch_size, in_channels, output_width, output_height)) # TODO
        m = 0
        for i in range(0, input_width, self.downsampling_factor):
            n = 0
            for j in range(0, input_height, self.downsampling_factor):
                Z[:,:,m, n] = A[:,:,i, j]
                n += 1
            m += 1
        self.shape1, self.shape2 = A.shape[2], A.shape[3]
        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, in_channels, output_width, output_height = dLdZ.shape
        input_width = output_width * self.downsampling_factor - (self.downsampling_factor - 1)
        input_height = output_height * self.downsampling_factor - (self.downsampling_factor - 1)
        
        dLdA = np.zeros((batch_size, in_channels, self.shape1, self.shape2)) # TODO
        for i in range(0, output_width):
            for j in range(0, output_height):
                dLdA[:,:,(i)*self.downsampling_factor, (j)*self.downsampling_factor] = dLdZ[:,:,i, j]

       
        return dLdA
