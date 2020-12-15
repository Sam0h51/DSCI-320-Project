import matplotlib.pyplot as plt
import numpy.random as rn
import numpy as np
import math
import csv


class Neural_Network():
    def __init__(self, insize, L2, L3, osize, Lrate):
        ## Store Constants
        self.insize = insize
        self.L2 = L2
        self.L3 = L3
        self.osize = osize
        self.Lrate = Lrate
        self.counter = 0
        self.batchsize = 5

        ## Initialize Weights
        self.W2 = rn.randn(L2, insize)
        self.W3 = rn.randn(L3, L2)
        self.WL = rn.randn(osize, L3)

        ## Initialize Biases
        self.B2 = rn.randn(L2, 1)
        self.B3 = rn.randn(L3, 1)
        self.BL = rn.randn(osize, 1)

        ## Initialize Update Storage
        self.updateWL = np.zeros((osize, L3))
        self.updateW3 = np.zeros((L3, L2))
        self.updateW2 = np.zeros((L2, insize))

        self.updateBL = np.zeros((osize, 1))
        self.updateB3 = np.zeros((L3, 1))
        self.updateB2 = np.zeros((L2, 1))

        print('W2: ', self.W2, '\nW3', self.W3, '\nWL', self.WL, '\nB2', self.B2, '\nB3', self.B3, '\nBL', self.BL)

    ## Sigma activation function
    def sigma(self, z):
        return 1/(1 + np.exp(-z))

    ## Derivative of sigma function, S should be the output of the sigma function
    def sigma_prime(self, S):
        return S*(1 - S)

    def forward_prop(self, xk):
        ## Input
        self.a1 = xk
##        print('a1:\n', self.a1)

        ## First Inner Layer Activation
        self.z2 = self.W2.dot(self.a1) + self.B2
        self.a2 = self.sigma(self.z2)
##        print('a2:\n', self.a2)

        ## Compute the diagonal gradient matrix
        self.D2 = np.diag(self.sigma_prime(self.a2).reshape(self.L2, ))

        ## Second Inner Layer Activation
        self.z3 = self.W3.dot(self.a2) + self.B3
        self.a3 = self.sigma(self.z3)
        self.D3 = np.diag(self.sigma_prime(self.a3).reshape(self.L3, ))
##        print('a3:\n', self.a3)

        ## Output Layer Activation
        self.zL = self.WL.dot(self.a3) + self.BL
        self.aL = self.sigma(self.zL)
        self.DL = np.diag(self.sigma_prime(self.aL).reshape(self.osize, ))
##        print('aL:\n', self.a3)

##        print(self.a1, self.z2, self.a2, self.D2, self.z3, self.a3, self.D3, self.zL, self.aL, self.DL)

        return self.aL

    def backward_prop(self, output, target):
        ## Compute Deltas
        self.deltaL = self.DL.dot(self.aL - target)
        self.delta3 = self.D3.dot(self.WL.T.dot(self.deltaL))
        self.delta2 = self.D2.dot(self.W3.T.dot(self.delta3))

        self.counter += 1

        ## Sum delta values for batch update
        if (self.counter == 0):
            self.updateWL = self.Lrate*self.deltaL.dot(self.a3.T)
            self.updateW3 = self.Lrate*self.delta3.dot(self.a2.T)
            self.updateW2 = self.Lrate*self.delta2.dot(self.a1.T)

            self.updateBL = self.Lrate*self.deltaL
            self.updateB3 = self.Lrate*self.delta3
            self.updateB2 = self.Lrate*self.delta2
        elif (self.counter == self.batchsize):
##            print(self.updateWL)
            self.WL = self.WL - self.updateWL/self.batchsize
            self.W3 = self.W3 - self.updateW3/self.batchsize
            self.W2 = self.W2 - self.updateW2/self.batchsize

            self.BL = self.BL - self.updateBL/self.batchsize
            self.B3 = self.B3 - self.updateBL/self.batchsize
            self.B2 = self.B2 - self.updateBL/self.batchsize

            self.counter = 0
        else:
            self.updateWL += self.Lrate*self.deltaL.dot(self.a3.T)
            self.updateW3 += self.Lrate*self.delta3.dot(self.a2.T)
            self.updateW2 += self.Lrate*self.delta2.dot(self.a1.T)

            self.updateBL += self.Lrate*self.deltaL
            self.updateB3 += self.Lrate*self.delta3
            self.updateB2 += self.Lrate*self.delta2

        ## Update Weights and Biases
##        self.WL = self.WL - self.Lrate*self.deltaL.dot(self.a3.T)
##        self.W3 = self.W3 - self.Lrate*self.delta3.dot(self.a2.T)
##        self.W2 = self.W2 - self.Lrate*self.delta2.dot(self.a1.T)
##
##        self.BL = self.BL - self.Lrate*self.deltaL
##        self.B3 = self.B3 - self.Lrate*self.delta3
##        self.B2 = self.B2 - self.Lrate*self.delta2

##        print(self.a1, self.z2, self.a2, self.D2, self.z3, self.a3, self.D3, self.zL, self.aL, self.DL)

    def train_network(self, xk, target):
        o = self.forward_prop(xk)
        self.backward_prop(o, target)

class Data_Ingest():
    def __init__(self, filename, xdim, ignore_top = False):
        self.X = np.array(())
        self.Y = np.array(())

        self.xdim = xdim
        self.pull_data(filename, ignore_top, xdim)

    def pull_data(self, filename, ignore_top, xdim):

        with open(filename, 'r') as infile:
            reader = csv.reader(infile)
            if(ignore_top):
                reader.__next__()
            
            for row in reader:
                for i in range(xdim):
                    self.X = np.append(self.X, float(row[i]))
                self.Y = np.append(self.Y, float(row[xdim]))

            self.X = self.X.reshape(int(self.X.size/xdim), xdim)

    def get_random_input(self):
        a, b = self.X.shape
        s = math.floor(rn.uniform(0, a))

        return self.X[s].reshaped(self.X[s].size, 1), self.Y[s]

    def get_XandY(self):
        return self.X, self.Y

    def generate_consistent_set(self, amount):
        a, b = self.X.shape
        consistent = np.array(())
        con_y = np.array(())
        for i in range(amount):
            s = math.floor(rn.uniform(0, a))
            consistent = np.append(consistent, self.X[s])
            con_y = np.append(con_y, self.Y[s])

        self.consistent = consistent.reshape(int(consistent.size/self.xdim), self.xdim)
        self.con_y = con_y

        return self.consistent, self.con_y

    def normalize_data(self):
        normalized = self.X.T
        a, b = normalized.shape
        for i in range(a):
            normalized[i] = normalized[i]/np.amax(normalized[i])
        return normalized.T

    def normalize_consistent_data(self):
        normalized = self.consistent.T
        a, b = normalized.shape
        for i in range(a):
            normalized[i] = (normalized[i] - np.amin(normalized[i]))/(np.amax(normalized[i]) - np.amin(normalized[i]))
        return normalized.T

## Create Data Structures
NeuNet = Neural_Network(10, 15, 7, 1, 0.1)
Dat = Data_Ingest('TrainingData.csv', 10, True)
Dat2 = Data_Ingest('TestingData.csv', 10)

## Setup for Training
E = 5000
X, y = Dat.generate_consistent_set(E)
Xt, yt = Dat2.get_XandY()
Xc, yc = Dat.get_XandY()

maxval = np.maximum(np.amax(X), np.amax(Xt))
X = X/maxval
Xt = Xt/maxval

X = Dat.normalize_consistent_data()
Xc = Dat.normalize_data()
Xt = Dat2.normalize_data()

def test_error(X, y):
    a, b = X.shape
    outs = np.array(())
    for i in range(a):
        outs = np.append(outs, NeuNet.forward_prop(X[i].reshape(10, 1)))
    error = outs - y
    classifications = np.round(outs, 0)
##    print(classifications)
    misclassifications = classifications - y

    return np.linalg.norm(error), np.linalg.norm(misclassifications)**2

def print_out(X, y):
    a, b = X.shape
    print('Output\t\tExpected')
    for i in range(a):
        print(NeuNet.forward_prop(X[i].reshape(10, 1)), '\t', y[i])



## Training the Neural Net
for i in range(E):
    NeuNet.train_network(X[i].reshape(10, 1), y[i])

    if(i%100 == 0):
        print('Input:\n')
        print(X[i].reshape(10, 1))
        print('Output\t\tExpected Output')
        print(NeuNet.forward_prop(X[i].reshape(10, 1)), '\t', y[i])
        print(test_error(Xt, yt))
    if(i%1000 == 0):
        print(test_error(Xc, yc))
    

print('Input:\n')
print(X[E-1])
print('Output\t\tExpected Output')
print(NeuNet.forward_prop(X[i].reshape(10, 1)), '\t', y[i])
print(test_error(Xt, yt))
print('\n\n\nTesting Data full output:')
print(NeuNet.WL, '\n\n', NeuNet.W3, '\n\n', NeuNet.W2, '\n\n', NeuNet.BL, NeuNet.B3, NeuNet.B2)
print_out(Xt, yt)


##print('\n\nOUTPUT=S:')
##print_out(X, y)




































