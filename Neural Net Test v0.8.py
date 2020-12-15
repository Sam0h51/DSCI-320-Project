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

        ## Initialize Weights
        self.W2 = rn.randn(L2, insize)
        self.W3 = rn.randn(L3, L2)
        self.WL = rn.randn(osize, L3)

        ## Initialize Biases
        self.B2 = rn.randn(L2, 1)
        self.B3 = rn.randn(L3, 1)
        self.BL = rn.randn(osize, 1)

    ## Sigma activation function
    def sigma(self, z):
        return 1/(1 + np.exp(-z))

    ## Derivative of sigma function, S should be the output of the sigma function
    def sigma_prime(self, S):
        return S*(1 - S)

    def forward_prop(self, xk):
        ## Input
        self.a1 = xk

        ## First Inner Layer Activation
        self.z2 = self.W2.dot(self.a1) + self.B2
        self.a2 = self.sigma(self.z2)

        ## Compute the diagonal gradient matrix
        self.D2 = np.diag(self.sigma_prime(self.a2).reshape(self.L2, ))

        ## Second Inner Layer Activation
        self.z3 = self.W3.dot(self.a2) + self.B3
        self.a3 = self.sigma(self.z3)
        self.D3 = np.diag(self.sigma_prime(self.a3).reshape(self.L3, ))

        ## Output Layer Activation
        self.zL = self.WL.dot(self.a3) + self.BL
        self.aL = self.sigma(self.zL)
        self.DL = np.diag(self.sigma_prime(self.aL).reshape(self.osize, ))

        return self.aL

    def backward_prop(self, output, target):
        ## Compute Deltas
        self.deltaL = self.DL.dot(self.aL - target)
        self.delta3 = self.D3.dot(self.WL.T.dot(self.deltaL))
        self.delta2 = self.D2.dot(self.W3.T.dot(self.delta3))

        ## Update Weights and Biases
        self.WL = self.WL - self.Lrate*self.deltaL.dot(self.a3.T)
        self.W3 = self.W3 - self.Lrate*self.delta3.dot(self.a2.T)
        self.W2 = self.W2 - self.Lrate*self.delta2.dot(self.a1.T)

        self.BL = self.BL - self.Lrate*self.deltaL
        self.B3 = self.B3 - self.Lrate*self.delta3
        self.B2 = self.B2 - self.Lrate*self.delta2

    def train_network(self, xk, target):
        o = self.forward_prop(xk)
        self.backward_prop(o, target)

class Data_Ingest():
    def __init__(self, filename, xdim, ignore_top = False):
        self.X = np.array(())
        self.Y = np.array(())

        self.xdim = xdim
        self.pull_data(filename, ignore_top, xdim)
        self.normalized = self.X

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
        self.normalized = self.X.T
        a, b = self.normalized.shape
        for i in range(a):
            self.normalized[i] = (self.normalized[i] - np.amin(self.normalized[i]))/(np.amax(self.normalized[i]) - np.amin(self.normalized[i]))
        self.normalized = self.normalized.T
        return self.normalized, self.Y

    def normalize_consistent_data(self):
        normalized_con = self.consistent.T
        a, b = normalized.shape
        for i in range(a):
            normalized[i] = (normalized[i] - np.amin(normalized[i]))/(np.amax(normalized[i]) - np.amin(normalized[i]))
        return normalized.T, self.con_y

    def get_split_consistent(self, amount):
        X = np.array(())
        y = np.array(())
        T = np.array(())
        u = np.array(())

        for i in range(self.Y.size):
            s = rn.rand()
            if(s > 0.2):
                X = np.append(X, self.normalized[i])
                y = np.append(y, self.Y[i])
            else:
                T = np.append(T, self.normalized[i])
                u = np.append(u, self.Y[i])

        X = X.reshape(int(X.size/self.xdim), self.xdim)
        T = T.reshape(int(T.size/self.xdim), self.xdim)
        a, b = X.shape

        consistent = np.array(())
        con_y = np.array(())

        for i in range(amount):
            s = math.floor(rn.uniform(0, a))
            consistent = np.append(consistent, X[s])
            con_y = np.append(con_y, y[s])

        self.consistent = consistent.reshape(int(consistent.size/self.xdim), self.xdim)
        self.con_y = con_y

        self.split_conX = X
        self.split_conY = y
        self.split_conT = T
        self.split_conU = u

        return self.consistent, self.con_y, T, u

    def get_split_XandY(self):
        return self.split_conX, self.split_conY

## Function to average testing set data
def average_error(X, y, NN):
    error = 0
    for i in range(y.size):
        error += (NN.forward_prop(X[i].reshape(10, 1)) - y[i])**2
    return error/y.size

def produce_confusion_matrix(X, y, NN):
    output = np.array(())
    for i in range(y.size):
        output = np.append(output, NN.forward_prop(X[i].reshape(10, 1)))
    output = np.round(output)

    con_matrix = np.array(([0, 0], [0, 0]))

    for i in range(y.size):
        if (y[i] == 0):
            if(output[i] == 0):
                con_matrix[0][0] += 1
            elif(output[i] == 1):
                con_matrix[1][0] += 1
            else:
                print("ERROR!")
        elif(y[i] == 1):
            if(output[i] == 0):
                con_matrix[0][1] += 1
            elif(output[i] == 1):
                con_matrix[1][1] += 1
            else:
                print("ERROR!")

    return con_matrix

## Computes total error and total misclassifications for a given data set
def test_error(X, y, NN):
    a, b = X.shape
    outs = np.array(())
    for i in range(a):
        outs = np.append(outs, NN.forward_prop(X[i].reshape(10, 1)))
    error = outs - y
    classifications = np.round(outs, 0)
    misclassifications = classifications - y

    return np.linalg.norm(error), np.linalg.norm(misclassifications)**2

## A function to print the outputs vs expected outputs of a given neural net
def print_out(X, y, NN):
    a, b = X.shape
    print('Output\t\tExpected')
    for i in range(a):
        print(NN.forward_prop(X[i].reshape(10, 1)), '\t', y[i])

## Create Data Structures
NeuNet1 = Neural_Network(10, 15, 7, 1, 1)
NeuNet2 = Neural_Network(10, 15, 7, 1, 0.1)
NeuNet3 = Neural_Network(10, 15, 7, 1, 0.01)
Dat = Data_Ingest('DesiredColumns.csv', 10, True)

## Setup for Training
E = 10000
Xs, ys = Dat.normalize_data()
X, y, Xt, yt = Dat.get_split_consistent(E)
Xk, yk = Dat.get_split_XandY()

## Arrays to store the single error(v) and average error(w) at each k(u)
u1 = np.array(())
v1 = np.array(())
w1 = np.array(())

u2 = np.array(())
v2 = np.array(())
w2 = np.array(())

u3 = np.array(())
v3 = np.array(())
w3 = np.array(())

k = 0
            
## Training the Neural Net
for i in range(E):
    NeuNet1.train_network(X[i].reshape(10, 1), y[i])
    NeuNet2.train_network(X[i].reshape(10, 1), y[i])
    NeuNet3.train_network(X[i].reshape(10, 1), y[i])
    
    u1 = np.append(u1, k)
    v1 = np.append(v1, (NeuNet1.forward_prop(X[i].reshape(10, 1)) - y[i])**2)
    w1 = np.append(w1, average_error(Xt, yt, NeuNet1))

    u2 = np.append(u2, k)
    v2 = np.append(v2, (NeuNet2.forward_prop(X[i].reshape(10, 1)) - y[i])**2)
    w2 = np.append(w2, average_error(Xt, yt, NeuNet2))

    u3 = np.append(u3, k)
    v3 = np.append(v3, (NeuNet3.forward_prop(X[i].reshape(10, 1)) - y[i])**2)
    w3 = np.append(w3, average_error(Xt, yt, NeuNet3))
    
    k += 1

    if(i%100 == 0):
        print('Input:\n')
        print(X[i])
        print('Output\t\tExpected Output')
        print('NN1: ', NeuNet1.forward_prop(X[i].reshape(10, 1)), '\t', y[i])
        print('NN2: ', NeuNet2.forward_prop(X[i].reshape(10, 1)), '\t', y[i])
        print('NN3: ', NeuNet3.forward_prop(X[i].reshape(10, 1)), '\t', y[i])
        print(test_error(Xt, yt, NeuNet1))
        print(test_error(Xt, yt, NeuNet2))
        print(test_error(Xt, yt, NeuNet3))
    

print('Input:\n')
print(X[E-1])
print('Output\t\tExpected Output')
print(NeuNet1.forward_prop(X[i].reshape(10, 1)), '\t', y[i])
print(test_error(Xt, yt, NeuNet1))
print('\n\nOUTPUT=S:')
print_out(Xk, yk, NeuNet1)
print('\n\nNN1 Confusion Matrix:')
print(produce_confusion_matrix(Xk, yk, NeuNet1))
print(produce_confusion_matrix(Xt, yt, NeuNet1))
print(produce_confusion_matrix(Xs, ys, NeuNet1))

print('\n\nNN2 Confusion Matrix:')
print(produce_confusion_matrix(Xk, yk, NeuNet2))
print(produce_confusion_matrix(Xt, yt, NeuNet2))
print(produce_confusion_matrix(Xs, ys, NeuNet2))

print('\n\nNN3 Confusion Matrix:')
print(produce_confusion_matrix(Xk, yk, NeuNet3))
print(produce_confusion_matrix(Xt, yt, NeuNet3))
print(produce_confusion_matrix(Xs, ys, NeuNet3))

fig, axs = plt.subplots(3)

axs[0].set_xlabel('k')
axs[0].set_ylabel('Error')
axs[0].set_title('Single point error and Average Squared Error for Neural Net with mu=1')

axs[1].set_xlabel('k')
axs[1].set_ylabel('Error')
axs[1].set_title('Single point error and Average Squared Error for Neural Net with mu=0.1')

axs[2].set_xlabel('k')
axs[2].set_ylabel('Error')
axs[2].set_title('Single point error and Average Squared Error for Neural Net with mu=0.01')



axs[0].plot(u1, v1, color='red')
axs[0].plot(u1, w1, color='green')

axs[1].plot(u2, v2, color='red')
axs[1].plot(u2, w2, color='green')

axs[2].plot(u3, v3, color='red')
axs[2].plot(u3, w3, color='green')


fig.show()



































