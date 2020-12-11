import matplotlib.pyplot as plt
import numpy.random as rn
import numpy as np
import math
import csv

class NeuralNet (object):
    def __init__(self, insize, outsize):
        self.inputlayersize  = insize
        self.innerlayersize  = 15
        self.innerlayer2size = 15
        self.outputlayersize = outsize
        self.scalingfactor = 1

        self.W1 = rn.randn(self.innerlayersize, self.inputlayersize)
        self.W2 = rn.randn(self.innerlayer2size, self.innerlayersize)
        self.W3 = rn.randn(self.outputlayersize, self.innerlayer2size)

        self.B1 = rn.randn(self.innerlayersize, 1)
        self.B2 = rn.randn(self.innerlayer2size, 1)
        self.B3 = rn.randn(self.outputlayersize, 1)

        self.counter = 0
        

    def sigma_f(self, S):
        return 1/(1 + np.exp(-S))

    def sigma_grad(self, S):
        return S*(1 - S)

    def forward_prop(self, X):
        self.z = np.dot(self.W1, X) + self.B1
        self.z2 = self.sigma_f(self.z)
        self.D2 = np.diag(self.sigma_grad(self.z2).reshape(self.innerlayersize, ))
        
        self.z3 = np.dot(self.W2, self.z2) + self.B2
        self.z3 = self.sigma_f(self.z3)
        self.D3 = np.diag(self.sigma_grad(self.z3).reshape(self.innerlayer2size, ))

        self.z4 = np.dot(self.W3, self.z3) + self.B3
        self.z4 = self.sigma_f(self.z4)
        self.D4 = np.diag(self.sigma_grad(self.z4).reshape(self.outputlayersize, ))

        return self.z4

    def back_prop(self, X, y, output):
        self.out_error = y - output
        
        self.out_del = self.D4.dot(self.out_error)
        self.delta2 = self.D3.dot(self.W3.T.dot(self.out_del))
        self.delta1 = self.D2.dot(self.W2.T.dot(self.delta2))

##        if(self.counter==0):
##            self.W1Update = self.delta1.dot(X.T)
##            self.W2Update = self.delta2.dot(self.z2.T)
##            self.W3Update = self.out_del.dot(self.z3.T)
##            self.B1Update = self.delta1
##            self.B2Update = self.delta2
##            self.B3Update = self.out_del
##        else:
##            self.W1Update += self.delta1.dot(X.T)
##            self.W2Update += self.delta2.dot(self.z2.T)
##            self.W3Update += self.out_del.dot(self.z3.T)
##            self.B1Update += self.delta1
##            self.B2Update += self.delta2
##            self.B3Update += self.out_del
##
##        self.counter += 1
##
##        if(self.counter == 10):
        self.W1 += self.delta1.dot(X.T)
        self.W2 += self.delta2.dot(self.z2.T)
        self.W3 += self.out_del.dot(self.z3.T)
        self.B1 += self.delta1
        self.B2 += self.delta2
        self.B3 += self.out_del
        self.counter = 0

    def train_network(self, X, Y):
        output = self.forward_prop(X)
        self.back_prop(X, Y, output)

    def predictOutput(self, test, out):
        print ("Predicted New Case Numbers Based on Prior Days")
        print ("Input: \n" + str(test))
        print ("Output: \t\tExpected Output")
        print (self.forward_prop(test), '\t', out)


X = np.array(())
y = np.array(())
Xt = np.array(())
yt = np.array(())

with open('TrainingData.csv', 'r') as infile:
    reader = csv.reader(infile)
    line = 0
    for row in reader:
        if(line > 0):
            X = np.append(X, float(row[0]))
            X = np.append(X, float(row[1]))
            X = np.append(X, float(row[2]))
            X = np.append(X, float(row[3]))
            X = np.append(X, float(row[4]))
            X = np.append(X, float(row[5]))
            X = np.append(X, float(row[6]))
            X = np.append(X, float(row[7]))
            X = np.append(X, float(row[8]))
            X = np.append(X, float(row[9]))
            y = np.append(y, float(row[10]))
        line += 1

with open('TestingData.csv', 'r') as infile:
    reader = csv.reader(infile)
    for row in reader:
        Xt = np.append(Xt, float(row[0]))
        Xt = np.append(Xt, float(row[1]))
        Xt = np.append(Xt, float(row[2]))
        Xt = np.append(Xt, float(row[3]))
        Xt = np.append(Xt, float(row[4]))
        Xt = np.append(Xt, float(row[5]))
        Xt = np.append(Xt, float(row[6]))
        Xt = np.append(Xt, float(row[7]))
        Xt = np.append(Xt, float(row[8]))
        Xt = np.append(Xt, float(row[9]))
        yt = np.append(yt, float(row[10]))

Epochs = 1000

NeuNet = NeuralNet(10, 1)
X = X.reshape(int(X.size/10), 10)
Xt = Xt.reshape(int(Xt.size/10), 10)

C = np.array(())
Ct = np.array(())
k = np.array(())

def loss_function(NN, X, Xt, y, yt):
    loss = 0.
    losst = 0.
    for i in range(y.size):
        loss += (y[i] - NN.forward_prop(X[i].reshape(10, 1)))**2
    for i in range(yt.size):
        losst += (yt[i] - NN.forward_prop(Xt[i].reshape(10, 1)))**2

    return loss/y.size, losst/yt.size

for i in range(Epochs):
    a = math.floor(rn.uniform(0, X.size/10))
    NeuNet.train_network(X[a].reshape(10, 1), y[a])

    u, v = loss_function(NeuNet, X, Xt, y, yt)
    C = np.append(C, u)
    Ct = np.append(Ct, v)
    k = np.append(k, i)

    if(i%100 == 0):
        NeuNet.predictOutput(X[a].reshape(10, 1), y[a])

NeuNet.predictOutput(X[a].reshape(10, 1), y[a])

print('Output: \t\tExpected Output:')
for i in range(y.size):
    print(NeuNet.forward_prop(X[i].reshape(10, 1)), '\t', y[i])

plt.plot(k, C, color='red')
plt.plot(k, Ct, color='blue')

plt.show()
    
























