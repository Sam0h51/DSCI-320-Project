import matplotlib.pyplot as plt
import numpy.random as rn
import numpy as np
import math

Xt = np.array(([0, 0, 0], [0, 0, 1], [0, 1, 0],
              [0, 1, 1], [1, 0, 0], [1, 0, 1],
              [1, 1, 0], [1, 1, 1]), dtype = float)
Yt = np.array(([1], [0], [0], [0], [0], [0], [0], [0]), dtype = float)
xPredicted = np.array(([0, 0, 1]), dtype = float).reshape(3, 1)

test = rn.randn(4, 1)
print(test)
print(np.diag(test.reshape(4, )))

T = np.array(())
V = np.array(())

for i in range(20):
    u = np.array([rn.normal(1, 0.3), rn.normal(1, 0.3)])
    T = np.append(T, u)
    V = np.append(V, [2])
    u = np.array([rn.normal(-1, 0.3), rn.normal(-1, 0.3)])
    T = np.append(T, u)
    V = np.append(V, [0])

print(T)
T = T.reshape(40, 2)
print(T)
print(V)
V = V.reshape(40, 1)

qPredicted = np.array(([1., 1.]), dtype=float).reshape(2, 1)
    

class NeuralNet (object):
    def __init__(self):
        self.inputlayersize  = 2
        self.innerlayersize  = 4
        self.innerlayer2size = 4
        self.outputlayersize = 1
        self.scalingfactor = 3

        self.W1 = rn.randn(self.innerlayersize, self.inputlayersize)
        self.W2 = rn.randn(self.innerlayer2size, self.innerlayersize)
        self.W3 = rn.randn(self.outputlayersize, self.innerlayer2size)

        self.B1 = rn.randn(self.innerlayersize, 1)
        self.B2 = rn.randn(self.innerlayer2size, 1)
        self.B3 = rn.randn(self.outputlayersize, 1)
        

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

        return self.z4*self.scalingfactor

    def back_prop(self, X, y, output):
        self.out_error = y - output
        
        self.out_del = self.D4.dot(self.out_error)
        self.delta2 = self.D3.dot(self.W3.T.dot(self.out_del))
        self.delta1 = self.D2.dot(self.W2.T.dot(self.delta2))

        self.W1 += self.delta1.dot(X.T)
        self.W2 += self.delta2.dot(self.z2.T)
        self.W3 += self.out_del.dot(self.z3.T)
        self.B1 += self.delta1
        self.B2 += self.delta2
        self.B3 += self.out_del

    def train_network(self, X, Y):
        output = self.forward_prop(X)
        self.back_prop(X, Y, output)

    def saveWeights(self):
        # save this in order to reproduce our cool network
        np.savetxt("weightsLayer1.txt", self.W1, fmt="%s")
        np.savetxt("weightsLayer2.txt", self.W2, fmt="%s")

    def predictOutput(self):
        print ("Predicted XOR output data based on trained weights: ")
        print ("Expected (X1-X3): \n" + str(qPredicted))
        print ("Output (Y1): \n" + str(self.forward_prop(qPredicted)))

myNeuNet = NeuralNet()
E = 10000

for i in range(E):
    a = math.floor(rn.uniform(0, 40))
    myNeuNet.train_network(T[a].reshape(2, 1), V[a])
    if(i%100 == 0):
        myNeuNet.predictOutput()

myNeuNet.predictOutput()






















