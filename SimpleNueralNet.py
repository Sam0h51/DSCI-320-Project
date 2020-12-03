import matplotlib.pyplot as plt
import numpy.random as rn
import numpy as np

## First, we need to generate some data

## Control Variables
N = 100
sigma = 0.3

## First, we want the red data
w = np.array(())
y = np.array(())

for i in range(3*N):
    x = np.array([rn.normal(1, sigma), rn.normal(-1, sigma)])
    yi = np.array([0, 1])
    w = np.append(w, x.transpose())
    y = np.append(y, yi)

## Reshaping the data array into a usable form
w = w.reshape(300, 2)
w = w.T
print(w.shape)

## Plotting the red data
for i in range(w[0].size):
    plt.plot(w[0][i], w[1][i], 'r.')

## Now, we need the blue data
u = np.array(())

for i in range(N):
    x1 = np.array([rn.normal(1, sigma), rn.normal(1, sigma)])
    x2 = np.array([rn.normal(-1, sigma), rn.normal(1, sigma)])
    x3 = np.array([rn.normal(-1, sigma), rn.normal(-1, sigma)])
    yi = np.array([1, 0])

    u = np.append(u, x1)
    u = np.append(u, x2)
    u = np.append(u, x3)
    
    y = np.append(y, yi)
    y = np.append(y, yi)
    y = np.append(y, yi)

u = u.reshape(300, 2)
u = u.T
print(u.shape)

## Plotting blue data
for i in range(u[0].size):
    plt.plot(u[0][i], u[1][i], 'b.')


y = y.reshape(600, 2)
y = y.T
print(y.shape)

def get_w(x):
    w1 = x[0:10]
    w2 = x[10:35]
    w3 = x[35:45]

    w1 = w1.shape(5, 2)
    w2 = w2.shape(5, 5)
    w3 = w3.shape(2, 5)

    return w1, w2, w3

def get_b(x):
    b1 = x[45, 50]
    b2 = x[50, 55]
    b3 = x[55, 57]

    return b1.T, b2.T, b3.T

def obj_f(x, blue, red, y):
    w1, w2, w3 = get_w(x)
    b1, b2, b3 = get_b(x)

    for i in range(blue[0].size):
        


plt.show()





















