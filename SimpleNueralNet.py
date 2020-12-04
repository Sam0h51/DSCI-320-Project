import matplotlib.pyplot as plt
import numpy.random as rn
import numpy as np

## First, we need to generate some data

## Control Variables
N = 100
sigma = 0.3

## Setting up the data plots
fig, axs = plt.subplots(1, 2)

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
    axs[0].plot(w[0][i], w[1][i], 'r.')

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
    axs[0].plot(u[0][i], u[1][i], 'b.')


y = y.reshape(600, 2)
y = y.T
print(y.shape)

## This function parses the front of the 1D input array into the 3
## weight matrices
def get_w(x):
    w1 = x[0:10]
    w2 = x[10:35]
    w3 = x[35:45]

    w1 = w1.reshape(5, 2)
    w2 = w2.reshape(5, 5)
    w3 = w3.reshape(2, 5)

    return w1, w2, w3

## This function parses the back of the 1d input array into the 3
## bias vectors
def get_b(x):
    b1 = x[45:50]
    b2 = x[50:55]
    b3 = x[55:57]

    b1 = b1.reshape(5, 1)
    b2 = b2.reshape(5, 1)
    b3 = b3.reshape(2, 1)

    return b1, b2, b3

## This function applies the sigma function to all elements of an
## input vector
def sigma_f(x):
    for i in range(x.size):
        x[i] = 1/(1 + np.exp(-x[i]))
    return x

## This is the neural net, with 2 intermediate layers of 5 nodes each
## and a 2 node output layer
def f(x, w1, w2, w3, b1, b2, b3):
    ## Layer 1          (2x1 --> 5x1)
    xk = np.matmul(w1, x) + b1
    xk = sigma_f(xk)

    ## Layer 2          (5x1 --> 5x1)
    xk = np.matmul(w2, xk) + b2
    xk = sigma_f(xk)

    ## Layer 3, output  (5x1 --> 2x1)
    xk = sigma_f(np.matmul(w3, xk) + b3)

    return xk
    
## This function compares the output of the neural net to the
## correct outputs for the data. This is the function we want to
## minimize
def obj_f(x, blue, red, y):
    w1, w2, w3 = get_w(x)
    b1, b2, b3 = get_b(x)

    tot = 0

    for i in range(blue[0].size):
        xk = np.array([[blue[0][i]], [blue[1][i]]])
        Yk = f(xk, w1, w2, w3, b1, b2, b3)
        xt = np.array([[red[0][i]], [red[1][i]]])
        Yt = f(xt, w1, w2, w3, b1, b2, b3)

        yk_cor = np.array([[y[0][i]], [y[1][i]]])
        yt_cor = np.array([[y[0][i + 300]], [y[1][i + 300]]])

        tot = tot + np.linalg.norm(Yk - yk_cor)**2
        tot = tot + np.linalg.norm(Yt - yt_cor)**2
        
    return tot/600

## Now, I'm going to try to use very fast simulated annealing to minimize
## the objective function

## VFSA Setup

T = 20
sig = 20
N = 100
e = 0.01
k = 0

x0 = np.ones(57)
lowest = obj_f(x0, w, u, y)
lowx = x0

uk = np.array(k)
vk = np.array(lowest)
wk = np.array(lowest)

## A function to get the next input
def get_next(x, sig):
    for i in range(x.size):
        x[i] = x[i] + rn.uniform(-1, 1)*sig
    return x

## VFSA Iterations
for i in range(N):
    print('LET THERE BE NO DOUBT! K IS ', k)
    sigk = sig/(1 + e*k)
    Tk = T/(1 + e*k)
    xt = get_next(x0, sigk)
    
    y1 = obj_f(x0, w, u, y)
    y2 = obj_f(xt, w, u, y)

    if(y2 < y1):
        x0 = xt
    else:
        s = rn.rand()
        if(np.exp(-1*(y2 - y1)/Tk) > s):
            x0 = xt

    if(y2 < lowest):
        lowest = y2
        print('Lowest: ', lowest)
        lowx = x0

    k = k+1
    uk = np.append(uk, k)
    vk = np.append(vk, obj_f(x0, w, u, y))
    wk = np.append(wk, lowest)

## Plot VFSA Results
axs[1].plot(uk, vk, color='red')
axs[1].plot(uk, wk, color='green')

## Inefficient Divider Plotting
wf1, wf2, wf3 = get_w(lowx)
bf1, bf2, bf3 = get_b(lowx)

print('Checking that the neural net works')
print(f(np.array([[-1], [-1]]), wf1, wf2, wf3, bf1, bf2, bf3))
print(f(np.array([[-1], [1]]), wf1, wf2, wf3, bf1, bf2, bf3))
print(f(np.array([[1], [-1]]), wf1, wf2, wf3, bf1, bf2, bf3))
print(f(np.array([[1], [1]]), wf1, wf2, wf3, bf1, bf2, bf3))
print('Check finished\n')

for i in range(10):
    for j in range(10):
        xcomp = -2 + (4./10)*j
        ycomp = -1.5 + (3./10)*i

        inp = np.array([[xcomp], [ycomp]])
        l = f(inp, wf1, wf2, wf3, bf1, bf2, bf3)
        print(l)
        if(np.linalg.norm(f(inp, wf1, wf2, wf3, bf1, bf2, bf3)) < 1):
            axs[0].plot(xcomp, ycomp, 'g.')



plt.show()





















