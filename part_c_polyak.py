import numpy as np

def generate_trainingdata(m=25):
    return np.array([0,0])+0.25*np.random.randn(m,2)
    # array of 25x2

def f(x, minibatch):
    # loss function sum_{w in training data} f(x,w)
    y=0; count=0
    for w in minibatch:
        z=x-w-1
        y=y+np.minimum(27*(z[0]**2+z[1]**2), (z[0]+6)**2+(z[1]+10)**2) 
        count=count+1
    return y/count

from matplotlib import projections
import sympy
import matplotlib.pyplot as plt
import numpy as np
from torch import linspace
from mpl_toolkits.mplot3d import axes3d

# x0, y0 = sympy.symbols("x0, y0", real=True)
# func= 8*(x0-10)**4+9*(y0-0)**2
# x_deriv = sympy.diff(func, x0)
# y_deriv = sympy.diff(func, y0)
# print(func,x_deriv,y_deriv)

# epsilon = 0.0001
# fstar = 0

# def f(xy):
#     return func.subs([(x0, xy[0]), (y0, xy[1])])

# def dfdx(xy):
#     return x_deriv.subs([(x0, xy[0]), (y0, xy[1])])

# def dfdy(xy):
#     return y_deriv.subs([(x0, xy[0]), (y0, xy[1])])

# def calc_polyak(fxy, fstar, slope):
#     return (fxy-fstar)/(slope.dot(np.transpose(slope)) + epsilon)

original_data = generate_trainingdata()
print(original_data)
data = original_data # for shuffling
num_training_dp = data.shape[0]
print(num_training_dp)

num_epochs = 3
num_minibatches = 2
batch_size = int(num_training_dp/num_minibatches)
# batch_size = 10 # calculate batch size OR num minibatches

starting_x = np.array([3, 3])

# alpha = 0.01
delta = 0.001
fstar = 0
epsilon = 0.001



list_x0s = []
list_x1s = []
list_fxs = []


for i in range(5):
    x0s = []
    x1s = []
    fxs = []
    
    x = starting_x
    
    for epoch in range(num_epochs):
        # SHUFFLE
        np.random.shuffle(data)
        minibatches = np.array_split(data, num_minibatches)
        print(minibatches)
        # break

        #print(minibatches.shape)
        for minibatch in minibatches:
            fx = f(x, minibatch)
            x0s.append(x[0])
            x1s.append(x[1])
            fxs.append(fx)

            fx0_delta = f([x[0]+delta, x[1]], minibatch)
            dfdx0 = (fx0_delta-fx)/delta

            fx1_delta = f([x[0], x[1]+delta], minibatch)
            dfdx1 = (fx1_delta-fx)/delta

            dfdx = np.array([dfdx0, dfdx1])
            
            alpha = (fx-fstar)/(dfdx.dot(np.transpose(dfdx)) + epsilon)
            
            x = x - alpha*dfdx
            
    list_x0s.append(x0s)
    list_x1s.append(x1s)
    list_fxs.append(fxs)


runs = range(len(list_x0s))

for run in runs:
    plt.plot(np.array(range(1, 1+len(list_x0s[run])))/num_minibatches, list_fxs[run], label=f"Run {run}")
# plt.title(f"Change in $f(x)$ value with {num_minibatches} mini-batches")
plt.title(f"Change in $f(x)$ value using Polyak")
plt.xlabel("Epochs")
plt.ylabel("$f(x)$")
#plt.yscale('log')
plt.legend()
plt.show()

for run in runs:
    scatter_colors = plt.scatter(list_x0s[run], list_x1s[run], c=np.array(range(1, 1+len(list_x0s[run])))/num_minibatches)
    plt.plot(list_x0s[run], list_x1s[run], alpha=0.5, label=f"Run {run}")

plt.scatter(3,3, c='r', marker='x', label='Origin')
plt.xlabel("$x_0$")
plt.ylabel("$x_1$")
plt.legend()
plt.title(f"Change in $x_0$ and $x_1$ values using Polyak")

plt.colorbar(scatter_colors, label="Epochs")
plt.show()