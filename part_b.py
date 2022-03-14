import numpy as np

def generate_trainingdata(m=25):
    return np.array([0,0])+0.25*np.random.randn(m,2)
    # array of 25x2

def f(x, minibatch):
    # loss function sum_{w in training data} f(x,w)
    y=0; count=0
    for w in minibatch:
        print(f"X:\t{x}")
        z=x-w-1
        print(f"Z:\t{z}")
        y=y+np.minimum(27*(z[0]**2+z[1]**2), (z[0]+6)**2+(z[1]+10)**2) 
        print(f"Y:\t{y}")  
        count=count+1
    return y/count

from matplotlib import projections
import sympy
import matplotlib.pyplot as plt
import numpy as np
from torch import linspace
from mpl_toolkits.mplot3d import Axes3D

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

# x_start_range = [0.01, 0.1, 1, 10, 100]

# x_start = 1
# y_start = 1

# alpha = 0.1

# curr_xy = [x_start, y_start]

# curr_z = f(curr_xy)

# xy_guesses = []
# z_values = []
original_data = generate_trainingdata()
print(original_data)
data = original_data # for shuffling
num_training_dp = data.shape[0]
print(num_training_dp)

num_epochs = 10
num_minibatches = 1
batch_size = 10 # calculate batch size OR num minibatches

starting_x = np.array([0, 0])

x = starting_x

for epoch in range(num_epochs):
    # SHUFFLE
    #for batch in np.arange(0, num_training_dp, batch_size):
    # shuffled_idx = np.random.shuffle(np.arange(num_training_dp))
    # minibatch_idxs = np.split(shuffled_idx, int(5), axis=0)
    # minibatches = data[minibatch_idxs]
    np.random.shuffle(data)
    minibatches = np.split(data, num_minibatches)
    print(len(minibatches))
    # break

    #print(minibatches.shape)
    for minibatch in minibatches:
        fxN = f(x, minibatch) 
        print(fxN)
    # xy_guesses.append(curr_xy)
    # z_values.append(curr_z)
    
        
    # slope = np.array([dfdx(curr_xy), dfdy(curr_xy)])   
    # alpha = 1 # calc_polyak(f(curr_xy), fstar, slope)
    # step = slope*alpha
    
    # curr_xy = curr_xy - step
    # curr_z = f(curr_xy)

# xy_guesses = np.array(xy_guesses)
# z_values = np.array(z_values)
      
# plt.scatter(xy_guesses[:, 0], xy_guesses[:, 1], c=range(num_iterations))
# plt.show()