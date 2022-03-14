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

num_epochs = 1
num_minibatches = 1
# batch_size = 10 # calculate batch size OR num minibatches

starting_x = np.array([0, 0])

x = starting_x

x0_length = 200
x1_length = 200

x0_space = np.linspace(-10, 5, x0_length)
x1_space = np.linspace(-15, 5, x1_length)

X, Y = np.meshgrid(x0_space, x1_space)
# Z = f([X, Y], data)
Z = np.zeros((x0_length, x1_length))

for x0 in range(x0_length):
    for x1 in range(x1_length):
        Z[x0, x1] = f([x0_space[x0], x1_space[x1]], data)

contour_colours = plt.contourf(X, Y, Z)
#plt.contour(X, Y, Z)
plt.colorbar(contour_colours, label="$f(x, N)$")
plt.xlabel("$x_{0}$")
plt.ylabel("$x_{1}$")
plt.title("Contour plot for function $f$ when $N=T$")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(X, Y, Z)
ax.set_xlabel("$x_{0}$")
ax.set_ylabel("$x_{1}$")
ax.set_zlabel("$f(x,N)$")
ax.set_title("Wireframe plot for function $f$ when $N=T$")
plt.show()


for epoch in range(num_epochs):
    # SHUFFLE
    np.random.shuffle(data)
    minibatches = np.split(data, num_minibatches)
    print(minibatches)
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