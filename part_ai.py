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

num_epochs = 10
num_minibatches = 5
# batch_size = 10 # calculate batch size OR num minibatches

starting_x = np.array([3, 3])

alpha = 0.1
delta = 0.001

x = starting_x

x0s = []
x1s = []
fxs = []

# for epoch in range(num_epochs):
#     # SHUFFLE
#     np.random.shuffle(data)
#     minibatches = np.split(data, num_minibatches)
#     print(minibatches)
#     # break

#     #print(minibatches.shape)
#     for minibatch in minibatches:
#         fx = f(x, minibatch)
#         x0s.append(x[0])
#         x1s.append(x[1])
#         fxs.append(fx)

#         fx0_delta = f([x[0]+delta, x[1]], minibatch)
#         dfdx0 = (fx0_delta-fx)/delta

#         fx1_delta = f([x[0], x[1]+delta], minibatch)
#         dfdx1 = (fx1_delta-fx)/delta

#         dfdx = np.array([dfdx0, dfdx1])
#         x = x - alpha*dfdx

# plt.plot(fxs)
# plt.xlabel("Iterations")
# plt.ylabel("$f(x)$")
# plt.show()

# scatter_colors = plt.scatter(x0s, x1s, c=range(len(x0s)))
# plt.plot(x0s, x1s, alpha=0.1)
# plt.xlabel("$x_0$")
# plt.ylabel("$x_1$")
# plt.colorbar(scatter_colors, label="Iteration")
# plt.show()
x0_length = 200
x1_length = 100

x0_space = np.linspace(-10, 5, x0_length)
x1_space = np.linspace(-15, 5, x1_length)

X, Y = np.meshgrid(x0_space, x1_space)
Z = np.zeros((x1_length, x0_length))
dzdx0_finite = np.zeros((x1_length, x0_length))
dzdx1_finite = np.zeros((x1_length, x0_length))
delta = 0.001

for ind_x0 in range(x0_length):
    for ind_x1 in range(x1_length):
        x0 = x0_space[ind_x0]
        x1 = x1_space[ind_x1]
        Z[ind_x1, ind_x0] = f([x0, x1], data)
        dzdx0_finite[ind_x1, ind_x0] = (f([x0+delta, x1], data) - Z[ind_x1, ind_x0])/delta
        dzdx1_finite[ind_x1, ind_x0] = (f([x0, x1+delta], data) - Z[ind_x1, ind_x0])/delta

contour_colours = plt.contourf(X, Y, Z)
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

contour_colours = plt.contourf(X, Y, dzdx0_finite)
#plt.contour(X, Y, Z)
plt.colorbar(contour_colours, label="${df}/{dx_0}$")
plt.xlabel("$x_{0}$")
plt.ylabel("$x_{1}$")
plt.title("Contour plot for finite difference estimate of ${df}/{dx_0}$ when $N=T$")
plt.show()

contour_colours = plt.contourf(X, Y, dzdx1_finite)
#plt.contour(X, Y, Z)
plt.colorbar(contour_colours, label="${df}/{dx_1}$")
plt.xlabel("$x_{0}$")
plt.ylabel("$x_{1}$")
plt.title("Contour plot for finite difference estimate of ${df}/{dx_1}$ when $N=T$")
plt.show()



#         print(fxN)
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