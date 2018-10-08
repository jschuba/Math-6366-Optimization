import math

import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

A = np.array([[-1,0.4,0.8],
              [1,0,0],
              [0,1,0] ])

b = np.array([1, 0, 0.3])

x0 = np.array([0,0,0])
x_des = np.array([7,2,-6])

n = 3
nt = 30

# Build the matrix M
M = np.zeros([n, nt])
for col in range(nt):
    M[:,col] = np.linalg.matrix_power(A, nt-col-1) @ b

t = cp.Variable(nt)
u = cp.Variable(nt)

objective = cp.Minimize(cp.sum_entries(t))

constraints = [ M @ u == x_des,
               u <= t,
               u >= -t,
               u <= (t+1)/2,
               u >= -(t+1)/2 ]

prob = cp.Problem(objective, constraints)
result = prob.solve()

# cvx returns np.matrix type, even though this should be a vector
u_val = np.sum(np.array(u.value), axis = 1)

x = list(range(nt))

fig = plt.figure()
ax = fig.gca()

plt.scatter(x, u_val)
ax.set_title("Minimum Fuel Optimial Control - actuator signal")
ax.set_xlabel("Time step")
ax.set_ylabel("Actuator value")


# Find the points at each time step
def find_next_position(A, b, x_prev, control_signal):
    return A @ x_prev + control_signal * b

point_list = []
point_list.append(x0)
for i in range(1,30):
    point_list.append(find_next_position(A, b, point_list[i-1], u_val[i-1]))
    
point_list = np.array(point_list)

fig = plt.figure()
fig.set_size_inches(16,16)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(point_list[:,0],
           point_list[:,1],
           point_list[:,2])

for i, txt in enumerate(range(nt)):
    ax.text(point_list[i,0], point_list[i,1], point_list[i,2], txt)

axis_line = [i/100 for i in range(-1000,1000,1)]

ax.scatter(axis_line, 0, 0, s=1, c='k')
ax.scatter(0, axis_line, 0, s=1, c='k')
ax.scatter(0, 0, axis_line, s=1, c='k')

llim, ulim = -7, 7
ax.set_xlim(llim,ulim)
ax.set_ylim(llim,ulim)
ax.set_zlim(llim,ulim)