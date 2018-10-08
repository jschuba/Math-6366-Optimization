import math

import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt

# Copy the original matlab code into python

def solve_chebyshev_problem(A, b):
    '''
    The goal is to find the Chebyshev center of a
    polyhedron, i.e., find the the largest euclidean ball
    that lies in a polyhedron
    
           P = {x \in R^2 : a_i'*x <= b_i, i=1,...,m}

    described by linear inequalites, where x is in R^2
    '''
    n, m = A.shape
    if n != b.shape[0]:
        raise ValueError
    
    r = cp.Variable()
    xc = cp.Variable(m)
    
    objective = cp.Maximize(r)
    constraints = []
    for row in range(n):
        constraints.append(A[row,:] @ xc + r* np.linalg.norm(A[row,:]) <= b[row])
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    
    radius = r.value
    center = np.array(xc.value)
    return radius, center

def plot_chebyshev_problem(A, b, radius = 0, center = [0,0]):
    '''
    Plots the solution to a chebyshev center problem
    The constraints are given by A and b,
    And the found ball is given by radius and center
    '''
    n, m = A.shape
    if n != b.shape[0]:
        raise ValueError
    
    lower_lim = center[0] - 1.1*radius - 1
    upper_lim = center[0] + 1.1*radius + 1
    
    x = np.linspace(lower_lim, upper_lim, 100)
    theta = np.linspace(0,2*math.pi,200)
    
    fig = plt.figure()
    fig.set_size_inches(8,8)
    ax = fig.gca()
    
    # Plot the constraints
    for row in range(n):
        plt.plot(x, -x * A[row,0]/A[row,1] + b[row]/A[row,1], color = "black")

    ax.plot(center[0] + radius*np.cos(theta), 
            center[1] + radius*np.sin(theta),
            color = 'red')
    ax.scatter(center[0],
               center[1],
               color = 'red')
    
    ax.set_xlim(lower_lim, upper_lim)
    ax.set_ylim(lower_lim, upper_lim)
    ax.set_title("Chebyshev center of a polygon")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

a1 = np.array([3,1])
a2 = np.array([2,-2])
a3 = np.array([-1,2])
a4 = np.array([-2,-2])

A = np.array([[3,1],
              [2,-2],
              [-1,2],
              [-2,-2]])
A.shape

b = np.ones(4)

r = cp.Variable(1)
xc = cp.Variable(2)

objective = cp.Maximize(r)

constraints = [a1 @ xc + r*math.sqrt(a1@a1) <= b[0],
               a2 @ xc + r*math.sqrt(a2@a2) <= b[1],
               a3 @ xc + r*math.sqrt(a3@a3) <= b[2],
               a4 @ xc + r*math.sqrt(a4@a4) <= b[3]
               ]

prob = cp.Problem(objective, constraints)
result = prob.solve()
print(r.value)
print(xc.value)

# Plotting
radius = r.value
center = np.array(xc.value)
x = np.linspace(-2,2,100)
theta = np.linspace(0,2*math.pi,200)

fig = plt.figure()
fig.set_size_inches(12,12)
ax = fig.gca()

y = -x * a1[0]/a1[1] + b[0]/a1[1]
ax.plot(x, y)
ax.plot(center[0] + radius*np.cos(theta), center[1] + radius*np.sin(theta))

ax.set_xlim(-1,1)
ax.set_ylim(-1,1)

radius, center = solve_chebyshev_problem(A, b)
plot_chebyshev_problem(A, b, radius, center)