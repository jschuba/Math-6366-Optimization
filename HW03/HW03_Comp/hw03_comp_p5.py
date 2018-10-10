import math

import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

A = np.array([[1,2,0,1],
              [0,0,3,1],
              [0,3,1,1],
              [2,1,2,5],
              [1,0,3,2] ])

c_max = np.array([100,100,100,100,100])
p = np.array([3,2,7,6])
p_t = np.array([2,1,4,2])
q = np.array([4,10,5,10])

n = 4

r = cp.Variable(n)
x = cp.Variable(n)

objective = cp.Maximize(cp.sum_entries(r))

constraints = [ A @ x <= c_max]

for j in range(n):
    constraints.append( r[j] <= p[j]*x[j] )
    constraints.append( r[j] <= p[j]*q[j] + p_t[j]*(x[j] - q[j]) )

prob = cp.Problem(objective, constraints)
result = prob.solve()

print ("The ideal product production is ")
print ( x.value)
print ("The revenue per product is")
print ( r.value)
print ("The total revenue is: ", result )
print ("The average price for each product is: ")
print ( np.divide(r.value, x.value))
print ("The material usage is: ")
print ( A @ x.value)


