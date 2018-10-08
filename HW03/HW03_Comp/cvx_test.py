import cvxpy as cp
import numpy as np

m = 30
n = 20

np.random.seed(1)
A = np.random.randn(m,n)
b = np.random.randn(m)

x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A*x - b))
constraints = [0 <= x, x <= 1]
prob = cp.Problem(objective, constraints)

result = prob.solve()

print(x.value)


# Problem 1c from HW03

x1 = cp.Variable()
x2 = cp.Variable()

#objective = cp.Minimize(x1 + x2)
#objective = cp.Minimize(cp.max_elemwise(x1,x2))
objective = cp.Minimize(x1**2 + 9 * x2**2)

constraints = [2*x1 + x2 >= 1,
               x1 + 3*x2 >= 1,
               x1 >= 0,
               x2 >= 0      ]

prob = cp.Problem(objective, constraints)
result = prob.solve()
print(x1.value)
print(x2.value)



# Problem 2 from HW03

A = np.array([[13,12,-2],
              [12,17,6],
              [-2,6,12]])
q = np.array([-22,-14.5,30])
r = 1

x = cp.Variable(3)

objective = cp.Minimize(0.5*cp.quad_form(x,A) + q.T*x + r)
constraints = [-1 <= x, x <= 1]
prob = cp.Problem(objective, constraints)

result = prob.solve()

print(x.value)
