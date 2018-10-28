
import math

import numpy as np
import cvxpy as cp

from scipy import sparse

from matplotlib import pyplot as plt

# Set random seed
np.random.seed(1)


n = 4096
t = list(range(n))
iota = []
for i in t:
    if i in range(1024) or i in range(2049,3072):
        iota.append(1)
    else:
        iota.append(-1)
        

y = [0.5*math.sin(2*math.pi/n * i) for i in t ]
y = np.array(iota) + np.array(y)
y_delta = y + 0.1*np.random.randn(len(y))

bounds = [np.min(y_delta), np.max(y_delta)]

B = sparse.diags(diagonals = [-1*np.ones(n), np.ones(n-1)], 
                 offsets = [0, 1])

num_reg = 100
beta_list = np.logspace(-10, 10, num_reg)
#beta_list = np.logspace(-1, 2, num_reg)

eval_data_error = lambda x, y_delta : np.linalg.norm(x-y_delta)
eval_reg_value = lambda x, B : np.linalg.norm(B@x)


# Part a
objective_values = []
x_star_list = []
data_errors = []
reg_values = []

for beta in beta_list:
    # CVXPY 
    print(f"Solving for beta = {beta}.")
    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm(x-y_delta) + beta*cp.norm(B@x))
    constraints = []
    prob = cp.Problem(objective, constraints)
    try:
        result = prob.solve()
    except Exception as e:
        print("\t" + str(e))
        result = prob.solve(solver="SCS")
    
    if np.any(x.value) == None:
        print("\t Solve failed for some reason.")
        continue
    x_star = np.sum(np.array(x.value),1)
    
    objective_values.append(result)
    x_star_list.append(x.value)
    data_errors.append(eval_data_error(x_star, y_delta))
    reg_values.append(eval_reg_value(x_star, B))

fig = plt.figure()
fig.set_size_inches(8,6)
ax = fig.gca()
ax.plot(data_errors, reg_values)
ax.set_title("Data mismatch vs regularization term")
ax.set_xlabel("Data mismatch")
ax.set_ylabel("Regularization term value")


#%% Solve with the best value for beta
beta_star = 12.9
d_star = 15.784090991770434

x = cp.Variable(n)
objective = cp.Minimize(cp.norm(x-y_delta) + beta_star*cp.norm(B@x))
constraints = []
prob = cp.Problem(objective, constraints)
result = prob.solve(solver="SCS")

x_stars = []
rel_errors = []
x_star = np.sum(np.array(x.value),1)
x_stars.append(x_star)
rel_error = (x_star-y)@(x_star-y) / (y@y)
rel_errors.append( rel_error )

# Solve the constrained problems:
for alpha in [1/3, 1, 3]:
    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm(B@x))
    constraints = [cp.norm(x-y_delta) <= alpha*d_star]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver="SCS")
    x_star = np.sum(np.array(x.value),1)
    x_stars.append(x_star)
    rel_error = (x_star-y)@(x_star-y) / (y@y)
    rel_errors.append( rel_error )
    
fig = plt.figure()
fig.set_size_inches(16,8)
axes = fig.subplots(nrows = 4)
titles = ["Solution with beta_star = 12.9",
          "Solution with 1/3 * regularization",
          "Solution with full regularization",
          "Solution with 3 * regularization"]

for i, ax in enumerate(axes):
    ax.plot(y_delta, label = "Perturbed Signal")
    ax.plot(y, label = "Original Signal")
    ax.plot(x_stars[i], label = "Solution")
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    ax.set_title(titles[i])
    ax.annotate(f"Relative Error = {round(rel_errors[i],4)}", [0, 0.5])
    
ax.legend(loc = 'upper right')    





