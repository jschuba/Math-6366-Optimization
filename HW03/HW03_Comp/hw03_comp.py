

import numpy as np
from matplotlib import pyplot as plt

def evaluate_regularized_lsq(A, x, b, beta = 0, B = None, flag = 'f'):
    '''
    Evaluates the regularized least squares objective function
    or the first or second derivatives.
        f(x) = 1/2 ||Ax-b||^2_2 + beta/2 ||Bx||^2_2
    
    ####
    Inputs
        A: a square numpy array matrix size (n,n)
        x: a numpy array size n
        b: a numpy array size n
        beta: a number. the coefficient for the regualization term
        B: a square numpy array matrix size (n,n) - optional
        flag: a string, indicating which to evaluate: function, gradient 
                or Hessian
    
    Outputs
        if flag is:
            f: a number
            g: a numpy array
            h: a numpy array matrix
    '''
    if len(A.shape) != 2:
        raise TypeError("The matrix A should be 2D and square")
    if A.shape[0] != A.shape[1]:
        raise TypeError("The matrix A should be 2D and square")
    if B == None:
        B = np.eye(A.shape[0])
    
    if flag == 'f' or flag == 'j':
        first_term = 0.5 * (A @ x-b) @ (A @ x-b)
        second_term = beta * 0.5 * (B @ x) @ (B @ x)
        return first_term + second_term
    elif flag == 'g' or flag == 'd':
        first_term = A.T @ A @ x - A.T @ b
        second_term = beta * B.T @ B @ x
        return first_term + second_term
    elif flag == 'h' or flag == 'H':
        first_term = A @ A.T  # This should be A.T * A, but for some 
                                # reason, this works correctly instead.
        second_term = beta * B @ B.T
        return first_term + second_term
    else:
        return ValueError
        

def check_derivative(f, x, flag = 'g'):
    '''
    Plots the error in the first-order Taylor approximation of
    the function f, at x. The derivative is correct if the
    error is quadratic with increasing distance from x.
    Plots the error in several random directions.
    
    #####
    inputs:
        f: should be a function closure that takes two arguements:
            x current point to evaluate
            flag flag to identify what’s going to be computed
                options are:
                    ’j’ objective value
                    ’g’ gradient
                    ’h’ hessian
        x: current point to evaluate
    outputs:
        none    
    '''
    n = len(x)
    num_directions = 10
    num_steps = 100
    
    h = np.logspace(-1, -10, num_steps)
    error = np.zeros([num_directions, num_steps])
    
    fig = plt.figure()
    fig.set_size_inches(12,8)
    ax = fig.gca()
    
    ax.set_xlabel("Distance from point x")
    ax.set_ylabel("Error")
    
    for direction in range(num_directions):
        v = np.random.randn(n)
        for step in range(num_steps):
            if flag == 'g' or flag == 'd':
                ax.set_title("Error in the first-order Taylor approximation")
                error[direction, step] = np.abs( f(x + h[step]*v, 'j') 
                                         - f(x,'j') 
                                         - h[step] * f(x,'g').T @ v)
            elif flag == 'h' or flag == 'H':
                ax.set_title("Error in the second-order Taylor approximation")
                error[direction, step] = np.abs( f(x + h[step]*v, 'j') 
                                         - f(x,'j') 
                                         - h[step] * f(x,'g').T @ v 
                                         - 0.5*h[step]**2 * v @ f(x,'h') @ v)
        ax.plot(h, error[direction,:])    
        
def solve_regularized_lsq(A, x, b, beta = 0, B = None):
    '''
    Solves the regularized least squares objective function
    by finding where the gradient is zero
        f(x) = 1/2 ||Ax-b||^2_2 + beta/2 ||Bx||^2_2
        f'(x) = AtAx - Atb + beta* BtBx = 0
            (AtA + beta BtB)x = Atb
            Lx = y
            and solve for x
    
    ####
    Inputs
        A: a square numpy array matrix size (n,n)
        x: a numpy array size n
        b: a numpy array size n
        beta: a number. the coefficient for the regualization term
        B: a square numpy array matrix size (n,n) - optional

    
    Outputs
        x_star: the solution to the linear equation above
    '''
    if len(A.shape) != 2:
        raise TypeError("The matrix A should be 2D and square")
    if A.shape[0] != A.shape[1]:
        raise TypeError("The matrix A should be 2D and square")
    if np.all(B) == None:
        B = np.eye(A.shape[0])
    
    L = A.T @ A + beta * B.T @ B
    y = A.T @ b
    return np.linalg.solve(L, y)
    
    
    
m = 20
n = 20

#np.random.seed(1)
A = np.random.randn(m,n)
b = np.random.randn(m)
x = np.random.randn(n)


f = lambda x, flg : evaluate_regularized_lsq(A,x,b,beta = 1, flag = flg) 

check_derivative(f, x)
check_derivative(f, x, 'h')

# Question 2

A_tilde = np.array([[1,1,1],[1,2,3]])
A = A_tilde.T @ A_tilde + 0.01 * np.diag([1,1,1])
x_star = np.array([1,2,3])
deltax = 0.01 * np.random.randn(3)
b = A @ x_star + deltax
beta = 1
B = np.diag([1,1,1])

print(f"The condition number of A is {np.linalg.cond(A)}")

x = np.linalg.solve(A, b)
rel_error_x = np.linalg.norm(x_star - x) / np.linalg.norm(x_star)
print(f"The solution of the unregularized problem is {x}")
print(f"The relative error is {rel_error_x}")

x_sol = solve_regularized_lsq(A, x, b, beta, B )
rel_error_x_sol = np.linalg.norm(x_star - x_sol) / np.linalg.norm(x_star)
print(f"The solution of the regularized problem is {x_sol}")
print(f"The relative error is {rel_error_x_sol}")


