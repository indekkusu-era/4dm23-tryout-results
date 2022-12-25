import numpy as np
from scipy.optimize import linprog

# ChatGPT wtf are you doing
def maximize_tournament_score(s, w, n):
    k, m = s.shape
    # Define the objective function
    c = [-sum([w[j]*s[i][j] for j in range(m)]) for i in range(k)]

    # Define the constraints
    A = np.ones((1,k))
    b = [n]
    bounds = [(0,1)]*k

    # Solve the optimization problem
    res = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method='simplex')

    return res
