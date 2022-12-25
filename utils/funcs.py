import numpy as np

def logit(x):
    return np.log(x / (1e6 - x))

def inverse_logit(x):
    return 1e6 / (1 + np.exp(-x))
