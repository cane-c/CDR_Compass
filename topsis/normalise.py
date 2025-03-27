import numpy as np

#convert the dataset into a scale-independent form (normalisation)
#takes matrix as an input
def normalize_matrix(X):
    sum_cols = np.sum(X*X, axis = 0)
    sum_cols = sum_cols**(1/2)
    return X/sum_cols



