import numpy as np

#based on # Created by: Prof. Valdecy Pereira, D.Sc.

# Calculate TOPSIS Euclidean distance for a matrix
def calculate_TP_distance(X, weights, ref_values = None):
    # Reshape X if it's 1D
    if X.ndim == 1:
        X = X.reshape(1, -1)
    #print(X)

    # Weighted normalized decision matrix
    v_ij = X * weights

    # Determine ideal and negative-ideal solutions
    if ref_values is None:
        p_ideal_A = np.max(v_ij, axis=0)
        n_ideal_A = np.min(v_ij, axis=0)
    else:
        p_ideal_A = np.full(v_ij.shape[1], ref_values['PIS'])
        n_ideal_A = np.full(v_ij.shape[1], ref_values['NIS'])

    #print("Positive Ideal Solution (PIS):")
    #print(p_ideal_A)
    #print("Negative Ideal Solution (NIS):")
    #print(n_ideal_A)
    
    # Calculate euclidean distances to ideal and negative-ideal solutions
    p_s_ij = np.sqrt(np.sum((v_ij - p_ideal_A)**2, axis=1))
    n_s_ij = np.sqrt(np.sum((v_ij - n_ideal_A)**2, axis=1))
    #print(p_s_ij)
    #print(n_s_ij)

    # Calculate relative closeness to the ideal solution
    # Avoid divide-by-zero errors
    c_i = n_s_ij / (p_s_ij + n_s_ij + 1e-9)
    #print("c_i")
    #print(c_i)
    return c_i

# Calculate TOPSIS euclidean distance for intervals
def calculate_TP_distance_Int(X_U, X_L, weights, ref_values):
    """
    X_U - upper bound, in (3, 6) it is 6 (for positive secnario)
    X_L - lower bound, in (3, 6) it is 3
    """
    # Reshape X if it's 1D
    if X_U.ndim == 1:
        X_U = X_U.reshape(1, -1)
    
    if X_L.ndim == 1:
        X_L = X_L.reshape(1, -1)

    # Weighted normalized decision matrix
    v_u = X_U * weights
    v_l = X_L * weights

    # Determine ideal and negative-ideal solution
    p_ideal_A = np.full(v_u.shape[1], ref_values['PIS'])
    n_ideal_A = np.full(v_l.shape[1], ref_values['NIS'])

    
    # Calculate euclidean distances to ideal and negative-ideal solutions
    p_s_ij = np.sqrt(np.sum((v_l - p_ideal_A)**2, axis=1))
    n_s_ij = np.sqrt(np.sum((v_u - n_ideal_A)**2, axis=1))


    # Calculate relative closeness to the ideal solution
    # Avoid divide-by-zero errors
    c_i = n_s_ij / (p_s_ij + n_s_ij + 1e-9)
    #print("c_i")
    #print(c_i)
    return c_i

# Calculate TOPSIS Euclidean distance for an array
def calculate_TP_dist_Array(X, weights, ref_values):
    """
    Calculate TOPSIS Euclidean distance and relative closeness for each value in a list.
    
    Parameters:
    - X: ndarray, 1D array of criterion values.
    - weights: list or ndarray, weights for each criterion (not used here for scalar values).
    - ref_values: dict, optional predefined PIS and NIS values.

    Returns:
    - ndarray, relative closeness scores for each value in the input.
    """
    # Ensure X is a 1D array
    X = np.array(X)
    if X.ndim != 1:
        raise ValueError("Input X must be a 1D array.")


    pis = ref_values['PIS']
    nis = ref_values['NIS']

    #print("PIS (Ideal):", pis)
    #print("NIS (Non-Ideal):", nis)

    # Calculate distances to PIS and NIS
    p_distances = np.sqrt((X - pis) ** 2)  # Distance to PIS
    n_distances = np.sqrt((X - nis) ** 2)  # Distance to NIS

    sum_p_distances = np.sum(p_distances)
    sum_n_distances = np.sum(n_distances)

    #print("Distances to PIS:", p_distances)
    #print("Distances to NIS:", n_distances)

    # Calculate relative closeness scores
    closeness_scores = sum_n_distances / (sum_p_distances + sum_n_distances + 1e-9)  # Avoid divide-by-zero

    return closeness_scores

