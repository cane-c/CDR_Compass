import numpy as np


# Calculate Compromise programming distances for a matrix

# Form 1: Directly applicable when ideal values are known and criteria are on a comparable scale or normalized beforehand.
def calculate_CP_distance_1(X, p, weights, ref_values):
    
    """Calculate the CP distance for Form 1 (no normalization)."""
    # Using ref_values directly as the ideal point
    PIS = ref_values['PIS']
    #print(PIS)
    normalized_X = np.abs(PIS - X)
    #print("Normalised:")
    #print(normalized_X)
    #print("Weights:")
    #print(weights)
    if p == float('inf') or str(p).lower() == 'inf':
        return np.max(weights * normalized_X, axis=1)
    else:
        p = float(p)
        # Handle finite p (p-norm)
        return np.sum(weights * (normalized_X **p), axis=1)

#Form 2: Suitable for cases with different scales or units of criteria, as it normalizes deviations within the metric.
def calculate_CP_distance_2(X, p, weights, ref_values):
    
    """Calculate the CP distance for Form 2 (external normalization)."""
    # Normalize using external reference values
    PIS = ref_values['PIS']
    NIS = ref_values['NIS']
    normalized_X = np.abs((PIS - X) / (PIS - NIS))

    if np.isinf(p):
        return np.max(weights * normalized_X, axis=1)
    else:
        # Handle finite p (p-norm)
        return np.sum(weights * (normalized_X **p), axis=1)


#Form 3: Ideal for scenarios where ideal and anti-ideal values are derived from the set of alternatives. It normalizes deviations and accommodates different scales or units of criteria..
def calculate_CP_distance_3(X, p, weights, ref_values):
    
    PIS = np.max(X, axis=0)
    NIS = np.min(X, axis=0)
    normalized_X = np.abs((PIS - X) / (PIS - NIS))

    if np.isinf(p):
        return np.max(weights * normalized_X, axis=1)
    else:
        # Handle finite p (p-norm)
        return np.sum(weights * (normalized_X **p), axis=1)

# Calculate CP distance for an array
def calculate_CP_dist_Array(X, weights, ref_values):
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

