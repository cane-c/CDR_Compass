import numpy as np

def transform_to_Array(data):
    """
    Transform the dictionary input into a NumPy array suitable for the calculate_TP_distance function.

    Parameters:
    - data: dict, where keys are alternatives (e.g., 'A1', 'A2') and values are dictionaries of criteria (e.g., 'C1', 'C2').

    Returns:
    - np.array: A 2D array with rows corresponding to alternatives and columns corresponding to criteria values.
    """
    # Extract criteria keys to maintain consistent column order
    criteria = list(next(iter(data.values())).keys())

    # Create the 2D array by iterating over alternatives and extracting values in criteria order
    matrix = np.array([[data[alt][crit] for crit in criteria] for alt in data.keys()])

    return matrix





