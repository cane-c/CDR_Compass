import numpy as np

def random_negative_exponential(low, high, mean, size=1):
    """
    Generate random values based on a negative exponential distribution.
    The lowest value (low) is the most probable.

    Parameters:
    - low: minimum value (inclusive).
    - high: maximum value (inclusive).
    - size: number of random samples to generate.

    Returns:
    - np.array of random values.
    """
     # Handle edge case: if min == max and mean == min, always return min_value
    if low == high and mean == low:
        return np.full(size, low, dtype=int)

    # Generate random values using the exponential distribution
    scale = (mean - low)/2  # Scale parameter for the exponential distribution
    raw_values = np.random.exponential(scale, size)
    
    # Adjust values to fit within the min and max range
    adjusted_values = low + raw_values
    
    # Round to nearest integer and clip to the range
    #integer_values = np.clip(np.round(adjusted_values), low, high).astype(int)
    #print(integer_values)
    #return integer_values
    clipped_values = np.clip(adjusted_values, low, high)  # Clip values to range [low, high]
    return np.round(clipped_values).astype(int)  # Round to integers

def random_positive_exponential(low, high, mean, size=1):
    """
    Generate random values based on a positive exponential distribution.
    The highest value (high) is the most probable.

    Parameters:
    - low: minimum value (inclusive).
    - high: maximum value (inclusive).
    - size: number of random samples to generate.

    Returns:
    - np.array of random values.
    """
    # Handle edge case: if min == max and mean == min, always return min_value
    if low == high and mean == low:
        return np.full(size, low, dtype=int)

    # Generate random values using the exponential distribution
    scale = (high - mean) / 2  # Scale parameter for the exponential distribution
    raw_values = np.random.exponential(scale, size)
    
    # Invert and adjust values to fit within the min and max range
    adjusted_values = high - raw_values

    # Round to nearest integer and clip to the range
    #integer_values = np.clip(np.round(adjusted_values), low, high).astype(int)
    #print(integer_values)
    #return integer_values
    clipped_values = np.clip(adjusted_values, low, high)  # Clip values to range [low, high]
    return np.round(clipped_values).astype(int)  # Round to integers

def random_normal(low, high, mean=None, std=None, size=1):
    """
    Generate random values based on a normal distribution.
    The mean value is the most probable.

    Parameters:
    - low: minimum value (inclusive).
    - high: maximum value (inclusive).
    - mean: mean value of the distribution (defaults to midpoint of [low, high]).
    - std: standard deviation (defaults to 1/6th of the range [low, high]).
    - size: number of random samples to generate.

    Returns:
    - np.array of random values.
    """
    if mean is None:
        mean = (low + high) / 2  # Default mean at the midpoint
    if std is None:
        std = (high - low) / 6  # Default std to cover 99.7% of range

    values = np.random.normal(loc=mean, scale=std, size=size)
    clipped_values = np.clip(values, low, high)  # Clip values to range [low, high]
    return np.round(clipped_values).astype(int)  # Round to integers
