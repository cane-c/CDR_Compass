import numpy as np

def generate_probability_distribution(intervals):
    """
    Generate a probability distribution based on evaluators' interval judgments.
    :param intervals: List of tuples representing evaluators' intervals (e.g., [(2,3), (2,5)]).
    :return: Dictionary representing probability distribution.
    """
    min_val = min(interval[0] for interval in intervals)
    max_val = max(interval[1] for interval in intervals)
    
    # Count occurrences for each value in the merged range
    value_counts = {i: 0 for i in range(min_val, max_val + 1)}
    for interval in intervals:
        for i in range(interval[0], interval[1] + 1):
            value_counts[i] += 1
    
    # Normalize to get probabilities
    total = sum(value_counts.values())
    probability_distribution = {k: v / total for k, v in value_counts.items()}
    
    return probability_distribution

def sample_performance(prob_distribution, num_samples=1000):
    values, probabilities = zip(*prob_distribution.items())
    return np.random.choice(values, size=num_samples, p=probabilities)

# Define evaluator intervals
c1_intervals = [(2, 3), (2, 5)]  # Overlapping intervals
c2_intervals = [(5, 5), (6, 7)]  # Exact value + interval

# Generate probability distributions
c1_distribution = generate_probability_distribution(c1_intervals)
c2_distribution = generate_probability_distribution(c2_intervals)

# Generate samples
num_samples = 1000
c1_samples = sample_performance(c1_distribution, num_samples)
c2_samples = sample_performance(c2_distribution, num_samples)

# Compute expected values
c1_expected = sum(k * v for k, v in c1_distribution.items())
c2_expected = sum(k * v for k, v in c2_distribution.items())

print(f"Probability Distribution for c1: {c1_distribution}")
print(f"Probability Distribution for c2: {c2_distribution}")
print(f"Expected Value for c1: {c1_expected}")
print(f"Expected Value for c2: {c2_expected}")
