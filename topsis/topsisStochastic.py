from collections import defaultdict
import numpy as np

import other
from .topsisDistance import calculate_TP_distance
from .normalise import normalize_matrix
import numpy as np
from evaluations import aggregate_MC_Int
from collections import OrderedDict
import random
import re



#based on # Created by: Prof. Valdecy Pereira, D.Sc.


#TOPSIS method adapted only to maximise criterion
#values to be passed - dataset that resembles: 
#{'A1': {'C1': {'min': 1, 'max': 1, 'mean': 1.0}, 'C2': {'min': 2, 'max': 2, 'mean': 2.0}}
# n is a number of random numbers to be generated
def topsis_MC(dataset, n, normalize=False, ref_values = None, criteria_to_analyze=None, distribution="uniform", weights=None):

    """
    Perform a Monte Carlo simulation with the TOPSIS method for a given dataset.
    
    Parameters:
    - dataset: dict, dataset structured as alternatives with criteria having min/max values.
    - n: int, number of random simulations to perform.
    - normalize: bool, whether to normalize the data, by default it is NO
    - ref_values : list of PIS and NIS
    - criteria_to_analyze: list, criteria to include in the analysis (default: all criteria).
    - distribution : By default it is a uniform. Other values: "positive_exponential", "negative_exponential", "normal"
    - weights: list, weights for each criterion (default: equal weights).
    
    Returns:
    - dict, results with ranking probabilities and average scores for each alternative.
    """

    # Extract specified criteria or use all if criteria_to_analyze is None
    if criteria_to_analyze is None:
        criteria_to_analyze = list(next(iter(dataset.values())).keys())

    # Automatically generate weights if not provided
    if weights is None:
        weights = np.ones(len(criteria_to_analyze))

    alternatives = list(dataset.keys())
    num_alternative = len(alternatives)
    
    # Initialize storage for ranking counts and cumulative closeness values
    ranking_counts = defaultdict(lambda: [0] * num_alternative)
    cumulative_scores = defaultdict(float)

    # To debug: store values for Ax, Cx
    debug_a4_c1_values = []

    for _ in range(n):
        # Simulate values for each alternative on each criterion
        if distribution == "uniform":
            randomised_matrix = np.array([
                [np.random.randint(dataset[alt][crit]['min'], dataset[alt][crit]['max']+1) for crit in criteria_to_analyze] 
                for alt in alternatives
            ]) #generate random integers between min and max values (inclusive)
        elif distribution == "negative_exponential":
            randomised_matrix = np.array([
                [other.random_negative_exponential(
                    dataset[alt][crit]['min'],  # Pass min value
                    dataset[alt][crit]['max'],  # Pass max value
                    dataset[alt][crit]['mean'],
                    size=1                      # Generate one value
                    )[0]  # Extract the single value from the array
                    for crit in criteria_to_analyze
                ]
                for alt in alternatives
            ])  # Generate values using negative exponential distribution
        elif distribution == "positive_exponential":
            randomised_matrix = np.array([
                [other.random_positive_exponential(
                    dataset[alt][crit]['min'],  # Pass min value
                    dataset[alt][crit]['max'],  # Pass max value
                    dataset[alt][crit]['mean'],
                    size=1                      # Generate one value
                    )[0]  # Extract the single value from the array
                    for crit in criteria_to_analyze
                ]
                for alt in alternatives
            ])  # Generate values using positive exponential distribution
        elif distribution == "normal":
            randomised_matrix = np.array([
                [other.random_normal(
                    dataset[alt][crit]['min'],  # Pass min value
                    dataset[alt][crit]['max'],  # Pass max value
                    dataset[alt][crit]['mean'], # Pass max value
                    std = None ,
                    size=1                      # Generate one value
                    )[0]  # Extract the single value from the array
                    for crit in criteria_to_analyze
                ]
                for alt in alternatives
            ])# Generate values using normal distribution
        else:
            raise ValueError("Unsupported distribution type.")

        # Debug: Capture values for A4, C4
        debug_a4_c1_values.append(randomised_matrix[alternatives.index('A4'), criteria_to_analyze.index('C1')])

        #convert the dataset into a scale-independent form (normalisation)
        if normalize:
            randomised_matrix = normalize_matrix(randomised_matrix)
        
        #calculate TOPSIS distance scores
        c_i = calculate_TP_distance(randomised_matrix, weights, ref_values)

        # Rank alternatives by closeness coefficient
        #argsort returns the indices that would sort the array. It indicated the positions of the scores in sorted order
        #e.g. if array is [0.7, 0.8, 0.6] the result is [1, 0, 2]
        rankings = np.argsort(-c_i)  # Higher closeness score = better rank
        
        #following the above example, Output would be: ['A2', 'A1', 'A3']
        ranked_alts = [alternatives[i] for i in rankings]

        # Update ranking counts
        for rank, alt in enumerate(ranked_alts, start=1):
            ranking_counts[alt][rank - 1] += 1  # Increment the count for this rank


        # Convert results to a dictionary with alternative labels
        results = {alt: float(score) for alt, score in zip(alternatives, c_i)}

        # Accumulate closeness scores for each alternative
        for alt, score in zip(alternatives, c_i):
            cumulative_scores[alt] += score

# Prepare final results
    final_results = {}
    for alt in alternatives:
        # Convert ranking counts to percentages
        rank_percentages = {f"rank{r+1}": (count / n) * 100 for r, count in enumerate(ranking_counts[alt])}
        # Calculate average closeness score
        average_closeness = cumulative_scores[alt] / n
        print(alt + "(" + str(average_closeness) + ")")
        # Combine results
        final_results[alt] = {**rank_percentages, "average_c": float(average_closeness)}

    # Debug: Print or visualize collected values for A4, C4
    #print(f"Generated values for A4, C4 over {n} simulations: {debug_a4_c1_values}")

    # Optional: Plot histogram of A4, C4 values
    
    """
    import matplotlib.pyplot as plt
    plt.hist(debug_a4_c1_values, bins=range(dataset['A4']['C1']['min'], dataset['A4']['C1']['max'] + 2), 
             align='left', edgecolor='black')
    plt.title(f"Histogram of Generated Values for A4, C1 ({n} Simulations)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    """

    return final_results

def topsis_MC_Actions(dataset, n, normalize=False, ref_values=None, 
                         criteria_to_analyze=None, distribution="uniform", weights=None):
    """
    Perform a Monte Carlo simulation with the TOPSIS method for a given dataset.

    Parameters:
    - dataset: dict, dataset structured as alternatives with criteria having min/max values.
    - n: int, number of random simulations to perform.
    - normalize: bool, whether to normalize the data, by default it is NO.
    - ref_values: list, reference PIS and NIS values.
    - criteria_to_analyze: list, criteria to include in the analysis (default: all criteria).
    - distribution: str, type of distribution (default: "uniform").
    - weights: list, weights for each criterion (default: equal weights).

    Returns:
    - dict, closeness index values for each alternative over all simulations.
    """
    # Extract all criteria if criteria_to_analyze is not specified
    if criteria_to_analyze is None:
        criteria_to_analyze = list(next(iter(dataset.values())).keys())
    
    # Default weights: equal weights for all criteria
    if weights is None:
        weights = np.ones(len(criteria_to_analyze))

    alternatives = list(dataset.keys())  # List of all alternatives
    num_alternative = len(alternatives)

    # Initialize storage for cumulative closeness values
    alternative_closeness_index = {strat: [] for strat in alternatives}

    for _ in range(n):
        # Generate random values based on the distribution
        if distribution == "uniform":
            randomised_matrix = np.array([
                [
                    np.random.randint(dataset[alt][crit]['min'], dataset[alt][crit]['max'] + 1)
                    for crit in criteria_to_analyze
                ] 
                for alt in alternatives
            ])
        elif distribution == "normal":
            randomised_matrix = np.array([
                [other.random_normal(                 
                    dataset[alt][crit]['min'],  # Pass min value
                    dataset[alt][crit]['max'],  # Pass max value
                    dataset[alt][crit]['mean'], # Pass mean value
                    size=1                      # Generate one value
                    )[0]  # Extract the single value from the array
                    for crit in criteria_to_analyze
                ] 
                for alt in alternatives
            ])
        elif distribution == "negative_exponential":
            randomised_matrix = np.array([
                [other.random_negative_exponential(
                    dataset[alt][crit]['min'],  # Pass min value
                    dataset[alt][crit]['max'],  # Pass max value
                    dataset[alt][crit]['mean'], # Pass mean value
                    size=1                      # Generate one value
                    )[0]  # Extract the single value from the array
                    for crit in criteria_to_analyze
                ]
                for alt in alternatives
            ])  # Generate values using negative exponential distribution
        elif distribution == "positive_exponential":
            randomised_matrix = np.array([
                [other.random_positive_exponential(
                    dataset[alt][crit]['min'],  # Pass min value
                    dataset[alt][crit]['max'],  # Pass max value
                    dataset[alt][crit]['mean'], # Pass max value
                    size=1                      # Generate one value
                    )[0]  # Extract the single value from the array
                    for crit in criteria_to_analyze
                ]
                for alt in alternatives
             ])
        else:
            raise ValueError("Unsupported distribution type. Supported: 'uniform', 'normal'.")
        
        # Optional normalization
        if normalize:
            randomised_matrix = normalize_matrix(randomised_matrix)
        
        # Calculate closeness index using TOPSIS distance
        c_i = calculate_TP_distance(randomised_matrix, weights, ref_values)

        # Store scores for each strategy
        for alt, score in zip(alternatives, c_i):
            alternative_closeness_index[alt].append(float(score))
    
    return alternative_closeness_index


def topsis_MC_Int_Actions(*expert_data, ref_values, n, criteria_to_analyze=None):
    """
    Runs a Monte Carlo TOPSIS simulation based on multiple expert interval datasets.
    
    Parameters:
        *expert_data (dicts): Multiple dictionaries of alternatives with expert intervals.
        ref_values (dict): PIS and NIS values.
        n (int): Number of iterations for the simulation.
        criteria_to_analyze (list): Criteria to include in the analysis (default: all criteria).
    
    Returns:
        dict: Closeness index values for each alternative over all simulations.
    """
    alternative_closeness_index = {alt: [] for alt in set().union(*[data.keys() for data in expert_data])}
    
    # Helper to extract numeric part for sorting
    def extract_number(key):
        match = re.search(r'\d+', key)
        return int(match.group()) if match else float('inf')

    # Build a sorted list of alternatives from your expert data
    alternatives_sorted = sorted(set().union(*[data.keys() for data in expert_data]),
                                 key=extract_number)
    alternative_closeness_index = OrderedDict((alt, []) for alt in alternatives_sorted)


    for _ in range(n):
        simulated_results = aggregate_MC_Int(*expert_data, n=1)  # Generate one iteration at a time
        
        # Convert simulated results into a matrix
        randomised_matrix = []
        alternatives = list(simulated_results.keys())
        
        for alternative in alternatives:
            row = []
            for criterion in (criteria_to_analyze if criteria_to_analyze else simulated_results[alternative].keys()):
                row.append(simulated_results[alternative][criterion][0])  # Take the single simulation value
            randomised_matrix.append(row)

        
        randomised_matrix = np.array(randomised_matrix)

        weights = np.ones(len(criteria_to_analyze)) if criteria_to_analyze else np.ones(randomised_matrix.shape[1])
        
        # Calculate closeness index using TOPSIS
        c_i = calculate_TP_distance(randomised_matrix, weights, ref_values=ref_values)
        
        # Store scores for each alternative
        for alt, score in zip(alternatives, c_i):
            alternative_closeness_index[alt].append(round(float(score), 3))
    

    return alternative_closeness_index


def topsis_MC_Int_Ranking(*expert_data, ref_values, n, criteria_to_analyze=None):
    """
    Runs a Monte Carlo TOPSIS probability estilation based on multiple expert interval datasets.
    
    Parameters:
        *expert_data (dicts): Multiple dictionaries of alternatives with expert intervals.
        ref_values (dict): PIS and NIS values.
        n (int): Number of iterations for the simulation.
        criteria_to_analyze (list): Criteria to include in the analysis (default: all criteria).
    
    Returns:
        dict: results with ranking probabilities and average scores for each alternative.

    """
    alternative_closeness_index = {alt: [] for alt in set().union(*[data.keys() for data in expert_data])}
        # Helper to extract numeric part for sorting
    def extract_number(key):
        match = re.search(r'\d+', key)
        return int(match.group()) if match else float('inf')

    # Build a sorted list of alternatives from your expert data
    alternatives_sorted = sorted(set().union(*[data.keys() for data in expert_data]),
                                 key=extract_number)
    alternative_closeness_index = OrderedDict((alt, []) for alt in alternatives_sorted)

    alternatives = list(alternative_closeness_index.keys())
    num_alternatives = len(alternatives)
    ranking_counts = defaultdict(lambda: [0] * num_alternatives)
    cumulative_scores = defaultdict(float)
    
    for _ in range(n):
        simulated_results = aggregate_MC_Int(*expert_data, n=1)
        
        randomised_matrix = []
        for alternative in alternatives:
            row = [simulated_results[alternative][criterion][0] for criterion in (criteria_to_analyze if criteria_to_analyze else sorted(simulated_results[alternative].keys()))]
            randomised_matrix.append(row)
        
        randomised_matrix = np.array(randomised_matrix)
        weights = np.ones(len(criteria_to_analyze)) if criteria_to_analyze else np.ones(randomised_matrix.shape[1])
        
        c_i = calculate_TP_distance(randomised_matrix, weights=weights, ref_values=ref_values)
        rankings = np.argsort(-c_i)
        ranked_alts = [alternatives[i] for i in rankings]
        
        for rank, alt in enumerate(ranked_alts, start=1):
            ranking_counts[alt][rank - 1] += 1
        
        for alt, score in zip(alternatives, c_i):
            alternative_closeness_index[alt].append(float(score))
            cumulative_scores[alt] += score
    
    final_results = {}
    for alt in alternatives:
        rank_percentages = {f"rank{r+1}": round((count / n) * 100, 3) for r, count in enumerate(ranking_counts[alt])}
        average_closeness = round(cumulative_scores[alt] / n, 3)
        final_results[alt] = {**rank_percentages, "average_c": average_closeness}
    
    return final_results



