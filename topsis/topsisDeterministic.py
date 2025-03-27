import numpy as np
from .topsisDistance import calculate_TP_distance, calculate_TP_distance_Int
from .normalise import normalize_matrix
from collections import defaultdict

#based on # Created by: Prof. Valdecy Pereira, D.Sc.


#TOPSIS method adapted only to maximise criterion
#values to be passed - dataset that resembles: 
#{'A1': {'C1': {'min': 1, 'max': 1, 'mean': 1.0}, 'C2': {'min': 2, 'max': 2, 'mean': 2.0}}
def topsis_deterministic(dataset, normalize=False, ref_values = None, criteria_to_analyze=None, weights=None):
    
    """
    Perform a deterministic TOPSIS method for a given dataset with worst, best and medium scenarii.
    
    Parameters:
    - dataset: dict, dataset structured as alternatives with criteria having min/max/mean values.
    - normalize: bool, whether to normalize the data, by deafult it is NOT.
    - ref_values : list of PIS and NIS
    - criteria_to_analyze: list, criteria to include in the analysis (default: all criteria).
    - weights: list, weights for each criterion (default: equal weights).
    
    Returns:
    - dictionary with min, max and mean values for each alternative
    """
    
    # Extract specified criteria or use all if criteria_to_analyze is None
    if criteria_to_analyze is None:
        criteria_to_analyze = list(next(iter(dataset.values())).keys())
 
    # Prepare X_min, X_max, X_mean based on the selected criteria
    alternatives = list(dataset.keys())
    X_min = np.array([[dataset[alt][crit]['min'] for crit in criteria_to_analyze] for alt in alternatives])
    X_max = np.array([[dataset[alt][crit]['max'] for crit in criteria_to_analyze] for alt in alternatives])
    X_mean = np.array([[dataset[alt][crit]['mean'] for crit in criteria_to_analyze] for alt in alternatives])
    print(X_min)

    # Automatically generate weights if not provided
    if weights is None:
        weights = np.ones(len(criteria_to_analyze))

    #convert the dataset into a scale-independent form (normalisation)
    if normalize:       
        X_min = normalize_matrix(X_min)
        X_max = normalize_matrix(X_max)
        X_mean = normalize_matrix(X_mean)
    
    # Calculate scores for min, max, and mean datasets
    c_i_min = calculate_TP_distance(X_min, weights, ref_values)
    c_i_max = calculate_TP_distance(X_max, weights, ref_values)
    c_i_mean = calculate_TP_distance(X_mean, weights, ref_values)

    # Convert results to a dictionary with alternative labels
    results_min = {alt: float(score) for alt, score in zip(alternatives, c_i_min)}
    results_max = {alt: float(score) for alt, score in zip(alternatives, c_i_max)}
    results_mean = {alt: float(score) for alt, score in zip(alternatives, c_i_mean)}

    # Return scores as a dictionary
    return {
        "min": results_min,
        "max": results_max,
        "mean": results_mean
    }

def topsis_deterministic_internal(dataset, normalize=False, ref_values = None, criteria_to_analyze=None, weights=None):
    
    """
    Perform a deterministic TOPSIS method for a given dataset with internal aggregation.
    
    Parameters:
    - dataset: dict, dataset structured as alternatives with criteria having 1 value on each criteria.
    - normalize: bool, whether to normalize the data, by deafult it is NOT.
    - ref_values : list of PIS and NIS
    - criteria_to_analyze: list, criteria to include in the analysis (default: all criteria).
    - weights: list, weights for each criterion (default: equal weights).
    
    Returns:
    - list of c_i scores for all alternatives
    """
    
    
    # Extract specified criteria or use all if criteria_to_analyze is None
    if criteria_to_analyze is None:
        criteria_to_analyze = list(next(iter(dataset.values())).keys())
 
    # Prepare array on the selected criteria
    alternatives = list(dataset.keys())
    evaluator_array = np.array([[dataset[alt][crit] for crit in criteria_to_analyze] for alt in alternatives])


    # Automatically generate weights if not provided
    if weights is None:
        weights = np.ones(len(criteria_to_analyze))

    #convert the dataset into a scale-independent form (normalisation)
    if normalize:       
        evaluator_array = normalize_matrix(evaluator_array)
    
    # Calculate scores for min, max, and mean datasets
    c_i = calculate_TP_distance(evaluator_array, weights, ref_values)

    # Convert results to a dictionary with alternative labels
    results = {alt: float(score) for alt, score in zip(alternatives, c_i)}
    #print("EVALUATOR")
    #print(results)

    # Return c_i scores for all alternatives
    return results
    
def topsis_determ_internal_aggregation(*dataset, normalize=False, ref_values = None, criteria_to_analyze=None, weights=None):
    
    """
    Perform a deterministic TOPSIS method with internal aggregation for a given number of evaluations
    
    Parameters:
    - dataset: dict, evaluators dataset structured as alternatives with criteria
    - normalize: bool, whether to normalize the data, by deafult it is NOT.
    - ref_values : list of PIS and NIS
    - criteria_to_analyze: list, criteria to include in the analysis (default: all criteria).
    - weights: list, weights for each criterion (default: equal weights).
    
    Returns:
    - dictionary with average, min and max c_i values for each alternative
    """

    evaluations = defaultdict(list)
    for evaluator in dataset:
        # Run deterministic TOPSIS for the current evaluator
        ci_scores = topsis_deterministic_internal(evaluator, normalize, ref_values, criteria_to_analyze, weights)

        # Store CI scores for each alternative
        for alternative, ci in ci_scores.items():
            evaluations[alternative].append(ci)
    
    # Compute aggregated results
    aggregated_results = {
        "average_ci": {alt: float(np.mean(scores)) for alt, scores in evaluations.items()},
        "min_ci": {alt: float(np.min(scores)) for alt, scores in evaluations.items()},
        "max_ci": {alt: float(np.max(scores)) for alt, scores in evaluations.items()},
    }
    # Return scores
    return aggregated_results

##TOPSIS method adapted only to maximise criterion for interval values
#values to be passed - dataset that resembles: 
# {'A1': {'EN1': {'lower_fence': np.int64(6), 'upper_fence': np.int64(7), 'q1': np.float64(6.5), 'q3': np.float64(7.0)}
def topsis_deterministic_Int(
    dataset, 
    ref_values,
    normalize=False, 
    criteria_to_analyze=None, 
    weights=None
):
    """
    Perform a deterministic TOPSIS method with uncertainty ranges.

    Parameters:
    - dataset: dict, structured as {'A1': {'EN1': {'lower_fence': x, 'upper_fence': y, 'q1': z, 'q3': w}, ...}}
    - normalize: bool, whether to normalize the data.
    - ref_values: list, reference values (PIS and NIS).
    - criteria_to_analyze: list, criteria to include (default: all).
    - weights: list, weights for each criterion (default: equal weights).

    Returns:
    - dict with TOPSIS scores for lower_fence, upper_fence, q1, q3, and midpoint.
    """

    #Extract criteria (if not provided)
    if criteria_to_analyze is None:
        criteria_to_analyze = list(next(iter(dataset.values())).keys())

    alternatives = list(dataset.keys())

    # Prepare matrices for each metric (convert values to float to avoid dtype issues)
    X_lower_fence = np.array([[float(dataset[alt][crit]['lower_fence']) for crit in criteria_to_analyze] for alt in alternatives])
    #print(X_lower_fence)
    #print("")
    X_upper_fence = np.array([[float(dataset[alt][crit]['upper_fence']) for crit in criteria_to_analyze] for alt in alternatives])
    #print(X_upper_fence)
    X_q1 = np.array([[float(dataset[alt][crit]['q1']) for crit in criteria_to_analyze] for alt in alternatives])
    X_q3 = np.array([[float(dataset[alt][crit]['q3']) for crit in criteria_to_analyze] for alt in alternatives])
    X_midpoint = (X_q1 + X_q3) / 2  # Midpoint between q1 and q3

    #Handle weights
    if weights is None:
        weights = np.ones(len(criteria_to_analyze))

    #Normalize if requested
    if normalize:
        X_lower_fence = normalize_matrix(X_lower_fence)
        X_upper_fence = normalize_matrix(X_upper_fence)
        X_q1 = normalize_matrix(X_q1)
        X_q3 = normalize_matrix(X_q3)
        X_midpoint = normalize_matrix(X_midpoint)

    #Calculate TOPSIS distances (based on (Jahanshahloo et al., 2006))
    c_i_optimistic = calculate_TP_distance_Int(X_upper_fence, X_lower_fence, weights, ref_values)
    #c_i_pessimistic = calculate_TP_distance_Int(X_lower_fence, X_upper_fence, weights, ref_values)
    c_i_positive = calculate_TP_distance_Int(X_q3, X_q1, weights, ref_values)
    #c_i_negative = calculate_TP_distance_Int(X_q1, X_q3, weights, ref_values)
    #c_i_midpoint = calculate_TP_distance(X_midpoint, weights, ref_values)

    #Convert results to dictionary with alternative labels
    def to_dict(scores):
        return {alt: round(float(score), 3) for alt, score in zip(alternatives, scores)}

    return {
        "range-based scenario": to_dict(c_i_optimistic),
        #"pessimistic scenario": to_dict(c_i_pessimistic),
        "weighted (average) scenario": to_dict(c_i_positive),
        #"pessimistic weighted": to_dict(c_i_negative),
        #"weighted average": to_dict(c_i_midpoint)
    }

    




