from math import sqrt
import numpy as np
from collections import defaultdict
from .cpDistance import calculate_CP_distance_1, calculate_CP_distance_2, calculate_CP_distance_3

#based on # Created by: Prof. Valdecy Pereira, D.Sc.


#Compromise Programming method adapted only to maximise criterion
#values to be passed - dataset that resembles: 
#{'A1': {'C1': {'min': 1, 'max': 1, 'mean': 1.0}, 'C2': {'min': 2, 'max': 2, 'mean': 2.0}}
def cp_deterministic(dataset, ref_values, form = 1, p = 1, criteria_to_analyze=None, weights=None):
    
    """
    Perform a deterministic TOPSIS method for a given dataset with worst, best and medium scenarii.
    
    Parameters:
    - dataset: dict, dataset structured as alternatives with criteria having min/max/mean values.
    - form: 1 is where no normalisation is required and the external ref values are used
            2 is where normalisation is required and external ref values are used
            3 is where normalisation is done and internal ref valeus are used
    - p is a parameter reflecting the decision maker's concern with respect to deviations from the ideal value
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

    # Choose calculation function based on the form
    calculate_CP_distance = {
        1: calculate_CP_distance_1,
        2: calculate_CP_distance_2,
        3: calculate_CP_distance_3
    }[form]

    # Compute distances for min, max, and mean
    if p == float('inf') or str(p).lower() == 'inf':
        results_min = calculate_CP_distance(X_min, p, weights, ref_values)
        results_max = calculate_CP_distance(X_max, p, weights, ref_values)
        results_mean = calculate_CP_distance(X_mean, p, weights, ref_values)
    else:
        results_min = calculate_CP_distance(X_min, p, weights, ref_values)**(1 / p)
        results_max = calculate_CP_distance(X_max, p, weights, ref_values)**(1 / p)
        results_mean = calculate_CP_distance(X_mean, p, weights, ref_values)**(1 / p)

    print("results min")
    print(results_min)

    # Convert results to a dictionary with alternative labels
    results_min = {alt: float(score) for alt, score in zip(alternatives, results_min)}
    results_max = {alt: float(score) for alt, score in zip(alternatives, results_max)}
    results_mean = {alt: float(score) for alt, score in zip(alternatives, results_mean)}

    # Return scores as a dictionary
    return {
        "min": results_min,
        "max": results_max,
        "mean": results_mean
    }

def cp_deterministic_internal(dataset, ref_values, form = 1, p = 1, criteria_to_analyze=None, weights=None):
    
    """
    Perform a deterministic CP method for a given dataset with internal aggregation.
    
    Parameters:
    - dataset: dict, dataset structured as alternatives with criteria having min/max/mean values.
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
    
    # Choose calculation function based on the form
    calculate_CP_distance = {
        1: calculate_CP_distance_1,
        2: calculate_CP_distance_2,
        3: calculate_CP_distance_3
    }[form]

    # Compute distances
    if p == float('inf') or str(p).lower() == 'inf':
        results = calculate_CP_distance(evaluator_array, p, weights, ref_values)

    else:
        results = calculate_CP_distance(evaluator_array, p, weights, ref_values)**(1 / p)


    # Convert results to a dictionary with alternative labels
    results = {alt: float(score) for alt, score in zip(alternatives, results)}

    print("EVALUATOR")
    print(results)

    # Return ranking for all alternatives
    return results

def cp_determ_internal_aggregation(*dataset, ref_values, form = 1, p = 1, criteria_to_analyze=None, weights=None):
    
    """
    Perform a deterministic TOPSIS method with internal aggregation for a given number of evaluations
    
    Parameters:
    - dataset: dict, evaluators dataset structured as alternatives with criteria
    - form: 1 is where no normalisation is required and the external ref values are used
            2 is where normalisation is required and external ref values are used
            3 is where normalisation is done and internal ref valeus are used
    - p is a parameter reflecting the decision maker's concern with respect to deviations from the ideal value
    - ref_values : list of PIS and NIS
    - criteria_to_analyze: list, criteria to include in the analysis (default: all criteria).
    - weights: list, weights for each criterion (default: equal weights)
    
    Returns:
    - dictionary with average, min and max c_i values for each alternative
    """

    evaluations = defaultdict(list)
    for evaluator in dataset:
        # Run deterministic cp for the current evaluator
        cp_scores = cp_deterministic_internal(evaluator, ref_values, form, p, criteria_to_analyze, weights)

        # Store cp scores for each alternative
        for alternative, cp in cp_scores.items():
            evaluations[alternative].append(cp)
    
    # Compute aggregated results
    aggregated_results = {
        "average": {alt: float(np.mean(scores)) for alt, scores in evaluations.items()},
        "min": {alt: float(np.max(scores)) for alt, scores in evaluations.items()}, #in CP smaller values are better!
        "max": {alt: float(np.min(scores)) for alt, scores in evaluations.items()},
    }
    # Return scores
    return aggregated_results

def cp_internal_strategies(input_dataset, ref_values, form=1, p=1, criteria_to_analyze=None, weights=None):
    """
    Process the input dataset to compute the compromise programming scores for each alternative.

    Parameters:
    - input_dataset: dict, the input dataset structured as alternatives with criteria and their statistical values (min, max, mean)
    - ref_values: list of PIS and NIS
    - form: int, normalization and reference values mode (default: 1)
    - p: int, decision maker's concern with deviations (default: 1)
    - criteria_to_analyze: list, criteria to include in the analysis (default: all criteria)
    - weights: list, weights for each criterion (default: equal weights)

    Returns:
    - dict, compromise programming scores (average, min, max) for each alternative
    """
    from collections import defaultdict
    import numpy as np

    # Transform input_dataset into evaluator-style datasets
    evaluators = []
    for key, value in input_dataset.items():
        evaluators.append(value)

    # Call the cp_determ_internal_aggregation function with evaluators
    aggregated_results = cp_determ_internal_aggregation(
        *evaluators,
        ref_values=ref_values,
        form=form,
        p=p,
        criteria_to_analyze=criteria_to_analyze,
        weights=weights
    )

    return aggregated_results
    




