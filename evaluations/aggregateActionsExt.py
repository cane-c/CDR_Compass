from multiprocessing import Value
from statistics import mean, geometric_mean


def aggregateActionsExt_Average(topsis_data, strategy):
    """
    Aggregates the min, max, and mean values for a given strategy.

    Parameters:
    - topsis_data: dict, structured as {'min': {alternative: value, ...}, 'max': {...}, 'mean': {...}}
    - strategy: set, alternatives to consider for aggregation (e.g., {'A1', 'A2', 'A3'}).

    Returns:
    - dict, aggregated values with keys 'min', 'max', and 'mean'.
    """
    aggregated_results = {}

    # Iterate over the keys ('min', 'max', 'mean')
    for key in topsis_data.keys():
        # Filter the values for the given strategy
        strategy_values = [topsis_data[key][alt] for alt in strategy if alt in topsis_data[key]]
        # Calculate the average
        aggregated_results[key] = sum(strategy_values) / len(strategy_values) if strategy_values else None

    return aggregated_results

def aggregateActionEval_mean(evaluation_matrix, selected_Actions):
    """
    Aggregates evaluations for selected actions based on criteria.
    Using arithemtic mean for all (min, max, mean) values

    Parameters:
    - evaluation_matrix: list of dictionaries, each containing alternatives and criteria evaluations.
    - selected_Actions: set or list of actions to consider for aggregation.

    Returns:
    - Dictionary containing aggregated min, max, and mean for each criterion.
    """


    # Initialize an empty dictionary to store aggregated matrix
    aggregated_matrix = {}

    # Iterate over the criteria of interest
    # Assuming all actions have the same criteria
    criteria = list(next(iter(evaluation_matrix.values())).keys())

    for criterion in criteria:
        # Gather all values for this criterion across selected actions
        min_values = [evaluation_matrix[action][criterion]['min'] for action in selected_Actions]
        max_values = [evaluation_matrix[action][criterion]['max'] for action in selected_Actions]
        mean_values = [evaluation_matrix[action][criterion]['mean'] for action in selected_Actions]

        # Aggregate min, max, and mean for the criterion
        aggregated_matrix[criterion] = {
            'min': mean(min_values),
            'max': mean(max_values),
            'mean': mean(mean_values)
        }


    return aggregated_matrix

def aggregateActionEval_geometricMean(evaluation_matrix, selected_Actions):
    """
    Aggregates evaluations for selected actions based on criteria.
    Using geometric mean for all (min, max, mean) values

    Parameters:
    - evaluation_matrix: list of dictionaries, each containing alternatives and criteria evaluations.
    - selected_Actions: set or list of actions to consider for aggregation.

    Returns:
    - Dictionary containing aggregated min, max, and mean for each criterion.
    """


    # Initialize an empty dictionary to store aggregated matrix
    aggregated_matrix = {}

    # Iterate over the criteria of interest
    # Assuming all actions have the same criteria
    criteria = list(next(iter(evaluation_matrix.values())).keys())

    for criterion in criteria:
        # Gather all values for this criterion across selected actions
        min_values = [evaluation_matrix[action][criterion]['min'] for action in selected_Actions]
        max_values = [evaluation_matrix[action][criterion]['max'] for action in selected_Actions]
        mean_values = [evaluation_matrix[action][criterion]['mean'] for action in selected_Actions]

        # Aggregate min, max, and mean for the criterion
        aggregated_matrix[criterion] = {
            'min': geometric_mean(min_values),
            'max': geometric_mean(max_values),
            'mean': geometric_mean(mean_values)
        }


    return aggregated_matrix

def aggregateActionEval_sum(evaluation_matrix, selected_Actions):
    """
    Aggregates evaluations for selected actions based on criteria.
    Using sum for all (min, max, mean) values

    Parameters:
    - evaluation_matrix: list of dictionaries, each containing alternatives and criteria evaluations.
    - selected_Actions: set or list of actions to consider for aggregation.

    Returns:
    - Dictionary containing aggregated min, max, and mean for each criterion.
    """


    # Initialize an empty dictionary to store aggregated matrix
    aggregated_matrix = {}

    # Iterate over the criteria of interest
    # Assuming all actions have the same criteria
    criteria = list(next(iter(evaluation_matrix.values())).keys())

    for criterion in criteria:
        # Gather all values for this criterion across selected actions
        min_values = [evaluation_matrix[action][criterion]['min'] for action in selected_Actions]
        max_values = [evaluation_matrix[action][criterion]['max'] for action in selected_Actions]
        mean_values = [evaluation_matrix[action][criterion]['mean'] for action in selected_Actions]

        # Aggregate min, max, and mean for the criterion
        aggregated_matrix[criterion] = {
            'min': sum(min_values),
            'max': sum(max_values),
            'mean': sum(mean_values)
        }


    return aggregated_matrix
