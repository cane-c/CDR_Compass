from multiprocessing import Value
from statistics import mean, geometric_mean


def aggregateActionEval(evaluation_matrix, selected_Actions):
    """
    Aggregates evaluations for selected actions based on criteria.
    Using Min - Max method (that is takes the worst and best values in criteria)

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
