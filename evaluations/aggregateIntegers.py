from multiprocessing import Value
from statistics import mean


def aggregateEvaluations(*evaluation_matrixes):
    # Initialize an empty dictionary to store aggregated matrix
    aggregated_matrix = {}

    # Get the list of alternatives from the first evaluation data
    alternatives = evaluation_matrixes[0].keys()

    # Iterate over each alternative for each evaluation matrix
    for alternative in alternatives:

        # Initialize a nested dictionary to store criteria for this alternative
        aggregated_matrix[alternative] = {}

        # Get the list of criteria from the first evaluation data for this alternative
        criteria = evaluation_matrixes[0][alternative].keys()

        # Iterate over each criterion for the current alternative
        for criterion in criteria:
            # Gather all values for this criterion across all evaluation matrixes
            criterion_values = [eval_data[alternative][criterion] for eval_data in evaluation_matrixes]

            # Calculate min, max, and mean for the criterion
            x1 = min(criterion_values)
            x2 = max(criterion_values)
            x3 = mean(criterion_values)

            # Store the aggregated results
            aggregated_matrix[alternative][criterion] = {
                'min': x1,
                'max': x2,
                'mean': x3
            }

    return aggregated_matrix
