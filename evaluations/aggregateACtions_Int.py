import numpy as np
from collections import defaultdict
from collections import OrderedDict

def aggregateEvaluations_Int(*evaluation_dicts):
    """
    Aggregates multiple evaluation dictionaries to compute:
    - Lower fence (minimum value across evaluations)
    - Upper fence (maximum value across evaluations)
    - Q1 (mean of the lowest values in the range)
    - Q3 (mean of the highest values in the range)
    
    Args:
        *evaluation_dicts: Variable number of evaluation dictionaries
    
    Returns:
        dict: Aggregated statistics in the form:
              {'Category': {'Subcategory': {'lower_fence': X, 'upper_fence': Y, 'q1': Z, 'q3': W}}}
    """
    aggregated_data = defaultdict(lambda: defaultdict(list))
    
    # Collect all values from all dictionaries
    for eval_dict in evaluation_dicts:
        for category, subcategories in eval_dict.items():
            for subcat, values in subcategories.items():
                aggregated_data[category][subcat].append(values)
    
    # Compute required statistics
    final_aggregated = {}
    
    for category, subcategories in aggregated_data.items():
        final_aggregated[category] = {}
        for subcat, all_values in subcategories.items():
            # Convert list of lists into NumPy array
            values_array = np.array(all_values)
            
            # Compute statistics
            lower_fence = values_array.min()
            upper_fence = values_array.max()
            q1 = values_array[:, 0].mean()  # Mean of all first elements
            q3 = values_array[:, 1].mean()  # Mean of all second elements
            
            # Store results
            final_aggregated[category][subcat] = {
                'lower_fence': int(lower_fence),
                'upper_fence': int(upper_fence),
                'q1': float(q1),
                'q3': float(q3)
            }
    
    return final_aggregated

import random
import re

import random
import re
from collections import OrderedDict

def aggregate_MC_Int(*expert_data, n):
    """
    Runs a Monte Carlo simulation based on multiple expert interval datasets.
    
    Parameters:
        *expert_data (dicts): Multiple dictionaries of alternatives with expert intervals.
        n (int): The number of iterations for the simulation.
    
    Returns:
        dict: A dictionary containing simulated values for each criterion per alternative.
    """
    simulated_results = OrderedDict()

    # Helper to extract numeric part for sorting
    def extract_number(key):
        match = re.search(r'\d+', key)
        return int(match.group()) if match else float('inf')

    # Gather all unique alternatives and sort them by numeric value after "A"
    alternatives = sorted(
        set().union(*[data.keys() for data in expert_data]),
        key=extract_number
    )

    for alternative in alternatives:
        simulated_results[alternative] = OrderedDict()
        
        # Maintain order of criteria as they appear in the first dataset that has the alternative
        criteria_list = []
        seen_criteria = set()
        
        for data in expert_data:
            if alternative in data:
                for criterion in data[alternative].keys():
                    if criterion not in seen_criteria:
                        seen_criteria.add(criterion)
                        criteria_list.append(criterion)  # Maintain order

        for criterion in criteria_list:
            expert_intervals = []
            
            for data in expert_data:
                if alternative in data and criterion in data[alternative]:
                    expert_intervals.append(data[alternative][criterion])
            
            num_experts = len(expert_intervals)
            samples = []
            
            for _ in range(n):
                U = random.random()  # Generate a random number U in [0,1]
                expert_index = int(U * num_experts)  # Select an expert with equal probability
                chosen_interval = expert_intervals[expert_index]
                
                # If the interval has equal min and max, it's a fixed value
                if chosen_interval[0] == chosen_interval[1]:
                    sample = chosen_interval[0]
                else:
                    sample = random.randint(chosen_interval[0], chosen_interval[1])  # Choose an integer
                
                samples.append(sample)
            
            simulated_results[alternative][criterion] = samples

    return simulated_results






