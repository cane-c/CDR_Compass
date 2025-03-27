from collections import defaultdict
import numpy as np
import other
from .topsisDistance import calculate_TP_dist_Array, calculate_TP_distance
from .normalise import normalize_matrix
from .topsisStochastic import topsis_MC_Actions
from evaluations import aggregate_MC_Int

#based on # Created by: Prof. Valdecy Pereira, D.Sc.


#TOPSIS method adapted only to maximise criterion
#values to be passed - dataset that resembles: 
#{'A1': {'C1': {'min': 1, 'max': 1, 'mean': 1.0}, 'C2': {'min': 2, 'max': 2, 'mean': 2.0}}
# n is a number of random numbers to be generated
def topsis_MC_Strategy(dataset, strategy, n, ref_values, normalize=False, criteria_to_analyze=None, distribution="uniform", weights=None):
    """
    Perform a Monte Carlo simulation with the TOPSIS method for a given strategy.

    Parameters:
    - dataset: dict, dataset structured as alternatives with criteria having min/max values.
    - strategy: set or list, subset of alternatives to analyze.
    - n: int, number of random simulations to perform.
    - normalize: bool, whether to normalize the data (default: False).
    - ref_values: dict, reference values for PIS and NIS - in this case ref values MUST BE provided, 
    otherwise there is nothing to comapre a strategy with.
    - criteria_to_analyze: list, criteria to include in the analysis (default: all criteria).
    - weights: list, weights for each criterion (default: equal weights).

    Returns:
    - list, closeness index values for the strategy over all simulations.
    """

    # Extract specified criteria or use all if criteria_to_analyze is None
    if criteria_to_analyze is None:
        criteria_to_analyze = list(next(iter(dataset.values())).keys())

    # Automatically generate weights if not provided
    if weights is None:
        weights = np.ones(len(criteria_to_analyze))

    # Convert the strategy to a list to iterate over alternatives
    strategy_alternatives = list(strategy)

    # Storage for closeness indices over simulations
    strategy_closeness_scores = []

    for _ in range(n):
        # Simulate values for each alternative in the strategy on each criterion
        if distribution == "uniform":
            simulated_values = np.array([
                [np.random.randint(dataset[alt][crit]['min'], dataset[alt][crit]['max']+1) for crit in criteria_to_analyze] 
                for alt in strategy_alternatives
            ]) #generate random integers between min and max values (inclusive)
        elif distribution == "negative_exponential":
            simulated_values = np.array([
                [other.random_negative_exponential(
                    dataset[alt][crit]['min'],  # Pass min value
                    dataset[alt][crit]['max'],  # Pass max value
                    size=1                      # Generate one value
                    )[0]  # Extract the single value from the array
                    for crit in criteria_to_analyze
                ]
                for alt in strategy_alternatives
            ])  # Generate values using negative exponential distribution
        elif distribution == "positive_exponential":
            simulated_values = np.array([
                [other.random_positive_exponential(
                    dataset[alt][crit]['min'],  # Pass min value
                    dataset[alt][crit]['max'],  # Pass max value
                    size=1                      # Generate one value
                    )[0]  # Extract the single value from the array
                    for crit in criteria_to_analyze
                ]
                for alt in strategy_alternatives
            ])  # Generate values using positive exponential distribution
        elif distribution == "normal":
            simulated_values = np.array([
                [other.random_normal(
                    dataset[alt][crit]['min'],  # Pass min value
                    dataset[alt][crit]['max'],  # Pass max value
                    dataset[alt][crit]['mean'], # Pass max value
                    std = None ,
                    size=1                      # Generate one value
                    )[0]  # Extract the single value from the array
                    for crit in criteria_to_analyze
                ]
                for alt in strategy_alternatives
            ])  # Generate values using normal distribution
        else:
            raise ValueError("Unsupported distribution type.")

        # Calculate the mean value for each criterion across the strategy
        mean_criterion_values = simulated_values.mean(axis=0)

        # Normalize if required
        if normalize:
            mean_criterion_values = normalize_matrix(mean_criterion_values.reshape(1, -1)).flatten()

        # Calculate the closeness index for the strategy
        strategy_closeness = calculate_TP_dist_Array(mean_criterion_values, weights, ref_values)
        
        # Store the closeness score
        strategy_closeness_scores.append(float(strategy_closeness))

    return strategy_closeness_scores

def topsis_MC_Strategies(dataset, n, normalize=False, ref_values = None, criteria_to_analyze=None, distribution="uniform", weights=None):

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
    - list, closeness index values for the strategy over all simulations.    
    """

    # Extract specified criteria or use all if criteria_to_analyze is None
    if criteria_to_analyze is None:
        criteria_to_analyze = list(next(iter(dataset.values())).keys())

    # Automatically generate weights if not provided
    if weights is None:
        weights = np.ones(len(criteria_to_analyze))

    strategies = list(dataset.keys())
    num_strategies = len(strategies)
    
    # Initialize storage for ranking counts and cumulative closeness values
    strategy_closeness_index = {strat: [] for strat in strategies}
   
    for _ in range(n):
        # Simulate values for each alternative in the strategy on each criterion
        if distribution == "uniform":
            randomised_matrix = np.array([
                [np.random.randint(dataset[alt][crit]['min'], dataset[alt][crit]['max']+1) for crit in criteria_to_analyze] 
                for alt in strategies
            ]) #generate random integers between min and max values (inclusive)
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
                for alt in strategies
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
                for alt in strategies
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
                for alt in strategies
            ])  # Generate values using normal distribution
        else:
            raise ValueError("Unsupported distribution type.")
        
        #convert the dataset into a scale-independent form (normalisation)
        if normalize:
            randomised_matrix = normalize_matrix(randomised_matrix)

        #calculate TOPSIS distance scores
        c_i = calculate_TP_distance(randomised_matrix, weights, ref_values)

        for strat, score in zip(strategies, c_i):
            strategy_closeness_index[strat].append(float(score))
 
    return strategy_closeness_index

def topsis_MC_criteriaFamilies_Strat(dataset, n, normalize=False, ref_values=None, criteria_families=None, weights=None):
    """
    Perform a Monte Carlo simulation with the TOPSIS method for a given dataset.
    
    Parameters:
    - dataset: dict, dataset structured as alternatives with criteria having min/max values.
    - n: int, number of random simulations to perform.
    - normalize: bool, whether to normalize the data, by default it is False.
    - ref_values: list of PIS and NIS values.
    - criteria_families: dict, mapping of families (e.g., "env", "soc") to their criteria sets.
    - weights: list, weights for each criterion (default: equal weights).
    
    Returns:
    - dict, containing closeness index values for each family and the total closeness index over all simulations.
    """
    # If no criteria families are provided, use all criteria as a single family
    if criteria_families is None:
        criteria_families = {"total": list(next(iter(dataset.values())).keys())}
    
    # Automatically generate weights if not provided
    if weights is None:
        total_criteria = list(next(iter(dataset.values())).keys())
        weights = {family: np.ones(len(criteria)) for family, criteria in criteria_families.items()}
        weights["total"] = np.ones(len(total_criteria))
    
    strategies = list(dataset.keys())
    
    # Initialize storage for closeness index values for each family and the total
    strategy_closeness_index = {strat: {f"c_i_{family}": [] for family in criteria_families} for strat in strategies}
    strategy_closeness_index["total"] = {strat: [] for strat in strategies}

    for _ in range(n):
        # Simulate values for each strategy on each criterion
        randomised_matrix = np.array([
            [np.random.uniform(dataset[strat][crit]['min'], dataset[strat][crit]['max']+1) for crit in criteria_families["total"]]
            for strat in strategies
        ])  # Generate random floats between min and max values
        
        # Normalize the dataset into a scale-independent form (if required)
        if normalize:
            randomised_matrix = normalize_matrix(randomised_matrix)
        
        # Calculate `c_i` for each family
        family_scores = {}
        for family, criteria in criteria_families.items():
            # Find indices of the criteria in the family
            family_indices = [criteria_families["total"].index(crit) for crit in criteria]
            
            # Extract the sub-matrix for the current family
            family_matrix = randomised_matrix[:, family_indices]
            
            # Calculate TOPSIS distance scores for the family
            c_i_family = calculate_TP_distance(family_matrix, weights[family], ref_values)
            family_scores[family] = c_i_family
        
        # Calculate the overall `c_i_total` score
        c_i_total = calculate_TP_distance(randomised_matrix, weights["total"], ref_values)

        # Store the results
        for strat_idx, strat in enumerate(strategies):
            for family, c_i_family in family_scores.items():
                strategy_closeness_index[strat][f"c_i_{family}"].append(float(c_i_family[strat_idx]))
            strategy_closeness_index["total"][strat].append(float(c_i_total[strat_idx]))

    return strategy_closeness_index

def topsis_MC_Strategies_From_Actions(dataset, strategies, n, normalize=False, ref_values=None,
                                      criteria_to_analyze=None, distribution="uniform", weights=None):
    """
    Perform a Monte Carlo simulation with the TOPSIS method for given strategies based on alternatives.

    Parameters:
    - dataset: dict, dataset structured as alternatives with criteria having min/max values.
    - strategies: dict, mapping of strategy names to sets of alternatives.
    - n: int, number of random simulations to perform.
    - normalize: bool, whether to normalize the data, by default it is NO.
    - ref_values: list, reference PIS and NIS values.
    - criteria_to_analyze: list, criteria to include in the analysis (default: all criteria).
    - distribution: str, type of distribution (default: "uniform").
    - weights: list, weights for each criterion (default: equal weights).

    Returns:
    - dict, averaged closeness index values for each strategy over all simulations.
    """
    # Calculate closeness indices for individual alternatives
    alternative_closeness = topsis_MC_Actions(
        dataset=dataset,
        n=n,
        normalize=normalize,
        ref_values=ref_values,
        criteria_to_analyze=criteria_to_analyze,
        distribution=distribution,
        weights=weights
    )
    
    # Initialize result dictionary for strategies
    strategy_closeness = {strategy: [] for strategy in strategies}

    # Compute closeness indices for each strategy
    for strategy_name, alternatives in strategies.items():
        # Ensure alternatives in strategy exist in the dataset
        valid_alternatives = [alt for alt in alternatives if alt in alternative_closeness]

        # Calculate average closeness index across valid alternatives for each simulation
        for i in range(n):
            # Get the i-th closeness index for all alternatives in the strategy
            closeness_values = [alternative_closeness[alt][i] for alt in valid_alternatives]
            # Calculate the average and store it
            strategy_closeness[strategy_name].append(sum(closeness_values) / len(closeness_values))

    return strategy_closeness

def topsis_MC_Int_Strategies(*expert_data, ref_values, l, actions_to_analyze=None, criteria_to_analyze=None):
    """
    Runs a Monte Carlo TOPSIS simulation focusing on probability distributions of the closeness index for specified actions.
    """
    actions_to_analyze = set(actions_to_analyze) if actions_to_analyze else set().union(*[data.keys() for data in expert_data])
    closeness_distributions = {action: [] for action in actions_to_analyze}
    avg_closeness_distributions = []
    
    for _ in range(l):
        simulated_results = aggregate_MC_Int(*expert_data, n=1)
        
        randomised_matrix = []
        selected_actions = sorted(actions_to_analyze)
        
        for action in selected_actions:
            row = [simulated_results[action][criterion][0] for criterion in (criteria_to_analyze if criteria_to_analyze else sorted(simulated_results[action].keys()))]
            randomised_matrix.append(row)
        
        randomised_matrix = np.array(randomised_matrix)
        weights = np.ones(len(criteria_to_analyze)) if criteria_to_analyze else np.ones(randomised_matrix.shape[1])
        
        c_i = calculate_TP_distance(randomised_matrix, weights=weights, ref_values=ref_values)
        avg_c_i = np.mean(c_i)
        avg_closeness_distributions.append(round(float(avg_c_i), 3))
        
    
    return avg_closeness_distributions