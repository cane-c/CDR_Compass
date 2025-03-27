def combine_strategies(*strategies):
    """
    Combines multiple strategies into one dictionary with named keys (e.g., S1, S2).
    
    Parameters:
    - *strategies: Variable number of strategies, each a dictionary.
    
    Returns:
    - dict: Combined dictionary with keys S1, S2, etc.
    """
    combined = {}
    for i, strategy in enumerate(strategies, start=1):
        strategy_name = f"S{i}"  # Create a key like S1, S2, etc.
        combined[strategy_name] = strategy  # Assign the strategy to this key
    return combined

def combine_Ext_Strategies(*strategies):
    """
    Combine multiple strategies into a single dictionary structure.

    Parameters:
    - *strategies: Variable number of strategies, each having the structure:
      {'min': value, 'max': value, 'mean': value}.

    Returns:
    - A combined dictionary with keys 'min', 'max', and 'mean',
      and sub-keys corresponding to the strategy names (S1, S2, ...).
    """
    combined_results = {'min': {}, 'max': {}, 'mean': {}}

    # Iterate through the strategies and assign them to S1, S2, ...
    for i, strategy in enumerate(strategies, start=1):
        strategy_name = f"S{i}"
        for key in ['min', 'max', 'mean']:
            if key in strategy:
                combined_results[key][strategy_name] = strategy[key]

    return combined_results
