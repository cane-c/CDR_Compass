import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

#for direct ratings
def showExpertMatrix(matrix):
    """
    shows the uncertainty in the expert matrix (min, med and max values)
    """
    # Convert aggregated_data into a DataFrame for easy plotting
    data_list = []
    
    for alternative, criteria in matrix.items():
        for criterion, stats in criteria.items():
            # Append each value with alternative and criterion as additional info
            data_list.append({'Alternative': alternative, 'Criterion': criterion, 'Value': stats['min'], 'Type': 'Min'})
            data_list.append({'Alternative': alternative, 'Criterion': criterion, 'Value': stats['max'], 'Type': 'Max'})
            data_list.append({'Alternative': alternative, 'Criterion': criterion, 'Value': stats['mean'], 'Type': 'Mean'})

    # Create a DataFrame
    df = pd.DataFrame(data_list)

    fig = px.box(df, y = "Value", facet_col="Criterion", color="Alternative", color_discrete_sequence=px.colors.qualitative.Safe)

    fig.show()

#for interval ratings

import pandas as pd
import plotly.express as px

def showExpertMatrix_Int(expertMatrix_Int, selected_criteria=None):
    """
    Generates a Plotly box plot from the aggregated expert matrix data.
    
    - Each sub-category (criterion) will have its own facet.
    - Different categories (alternatives) will be color-coded.
    - Allows filtering by a list of selected sub-categories.

    Args:
        expertMatrix_Int (dict): Aggregated data from aggregateEvaluations_Int function.
        selected_criteria (list, optional): List of sub-categories (criteria) to display. Defaults to all.
    """
    # Convert aggregated data into a structured list
    data_list = []
    
    for alternative, criteria in expertMatrix_Int.items():
        for criterion, stats in criteria.items():
            # If filtering is enabled and this criterion is not in the list, skip it
            if selected_criteria and criterion not in selected_criteria:
                continue

            # Add values for lower fence, q1, q3, and upper fence
            data_list.append({'Alternative': alternative, 'Criterion': criterion, 'Value': stats['lower_fence'], 'Type': 'Lower Fence'})
            data_list.append({'Alternative': alternative, 'Criterion': criterion, 'Value': stats['q1'], 'Type': 'Q1'})
            data_list.append({'Alternative': alternative, 'Criterion': criterion, 'Value': (stats['q1'] + stats['q3']) / 2, 'Type': 'Median'})  # Approximate median
            data_list.append({'Alternative': alternative, 'Criterion': criterion, 'Value': stats['q3'], 'Type': 'Q3'})
            data_list.append({'Alternative': alternative, 'Criterion': criterion, 'Value': stats['upper_fence'], 'Type': 'Upper Fence'})

    # Convert list into a DataFrame
    df = pd.DataFrame(data_list)

    # If there's nothing to plot (empty selection), show a message
    if df.empty:
        print("No matching sub-categories found. Check your selected_criteria list.")
        return

    # Create a box plot with facet_col for sub-categories
    fig = px.box(
        df,
        y="Value",
        facet_col="Criterion",  # Each sub-category gets its own facet
        color="Alternative",  # Different categories get different colors
        color_discrete_sequence=px.colors.qualitative.Safe
    )

    # Improve layout for readability
    fig.update_layout(
        title="Expert Matrix Box Plot",
        xaxis_title="Alternatives",
        yaxis_title="Values",
        boxmode="group",
        margin=dict(l=50, r=50, t=50, b=50),
        height=600,
        width=1200
    )

    fig.show()

def visualize_MC_Int(simulated_output, alternative):
    """
    Visualizes the density and distribution of simulation results with MC for a chosen alternative using violin plots.
    
    Parameters:
        simulated_output (dict): The output of the Monte Carlo simulation. For e.g.:     'A1': {
        'EN1': [6, 5, 7, 6, 4],
        'EN2': [4, 3, 4, 4, 5]
    },
        alternative (str): The specific alternative to visualize.
    """
    if alternative not in simulated_output:
        raise ValueError(f"Alternative '{alternative}' not found in simulation results.")
    
    # Prepare the DataFrame
    data_list = []
    for criterion, scores in simulated_output[alternative].items():
        for score in scores:
            data_list.append({"Criterion": criterion, "Score": score})
    
    df = pd.DataFrame(data_list)
    
    # Create the violin plot
    fig = px.violin(
        df,
        y="Score",
        x="Criterion",
        box=True,  # Show a boxplot inside the violin plot
        #points="all",  # Show all individual points
        title=f"Density and Distribution of Simulation Results for {alternative}",
        labels={"Criterion": "Criteria", "Score": "Simulated Scores"},
        color="Criterion",  # Color each criterion differently
        color_discrete_sequence=px.colors.qualitative.Safe,  # Predefined color scheme
    )
    
    # Customize layout
    fig.update_layout(
        yaxis_title="Simulated Scores",
        xaxis_title="Criteria",
        showlegend=False  # Hide legend for better clarity
    )
    
    fig.show()