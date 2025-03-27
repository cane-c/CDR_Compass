import pandas as pd
import plotly.express as px

def showDeterministicTopsis(matrix):
    """
    compares alternatives based on the closeness index, has three scenarii: best, worst and medium
    """
    # Convert aggregated_data into a DataFrame for easy plotting
    data_list = []
    
    for scenario, alternatives in matrix.items():
        for alternative, score in alternatives.items():
            # Append each value with alternative and criterion as additional info
            data_list.append({'Scenario': scenario, 'Alternative': alternative, 'Value': score})


    # Create a DataFrame
    df = pd.DataFrame(data_list)

    #fig = px.box(df, y = "Value", facet_col="Scenario", color="Alternative", color_discrete_sequence=px.colors.qualitative.Safe,  title="TOPSIS Results for Deterministic Scenarios",)

    # Update layout for better readability
    #fig.update_layout(
    #yaxis_title="TOPSIS Scores",
    #xaxis_title="Scenario",
    #boxmode="group"
    #)

    # Create the scatter plot with facets for each scenario
    fig = px.scatter(
        df,
        y="Value",
        x="Alternative",  # Set Alternatives as the X-axis
        facet_col="Scenario",
        color="Alternative",
        text = "Alternative",
        size=[10] * len(df),  # Make marker sizes uniform
        color_discrete_sequence=px.colors.qualitative.Safe,
        title="TOPSIS Results for Deterministic Scenarios",
    )

    # Customize layout
    fig.update_traces(
        marker=dict(symbol='circle', opacity=0.8),  # Use circle markers
        textposition="top center"  # Position text above markers
    )

    fig.update_layout(
        yaxis_title="TOPSIS Scores",
        xaxis_type='category',  # Ensure categorical spacing for scenarios
        showlegend=True
    )

    fig.show()

def showDeterministicTopsisStrategies(*strategies):
    """
    Visualizes the TOPSIS results for multiple strategies.
    Parameters:
    - *strategies: Variable number of strategies, each a list containing [min, max, mean
    """
    # Prepare the DataFrame
    data_list = []
    for i, strategy_scores in enumerate(strategies, start=1):
        strategy_name = f"Strategy{i}"
        for scenario, score in strategy_scores.items():
            data_list.append({"Strategy": strategy_name, "Scenario": scenario, "Score": score})

    print(data_list)
    df = pd.DataFrame(data_list)

    # Create the scatter plot
    fig = px.scatter(
        df,
        x="Strategy",
        y="Score",
        facet_col="Scenario",
        color="Strategy",
        text="Score",  # Add scores as text
        size=[10] * len(df),  # Uniform marker size
        color_discrete_sequence=px.colors.qualitative.Safe,
        title="TOPSIS Results for Deterministic Strategies"
)

    # Customize layout
    fig.update_traces(
        marker=dict(symbol='circle', opacity=0.8),  # Use circle markers
        textposition="top center"  # Position text above markers
)

    fig.update_layout(
        yaxis_title="TOPSIS Scores",
        xaxis_type='category',  # Ensure categorical spacing for scenarios
        showlegend=True,  # Show legend for scenarios
        xaxis=dict(tickangle=45)  # Rotate strategy names for better visibility
    )

    fig.show()