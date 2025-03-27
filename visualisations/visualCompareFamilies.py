import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objects import Figure

def showCompareFamilies(matrix, x_family, y_family):
    """
    Visualize a comparison between two criteria families for multiple strategies using a scatter plot.

    Parameters:
    - matrix: dict, output from topsis_MC_Strategies, where each alternative contains closeness index values.
    - x_family: str, the family to use for the x-axis (e.g., "c_i_env").
    - y_family: str, the family to use for the y-axis (e.g., "c_i_soc").
    """
    # Prepare the DataFrame
    data_list = []
    for strategy, criteria_scores in matrix.items():
        # Ensure x_family and y_family exist in the strategy's criteria_scores
        if x_family in criteria_scores and y_family in criteria_scores:
            # Iterate over the simulations for the given families
            for i in range(len(criteria_scores[x_family])):
                data_list.append({
                    "Strategy": strategy,
                    x_family: criteria_scores[x_family][i],
                    y_family: criteria_scores[y_family][i],
                })

    # Create a DataFrame
    df = pd.DataFrame(data_list)

    # Create the scatter plot
    fig = px.scatter(
        df,
        x=x_family,
        y=y_family,
        color="Strategy",  # Differentiate points by strategy
        title=f"Comparison Between {x_family} and {y_family} Across Strategies",
        labels={x_family: x_family, y_family: y_family},
        hover_data=["Strategy"],  # Add additional hover information
        color_discrete_sequence=px.colors.qualitative.Set2,  # Custom color scheme
    )
    fig.update_traces(marker_size=15)

    fig.add_shape(
        type="line",
        x0=0.5, x1=0.5, 
        y0=0, y1=1,  # vertical line at y=0.5
        line=dict(color="black", width=3, dash="dash"),  # Black dashed line
    )

    fig.add_shape(
        type="line",
        x0=0, x1=1, 
        y0=0.5, y1=0.5,  # horizontal line at y=0.5
        line=dict(color="black", width=3, dash="dash"),  # Black dashed line
    )

    # Customize layout
    fig.update_layout(
        xaxis=dict(
            title=x_family,
            range=[0, 1]  # Fix x-axis range between 0 and 1
        ),
        yaxis=dict(
            title=y_family,
            range=[0, 1]  # Fix y-axis range between 0 and 1
        ),
        showlegend=True  # Display legend for strategies
    )

    fig.show()


def showCompareActions(env_input, soc_input):
    """
    Visualize a comparison between two criteria families for multiple strategies using a scatter plot.

    Parameters:
    - env_input: dict, contains environmental criteria values for each strategy (key: strategy, value: list of scores).
    - soc_input: dict, contains social criteria values for each strategy (key: strategy, value: list of scores).
    """
    # Prepare the combined DataFrame
    data_list = []

    for alternative in env_input.keys():
        if alternative in soc_input:  # Ensure the alternative exists in both inputs
            for i in range(len(env_input[alternative])):
                data_list.append({
                    "alternative": alternative,
                    "Environmental": env_input[alternative][i],
                    "Social": soc_input[alternative][i]
                })

    # Create the DataFrame
    df = pd.DataFrame(data_list)

    # Create the scatter plot
    fig = px.scatter(
        df,
        x="Environmental",
        y="Social",
        color="alternative",
        title="Comparison Between Social and Operational Criteria",
        labels={"Environmental": "Social Criteria", "Social": "Operational Criteria"}, #rename here the x and y axis if needed
        hover_data=["alternative"],
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_traces(marker=dict(size=10))
       
    fig.add_shape(
        type="line",
        x0=0.5, x1=0.5, 
        y0=0, y1=1,  # vertical line at y=0.5
        line=dict(color="black", width=3, dash="dash"),  # Black dashed line
    )

    fig.add_shape(
        type="line",
        x0=0, x1=1, 
        y0=0.5, y1=0.5,  # horizontal line at y=0.5
        line=dict(color="black", width=3, dash="dash"),  # Black dashed line

    )

    fig.show()
