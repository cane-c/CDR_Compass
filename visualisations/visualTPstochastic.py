import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

def showScatterMC(*strategies):
    """
    Visualize the density and distribution of simulation results for multiple strategies using violin plots.

    Parameters:
    - *strategies: Variable number of strategies, each a list of scores representing simulation results.
    """
    # Prepare the DataFrame
    data_list = []
    for i, strategy_scores in enumerate(strategies, start=1):
        strategy_name = f"Strategy{i}"
        for score in strategy_scores:
            data_list.append({"Strategy": strategy_name, "Score": score})

    df = pd.DataFrame(data_list)

    # Create the violin plot
    fig = px.violin(
        df,
        y="Score",  # Scores on the y-axis
        facet_col="Strategy",
        box=True,  # Show a boxplot inside the violin plot
        points="all",  # Show all individual points
        title="Density and Distribution of Simulation Results for Multiple Strategies",
        labels={"Strategy": "Strategies", "Score": "Closeness Scores"},
        color="Strategy",  # Color each strategy differently
        color_discrete_sequence=px.colors.qualitative.Safe,  # Predefined color scheme
    )

    # Customize layout
    fig.update_layout(
        yaxis_title="Closeness Scores",
        xaxis_title="Strategies",
        showlegend=True  # Show legend for scenarios
    )

    fig.show()

def showScatterMC_dictionary(strategy_closeness_index):
    """
    Visualize the density and distribution of simulation results for multiple strategies using violin plots.

    Parameters:
    - strategy_closeness_index: dict, where keys are strategy names (e.g., "S1", "S2")
      and values are lists of scores representing simulation results.
    """

    #Round data to have 4 decimal values max
    strategy_closeness_index = {
        strategy: [round(score, 4) for score in scores]
        for strategy, scores in strategy_closeness_index.items()
    }

    # Prepare the DataFrame
    data_list = []
    for strategy_name, scores in strategy_closeness_index.items():
        for score in scores:
            data_list.append({"Alternative": strategy_name, "Score": score})

    df = pd.DataFrame(data_list)

    fig = px.violin(
        df,
        x="Alternative",  # Set the category on the x-axis
        y="Score",
        box=True,
        points=False, #set to "all" if i want to have points next to violin, or "none" if i want to make them disappear
        title="Density and Distribution of Simulation Results for Multiple Alternatives",
        labels={"Alternative": "Alternatives", "Score": "Closeness Scores"},
        color="Alternative",
        color_discrete_sequence=px.colors.qualitative.Safe,
)


    # Customize layout
    fig.update_layout(
        yaxis_title="Closeness Scores",
        xaxis_title="Alternatives",
        showlegend=True  # Show legend for scenarios
    )

    fig.show()

def plot_3d_scatter(x_values, y_values, z_values):
    """
    Plots a 3D scatter plot using Plotly with the given x, y, and z values.
    Used to plot a strategy and demonstrate its impacts on three criteria families

    Parameters:
    x_values (list): List of numeric values for the x-axis. For e.g. only environmental closeness indexes
    y_values (list): List of numeric values for the y-axis. For e.g. only social cl
    z_values (list): List of numeric values for the z-axis. For e.g. only operation cl
    """

    # Validate input lengths
    if not (len(x_values) == len(y_values) == len(z_values)):
        raise ValueError("All input lists must have the same length.")

    # Create a 3D scatter plot
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x_values, 
        y=y_values, 
        z=z_values,
        mode='markers',  # Only markers, no lines
        marker=dict(size=5, color="blue"),
        name="Data Points"
    ))

    # Ensure (0.5, 0.5, 0.5) is highlighted (if present or not)
    fig.add_trace(go.Scatter3d(
        x=[0.5], 
        y=[0.5], 
        z=[0.5], 
        mode='markers',
        marker=dict(
            size=10,  # Larger size to make it visible
            color='red',  # Different color to highlight
            opacity=1.0,
            symbol='diamond'
        ),
        name="Point (0.5, 0.5, 0.5)" #neutral CL
    ))

    # Layout settings
    fig.update_layout(
        title="Impacts of a CDR Strategy on three criteria families",
        scene=dict(
            xaxis_title="Environmental Criteria",
            yaxis_title="Social Criteria",
            zaxis_title="Operational Criteria"
        )
    )

    # Show the plot
    fig.show()

