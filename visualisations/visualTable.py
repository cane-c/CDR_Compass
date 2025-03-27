import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.colors import n_colors
import numpy as np
np.random.seed(1)

def draw_table(results):
    """
    For Stochastic data where alternatives are also ranked
    Returns two tables, one comparing the ranking the other comparing the topsis closeness index
    """
    
    # Extract alternatives and data
    alternatives = list(results.keys())
    rank_columns = sorted(
        [col for col in results[alternatives[0]].keys() if col.startswith('rank')],
        key=lambda x: int(x[4:])  # Extract the number part of "rankX" and sort numerically
    )
    avg_closeness_col = "average_c"

    # Extract rank percentages and average closeness values
    rank_data = [[results[alt][rank] for alt in alternatives] for rank in rank_columns]

    # Combine all data into a table structure
    table_data = [alternatives] + rank_data

    # Define column headers
    headers = ["<b>Alternative</b>"] + [f"<b>{rank}</b>" for rank in rank_columns] 

    # Extract data for the bar chart
    avg_closeness = [results[alt]["average_c"] for alt in alternatives]
    
    colors = n_colors('rgb(255, 200, 200)', 'rgb(200, 0, 0)', 101, colortype='rgb')

    # Apply gradient coloring for each column
    def get_gradient_colors(column):
        return [colors[int(value)] for value in column]

    fill_colors = [
        ['white'] * len(alternatives)  # White for the alternatives column
    ] + [get_gradient_colors(col) for col in rank_data]

   # Create the table
    table = go.Table(
        header=dict(
            values=headers,
            line_color='white',
            fill_color='white',
            align='center',
            font=dict(color='black', size=12)
        ),
        cells=dict(
            values=table_data,
            line_color=fill_colors,
            fill_color=fill_colors,
            align='center',
            font=dict(color='black', size=11)
        )
    )

    # Create the scatter plot for average closeness
    scatter_plot = go.Scatter(
        x=alternatives,
        y=avg_closeness,
        mode='markers+text',  # Add text labels to markers and lines
        marker=dict(size=10, color='rgb(99, 110, 250)'),
        name='Average Closeness',
        text=[f"{value:.2f}" for value in avg_closeness],  # Labels showing average closeness values
        textposition='top center',  # Position labels above the points
        textfont=dict(size=12, color='black')  # Customize label font
)

    # Combine into a single figure
    fig = go.Figure()

    # Add the table
    fig.add_trace(table)

    # Add the bar chart
    fig.add_trace(scatter_plot)

    # Adjust layout to position the table and chart
    fig.update_layout(
        yaxis=dict(domain=[0, 0.45]),  # Table
        yaxis2=dict(domain=[0.6, 1], anchor='x2'),  # Bar chart
        xaxis2=dict(anchor='y2'),  # Anchor bar chart axes
        margin=dict(t=50, l=50, b=50),
        title="TOPSIS Results: Ranking and Average Closeness based on stochasticity",
        height=800,
    )

    fig.show()

def show_average_ci_table(results):
    """
    Create a scatter plot for average closeness index (CI) values for each alternative.

    Parameters:
    - results: dict, containing 'average_ci', 'min_ci', and 'max_ci' for each alternative.
    """
    # Extract alternatives and average closeness values
    alternatives = list(results["average_ci"].keys())
    avg_closeness = [results["average_ci"][alt] for alt in alternatives]

    # Create the scatter plot
    scatter_plot = go.Figure(data=go.Scatter(
        x=alternatives,
        y=avg_closeness,
        mode='markers+text',  # Add text labels to markers and points
        marker=dict(size=10, color='rgb(99, 110, 250)'),
        name='Average Closeness',
        text=[f"{value:.2f}" for value in avg_closeness],  # Labels showing average closeness values
        textposition='top center',  # Position labels above the points
        textfont=dict(size=12, color='black')  # Customize label font
    ))

    # Update layout
    scatter_plot.update_layout(
        title="Average Closeness Index (CI) for Alternatives",
        xaxis_title="Alternatives",
        yaxis_title="Average CI",
        xaxis=dict(tickangle=45),  # Rotate alternative labels for better readability
        yaxis=dict(range=[0, 1]),  # Set the y-axis range (0 to 1 for CI)
        title_x=0.5,  # Center the title
        title_font=dict(size=16),
        showlegend=False  # No legend needed since we label directly
    )

    scatter_plot.show()