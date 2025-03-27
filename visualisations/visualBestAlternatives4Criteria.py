import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_best_alternatives(dataset, c_i_results):
    """
    Visualize the best alternatives for each criterion based on maximum value,
    along with their closeness index (c_i).

    Parameters:
    - dataset: dict, structured as alternatives with criteria and their values.
    - c_i_results: dict, overall closeness index (c_i) values for each alternative.
    """
    # Convert dataset and c_i_results into a DataFrame
    data = []
    for alt, criteria in dataset.items():
        for criterion, value in criteria.items():
            data.append({"Alternative": alt, "Criterion": criterion, "Value": value['max'], "C_i": c_i_results[alt]})

    df = pd.DataFrame(data)

    # Create dropdown options for each criterion
    criteria_options = df['Criterion'].unique()

    # Generate a bar chart for the first criterion as default
    def filter_criterion(criterion):
        # Filter for alternatives with max value for the selected criterion
        max_value = df[df['Criterion'] == criterion]['Value'].max()
        filtered_df = df[(df['Criterion'] == criterion) & (df['Value'] == max_value)]
        return filtered_df

    # Initial data for the first criterion
    default_criterion = criteria_options[0]
    filtered_df = filter_criterion(default_criterion)

    # Create a bar chart
    fig = go.Figure()

    # Add bars for the initial criterion
    fig.add_trace(go.Bar(
        x=filtered_df['Alternative'],
        y=filtered_df['C_i'],
        text=filtered_df['C_i'].round(2),
        textposition='outside',
        marker=dict(color='rgb(99, 110, 250)'),
        name=f"Best Alternatives for {default_criterion}"
    ))

    # Update layout for dropdown interactivity
    fig.update_layout(
        title="Best Alternatives by Criterion",
        xaxis_title="Alternatives",
        yaxis_title="Closeness Index (C_i)",
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        label=criterion,
                        method="update",
                        args=[
                            {"x": [filter_criterion(criterion)['Alternative']],
                             "y": [filter_criterion(criterion)['C_i']],
                             "text": [filter_criterion(criterion)['C_i'].round(2)]},
                            {"title": f"Best Alternatives for {criterion}"}
                        ]
                    )
                    for criterion in criteria_options
                ],
                direction="down",
                showactive=True,
                x=0.5,
                y=1.15
            )
        ]
    )

    # Show the plot
    fig.show()

def scatter_best_alternatives(dataset, x_family, y_family):
    """
    Visualize the best alternatives for each criterion based on maximum value,
    along with their closeness index on different families (c_i).

    Parameters:
    - dataset: dict, structured as alternatives with criteria and their values.
    - x_family: c_i results dict, the family to use for the x-axis (e.g., "c_i_env").
    - y_family: c_i results dict, the family to use for the y-axis (e.g., "c_i_soc").
    """
    # Convert dataset and c_i results into a DataFrame. 
    # Attention to value 'max', 'min' or 'mean' - it depends on the aggregation and which performance values you want to extract
    data = []
    for alt, criteria in dataset.items():
        for criterion, value in criteria.items():
            data.append({"Alternative": alt, 
                         "Criterion": criterion, 
                         "Value": value['max'],  # Use 'max', 'min', or other key depending on the requirement
                         "C_i_environment": x_family[alt], 
                         "C_i_social": y_family[alt] })

    df = pd.DataFrame(data)

    # Create dropdown options for each criterion
    criteria_options = df['Criterion'].unique()

    # Generate a scatter plot for the first criterion as default

    def filter_criterion(criterion):
        # Filter for alternatives with max value for the selected criterion
        criterion_df = df[df['Criterion'] == criterion]
        max_value = criterion_df['Value'].max()
        return criterion_df[criterion_df['Value'] == max_value]

    # Initial data for the first criterion
    default_criterion = criteria_options[0]
    filtered_df = filter_criterion(default_criterion)

    # Create the initial scatter plot
    fig = go.Figure()

    # Add initial scatter points for the default criterion
    # Add scatter points for all alternatives with initial filtered data
    for alternative in filtered_df['Alternative'].unique():
        alt_data = filtered_df[filtered_df['Alternative'] == alternative]
        fig.add_trace(
            go.Scatter(
                x=alt_data["C_i_environment"],
                y=alt_data["C_i_social"],
                mode="markers+text",
                marker=dict(size=20, line=dict(width=1), symbol='circle'),
                name=alternative,
                text=alt_data["Alternative"],
                textposition="top center",
                textfont=dict(size=12),
            )
        )

    #fig.update_traces(marker_size=15)

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

   # Dropdown menu for criteria
    buttons = []
    for criterion in criteria_options:
        filtered_df = filter_criterion(criterion)
        # Ensure unique colors for alternatives
        unique_alternatives = filtered_df["Alternative"].unique()
        color_map = {alt: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
             for i, alt in enumerate(unique_alternatives)}

        # Map the colors to the 'Alternative' column
        filtered_df["color"] = filtered_df["Alternative"].map(color_map)
        
        # Add button to update the scatter plot based on the criterion
        buttons.append(
            dict(
                label=criterion,
                method="update",
                args=[
                    {
                        "x": [filtered_df["C_i_environment"].values],
                        "y": [filtered_df["C_i_social"].values],
                        "text": [filtered_df["Alternative"].values],
                        "marker.color": [filtered_df["color"]],
                        "marker.size": [20] * len(filtered_df),
                    },
                    {"title": f"Best Alternatives for {criterion}"}
                ]
            )
        )

    # Update layout with dropdown menu
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.5,
                y=1.15
            )
        ],
        title=f"Best Alternatives for {default_criterion}",
        xaxis=dict(title="Environmental Closeness Index", range=[0, 1]),
        yaxis=dict(title="Social Closeness Index", range=[0, 1]),
    )


    # Show the plot
    fig.show()



