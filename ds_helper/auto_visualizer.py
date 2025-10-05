import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from column_detector import detect_column_types
import numpy as np

def auto_visualize(df):

    # Get the column types from the helper function
    print("Analyzing data and selecting appropriate plots...")
    print("-" * 40)

    # Get the column types from the helper function
    column_types = detect_column_types(df)
    
    # Calculate the number of plots needed
    num_individual_plots = len(column_types)
    numerical_cols = [col for col, col_type in column_types.items() if col_type == 'numerical']
    
    # Add one extra plot for the correlation heatmap if numerical columns exist
    num_total_plots = num_individual_plots + (1 if numerical_cols else 0)

    # Create a figure and axes object with the correct number of subplots
    fig, axes = plt.subplots(ncols=num_total_plots, figsize=(num_total_plots * 5, 5))
    ax_list = iter(axes)

    # Iterate through the columns and call the correct helper function
    for col_name, col_type in column_types.items():
        if col_name in df.columns:
            current_ax = next(ax_list)
            if col_type == 'numerical':
                _plot_numerical(df[col_name], ax=current_ax)
            elif col_type == 'categorical':
                _plot_categorical(df[col_name], ax=current_ax)
            elif col_type == 'text':
                _plot_text(df[col_name], ax=current_ax)
    
    # Finally, plot the correlation heatmap if applicable
    if numerical_cols:
        plot_correlation_heatmap(df, numerical_cols, ax=next(ax_list))
    
    # Adjust layout to prevent labels from overlapping
    plt.tight_layout()

    print("-" * 40)
    print("All plots generated successfully.")
    
    return fig

# Helper Functions with an underscore to denote they are for internal use
def _plot_numerical(data, ax):
    """Handles numerical columns (histograms, boxplots, etc.)."""
    ax.hist(data)
    ax.set_title(f"Histogram for {data.name}")
    print(f"-> Plotting numerical data for {data.name}.")

def _plot_categorical(data, ax):
    """Handles categorical columns (bar charts, count plots)."""
    # Use value_counts() for pandas Series to get counts of unique values
    value_counts = data.value_counts()
    ax.bar(value_counts.index, value_counts.values)
    ax.set_title(f"Categorical Data Plot for {data.name}")
    print(f"-> Plotting categorical data for {data.name}.")

def _plot_text(data, ax):
    """Handles text columns (word clouds, frequency plots)."""
    # Placeholder for a more complex plot, e.g., word cloud or frequency plot
    ax.set_title(f"Text Data Plot for {data.name} (Placeholder)")
    ax.set_xlabel("Placeholder")
    ax.set_ylabel("Placeholder")
    print(f"-> Plotting text data for {data.name}.")

# Helper function without an underscore, as it might be a public utility
def plot_correlation_heatmap(df, numerical_cols, ax):
    """Makes a heatmap for correlations."""
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap")
    print("-> Plotting correlation heatmap for numerical columns.")

df=pd.read_csv(r"C:\Users\Osheen kumar\Downloads\Titanic-Dataset.csv")

auto_visualize(df)