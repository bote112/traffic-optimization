import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
# Define the paths to the CSV files created by analyze_results.py
BASELINE_DATA_FILE = "baseline_data.csv"
AGENT_DATA_FILE = "agent_data.csv"

# Set a consistent style for the plots
sns.set_theme(style="whitegrid")

# --- Main Plotting Function ---
def create_comparison_plots(baseline_df, agent_df):
    """
    Generates and displays side-by-side histograms comparing the performance
    of the baseline and the RL agent.

    Args:
        baseline_df (pd.DataFrame): The processed data for the baseline.
        agent_df (pd.DataFrame): The processed data for the agent.
    """
    # Create a figure with two subplots, sharing the Y-axis for easier comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle('Performance Comparison: Baseline vs. Trained RL Agent', fontsize=20, y=1.02)

    # --- Plot 1: Trip Duration Distribution ---
    ax1 = axes[0]
    sns.histplot(baseline_df['duration'], ax=ax1, color='skyblue', kde=True, label='Baseline', binwidth=50)
    sns.histplot(agent_df['duration'], ax=ax1, color='lightcoral', kde=True, label='RL Agent', binwidth=50)
    ax1.set_title('Distribution of Total Trip Durations', fontsize=16)
    ax1.set_xlabel('Trip Duration (seconds)', fontsize=12)
    ax1.set_ylabel('Number of Vehicles', fontsize=12)
    ax1.legend()
    # Add vertical lines for the median (50th percentile)
    ax1.axvline(baseline_df['duration'].median(), color='royalblue', linestyle='--', label=f"Baseline Median: {baseline_df['duration'].median():.1f}s")
    ax1.axvline(agent_df['duration'].median(), color='darkred', linestyle='--', label=f"Agent Median: {agent_df['duration'].median():.1f}s")
    ax1.legend() # Call legend again to include the new labels

    # --- Plot 2: Waiting Time Distribution ---
    ax2 = axes[1]
    sns.histplot(baseline_df['waitingTime'], ax=ax2, color='skyblue', kde=True, label='Baseline', binwidth=20)
    sns.histplot(agent_df['waitingTime'], ax=ax2, color='lightcoral', kde=True, label='RL Agent', binwidth=20)
    ax2.set_title('Distribution of Total Waiting Times', fontsize=16)
    ax2.set_xlabel('Waiting Time (seconds)', fontsize=12)
    ax2.set_ylabel('') # Don't repeat the Y-axis label
    ax2.legend()
    # Add vertical lines for the median (50th percentile)
    ax2.axvline(baseline_df['waitingTime'].median(), color='royalblue', linestyle='--', label=f"Baseline Median: {baseline_df['waitingTime'].median():.1f}s")
    ax2.axvline(agent_df['waitingTime'].median(), color='darkred', linestyle='--', label=f"Agent Median: {agent_df['waitingTime'].median():.1f}s")
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
    plt.show()

if __name__ == "__main__":
    try:
        # Load the clean data from the CSV files
        baseline_data = pd.read_csv(BASELINE_DATA_FILE)
        agent_data = pd.read_csv(AGENT_DATA_FILE)

        # Generate the plots
        create_comparison_plots(baseline_data, agent_data)

    except FileNotFoundError as e:
        print(f"Error: Could not find data file. Please run analyze_results.py first.")
        print(f"Details: {e}")
