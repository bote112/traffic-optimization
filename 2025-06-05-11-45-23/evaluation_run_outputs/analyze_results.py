import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
# Define the paths to the folders containing your evaluation output files.
BASELINE_OUTPUT_DIR = "baseline_outputs"
AGENT_OUTPUT_DIR = "agent_outputs"

# --- Main Functions ---

def process_folder(folder_path):
    """
    Reads all tripinfo XML files in a given folder, extracts the data,
    and returns a single Pandas DataFrame.

    Args:
        folder_path (str): The path to the folder containing XML files.

    Returns:
        pd.DataFrame: A DataFrame with 'duration' and 'waitingTime' for all trips,
                      or None if the folder doesn't exist or is empty.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Directory not found at '{folder_path}'")
        return None

    all_trip_durations = []
    all_trip_waiting_times = []

    # Loop through every file in the specified directory
    for filename in os.listdir(folder_path):
        if filename.endswith(".xml"):
            file_path = os.path.join(folder_path, filename)
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                # Extract data from each tripinfo tag in the file
                for trip in root.findall('tripinfo'):
                    all_trip_durations.append(float(trip.get('duration')))
                    all_trip_waiting_times.append(float(trip.get('waitingTime')))

            except (ET.ParseError, FileNotFoundError) as e:
                print(f"Warning: Could not parse {filename}. Reason: {e}")

    if not all_trip_durations:
        return None
        
    # Create a Pandas DataFrame from the aggregated data
    df = pd.DataFrame({
        'duration': all_trip_durations,
        'waitingTime': all_trip_waiting_times
    })
    return df


def print_summary(dataframe, model_name):
    """
    Takes a DataFrame and prints a formatted statistical summary.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing trip data.
        model_name (str): The name of the model for the report header.
    """
    if dataframe is None or dataframe.empty:
        print(f"\nNo data available to generate summary for {model_name}.")
        return

    print(f"\n--- Performance Summary for: {model_name} ---")
    print(f"Data collected from {len(dataframe)} total vehicle trips.")

    print("\n--- Trip Duration Stats (seconds) ---")
    print(dataframe['duration'].describe(percentiles=[.5, .75, .95, .99]))

    print("\n--- Waiting Time Stats (seconds) ---")
    print(dataframe['waitingTime'].describe(percentiles=[.5, .75, .95, .99]))
    print("--------------------------------------------------")


if __name__ == "__main__":
    # Process both the baseline and agent output folders
    print("Starting analysis of evaluation results...")
    
    baseline_df = process_folder(BASELINE_OUTPUT_DIR)
    agent_df = process_folder(AGENT_OUTPUT_DIR)

    # Print the detailed summaries to the console
    print_summary(baseline_df, "Baseline Controller")
    print_summary(agent_df, "Trained RL Agent")

    # Save the aggregated data to CSV files for easy use in other scripts
    if baseline_df is not None:
        baseline_df.to_csv("baseline_data.csv", index=False)
        print("\nBaseline data saved to baseline_data.csv")
    
    if agent_df is not None:
        agent_df.to_csv("agent_data.csv", index=False)
        print("Agent data saved to agent_data.csv")