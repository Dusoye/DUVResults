import pandas as pd
import glob
import os

# Path to your CSV files
path = './output/'

# Get all CSV files in the directory
all_files = glob.glob(os.path.join(path, "*.csv"))

# Initialize an empty list to store DataFrames
dfs = []

# Read each CSV file
for filename in all_files:
    df = pd.read_csv(filename)
    dfs.append(df)

# Concatenate all DataFrames
combined_df = pd.concat(dfs, ignore_index=True)

# Group by Event ID and Runner ID, then count occurrences
duplicates = combined_df.groupby(['Event ID', 'Runner ID']).size().reset_index(name='count')

# Filter for cases where a runner appears more than once in an event
duplicates = duplicates[duplicates['count'] > 1]

if duplicates.empty:
    print("No duplicates found where the same runner ID appears multiple times within the same event ID.")
else:
    print("Duplicates found:")
    print(duplicates)

    # Optional: Get more details about these duplicates
    for _, row in duplicates.iterrows():
        event_id = row['Event ID']
        runner_id = row['Runner ID']
        occurrences = combined_df[(combined_df['Event ID'] == event_id) & (combined_df['Runner ID'] == runner_id)]
        print(f"\nEvent ID: {event_id}, Runner ID: {runner_id}")
        print(occurrences[['Event', 'Date', 'Performance', 'Rank', 'M/F']])
