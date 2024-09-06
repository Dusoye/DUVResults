import os
import pandas as pd
from collections import OrderedDict

'''Verify the number of events scraped to check against DUV site'''
def count_unique_events(folder_path):
    # Dictionary to store results
    results = OrderedDict()

    # Iterate through all files in the folder
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            
            try:
                # Read the CSV file with low_memory=False to avoid DtypeWarning
                df = pd.read_csv(file_path, low_memory=False)
                
                # Count unique event_id
                if 'Event ID' in df.columns:
                    # Convert 'Event ID' column to string type to ensure consistent comparisons
                    df['Event ID'] = df['Event ID'].astype(str)
                    unique_count = df['Event ID'].nunique()
                    results[filename] = unique_count
                else:
                    results[filename] = "No 'Event ID' column found"
            except Exception as e:
                results[filename] = f"Error reading file: {str(e)}"

    return results

def main():
    folder_path = './output'
    
    print("Counting unique event IDs in CSV files...")
    results = count_unique_events(folder_path)
    
    print("\nResults:")
    print("-" * 50)
    for filename, count in results.items():
        print(f"{filename}: {count}")
    print("-" * 50)
    
    total_unique = sum(count for count in results.values() if isinstance(count, int))
    print(f"Total unique event IDs across all files: {total_unique}")

if __name__ == "__main__":
    main()