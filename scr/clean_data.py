import pandas as pd
import numpy as np
import glob
import os
import re
from datetime import datetime, timedelta
import argparse

def load_and_concat_csv(folder_path, chunksize=None):
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    df_list = []

    for filename in all_files:
        try:
            df_chunks = pd.read_csv(filename, chunksize=chunksize, 
                                    low_memory=False, encoding='utf-8')
            
            if chunksize:
                df = pd.concat(df_chunks, ignore_index=True)
            else:
                df = next(df_chunks)
            
            df['source_file'] = os.path.basename(filename)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading file {filename}: {str(e)}")

    combined_df = pd.concat(df_list, ignore_index=True, sort=False)
    return combined_df

def split_runner_name(data):
    # Initialize 'Surname' and 'First Name' columns with empty strings
    data['Surname'] = ''
    data['First Name'] = ''
    
    # Check 'Surname, first name' column first
    if 'Surname, first name' in data.columns:
        mask = data['Surname, first name'].notna()
        temp = data.loc[mask, 'Surname, first name'].str.split(',', expand=True)
        data.loc[mask, 'Surname'] = temp[0].str.strip()
        data.loc[mask, 'First Name'] = temp[1].str.strip() if temp.shape[1] > 1 else ''
    
    # Check 'Original name\nSurname, first name' column if 'Surname' is still empty
    original_name_column = 'Original name\nSurname, first name'
    if original_name_column in data.columns:
        mask = (data['Surname'] == '') & (data[original_name_column].notna())
        temp = data.loc[mask, original_name_column].str.split(',', expand=True)
        data.loc[mask, 'Surname'] = temp[0].str.strip()
        data.loc[mask, 'First Name'] = temp[1].str.strip() if temp.shape[1] > 1 else ''
    
    return data

def process_performance(df, input_col, time_col, distance_col):
    def convert_to_seconds(time_str):
        total_seconds = 0
        # Check for days
        day_match = re.search(r'(\d+)\s*d', time_str)
        if day_match:
            total_seconds += int(day_match.group(1)) * 24 * 3600
        
        # Extract HH:MM:SS part
        time_part = re.search(r'(\d+:)?\d+:\d+(\.\d+)?', time_str)
        if time_part:
            parts = time_part.group().split(':')
            if len(parts) == 3:
                hours, minutes, seconds = map(float, parts)
                total_seconds += int(hours * 3600 + minutes * 60 + seconds)
            elif len(parts) == 2:
                minutes, seconds = map(float, parts)
                total_seconds += int(minutes * 60 + seconds)
            else:
                total_seconds += int(float(parts[0]))
        
        return total_seconds

    def process_entry(entry, event_type):
        if pd.isna(entry) or pd.isna(event_type):
            return None, None

        entry = str(entry).strip()
        
        # Check if it's a time entry (including multi-day format)
        time_pattern = re.compile(r'((\d+)\s*d\s*)?(\d+:)?\d+:\d+(\.\d+)?\s*h?')
        if time_pattern.match(entry) or event_type in ['Distance']:
            # Remove any trailing 'h' and convert to seconds
            time_str = entry.rstrip(' h')
            return convert_to_seconds(time_str), None
        
        # If it's not a time, it's a distance
        # Update: Handle more flexible space between the number and the unit
        distance_pattern = re.compile(r'(\d+(\.\d+)?)\s*(km|mi)')
        match = distance_pattern.search(entry)  # Changed to search instead of match
        if match:
            distance = float(match.group(1))
            return None, distance
        
        # If we can't determine the type, return None for both
        return None, None

    df[time_col], df[distance_col] = zip(*df.apply(lambda row: process_entry(row[input_col], row['Event Type']), axis=1))
    
    return df


def split_distance_column(df):
    # Updated patterns to handle more unit variations
    distance_pattern = re.compile(r'(\d+(?:\.\d*)?)\s*\.?\s*(km|k|mi|m|mile|miles|h|d)', re.IGNORECASE)
    race_type_pattern = re.compile(r'(\d+(?:\.\d*)?)\s*\.?\s*(km|k|mi|m|mile|miles|h|d)\s*(.*)', re.IGNORECASE)
    
    def standardize_unit(unit):
        unit = unit.lower()
        if unit in ['k', 'km']:
            return 'km'
        elif unit in ['m', 'mi', 'mile', 'miles']:
            return 'mi'
        else:
            return unit  # 'h' and 'd' remain unchanged
    
    def split_distance(entry):
        if pd.isna(entry):
            return None, None, None
        
        # Convert to string if it's not already
        entry = str(entry)
        
        distance_match = distance_pattern.search(entry)
        race_type_match = race_type_pattern.search(entry)
        
        if distance_match:
            # Clean up the distance format
            distance_value = distance_match.group(1).rstrip('.')
            unit = standardize_unit(distance_match.group(2))
            distance = f"{distance_value}{unit}"
            
            # Update race_type logic
            if unit == 'h':
                race_type = 'Time'
            elif unit == 'd':
                race_type = 'Multi-day'
            else:
                race_type = 'Distance'
        else:
            distance = None
            race_type = None
        
        # Extract terrain
        if race_type_match:
            terrain_start = race_type_match.end(2)
            terrain = entry[terrain_start:].strip()
        else:
            terrain = None
        
        return distance, terrain, race_type
    
    df[['Distance/Time', 'Terrain', 'Event Type']] = df['Distance'].apply(split_distance).apply(pd.Series)

    return df

def convert_miles_to_km(entry):
    if pd.isna(entry):
        return None
    match = re.match(r'(\d+\.?\d*)(mi|km)', str(entry))
    if match:
        distance, unit = float(match.group(1)), match.group(2)
        return distance * 1.6 if unit == 'mi' else distance
    return None

def standardize_terrain(terrain):
    terrain = str(terrain).lower()
    if 'trail' in terrain:
        return 'trail'
    elif 'road' in terrain:
        return 'road'
    elif 'track' in terrain:
        return 'track'
    else:
        return 'other'

def parse_date_range(date_str):
    if pd.isna(date_str):
        return None
    
    date_str = str(date_str).strip()
    
    # Handle full date range format "DD.MM.YYYY-DD.MM.YYYY"
    match = re.match(r'(\d{1,2})\.(\d{1,2})\.(\d{4})-\d{1,2}\.\d{1,2}\.\d{4}', date_str)
    if match:
        day, month, year = match.groups()
        return f"{day.zfill(2)}.{month.zfill(2)}.{year}"
    
    # Handle the case like "01.-02.12.2023" or "28.11.-01.12.2023"
    match = re.match(r'(\d{1,2})\.(\d{1,2})?\.?-\d{1,2}\.(\d{1,2})\.(\d{4})', date_str)
    if match:
        groups = match.groups()
        if groups[1]:  # If start month is provided
            day, month, _, year = groups
        else:  # If start month is not provided
            day, month, year = groups[0], groups[2], groups[3]
        return f"{day.zfill(2)}.{month.zfill(2)}.{year}"
    
    # If it's already in the correct format, return as is
    if re.match(r'\d{2}\.\d{2}\.\d{4}', date_str):
        return date_str
    
    # If it's a single date without year, add the current year
    if re.match(r'\d{1,2}\.\d{1,2}\.?$', date_str):
        return f"{date_str.rstrip('.')}1970"  # Using 1970 as a placeholder year
    
    # If it's a date range
    if '-' in date_str:
        try:
            start, end = date_str.rsplit('-', 1)
            start = start.strip()
            end = end.strip()
            
            # Extract day, month, and year components
            start_parts = re.findall(r'\d+', start)
            end_parts = re.findall(r'\d+', end)
            
            # Ensure we have at least day and month for start date
            if len(start_parts) < 2:
                return None  # Return None if format is unexpected
            
            # Get year from start date if available, otherwise from end date
            if len(start_parts) == 3:  # Full start date provided
                year = start_parts[2]
            elif len(end_parts) == 3:  # Full end date provided
                year = end_parts[2]
            else:
                year = str(datetime.now().year)
            
            # Construct the full start date
            return f"{start_parts[0].zfill(2)}.{start_parts[1].zfill(2)}.{year}"
        except Exception:
            return None
    
    # If it's a single date with year
    if re.match(r'\d{1,2}\.\d{1,2}\.\d{4}', date_str):
        parts = date_str.split('.')
        return f"{parts[0].zfill(2)}.{parts[1].zfill(2)}.{parts[2]}"
    
    # If we can't parse the date, return None
    return None

def extract_location(event):
    if pd.isna(event) or not isinstance(event, str):
        return 'Unknown'

    # Look for country codes in brackets from right to left
    matches = re.findall(r'\((\w+)\)', event)
    
    for match in reversed(matches):
        # Check if the extracted text is likely to be a country code
        if (len(match) == 3 or len(match) == 2) and match.isupper():
            return match
    
    # If no country code is found, return the last bracketed text
    if matches:
        #print(f"Unusual location format found: {matches[-1]} in event: {event}")
        return matches[-1]
    else:
        #print(f"No location found in event: {event}")
        return 'Unknown'


def categorize_age_group(age):
    if pd.isna(age):
        return 'Unknown'
    elif age < 20:
        return 'Under 20'
    elif 20 <= age < 30:
        return '20-29'
    elif 30 <= age < 40:
        return '30-39'
    elif 40 <= age < 50:
        return '40-49'
    elif 50 <= age < 60:
        return '50-59'
    elif 60 <= age < 70:
        return '60-69'
    else:
        return '70+'

def add_age_group(df):
    df['Age Group'] = df['Age'].apply(categorize_age_group)
    return df

def add_runner_statistics(df):
    # Sort the dataframe by Runner ID and Date
    df = df.sort_values(['Runner ID', 'Date'])
    
    # Group by Runner ID
    grouped = df.groupby('Runner ID')
    
    # Number of races (Experience Level)
    df['Race Count'] = grouped.cumcount()
    
    # Determine the distance to use for cumulative calculation
    df['Distance For Cumulative'] = df.apply(
        lambda row: row['Distance Finish'] if row['Event Type'] == 'Time' else row['Distance KM'],
        axis=1
    )
    
    # Cumulative sum of Distance (excluding current race)
    df['Cumulative Distance KM'] = grouped['Distance For Cumulative'].transform(
        lambda x: x.shift().cumsum()
    )
    
    # Rolling average of Winner Percentage (excluding current race)
    #df['Avg Winner Percentage'] = grouped['Winner Percentage'].transform(
    #    lambda x: x.shift().expanding().mean()
    #)
    
    # Remove the temporary column
    df = df.drop('Distance For Cumulative', axis=1)
    
    # Replace NaN values with 0 for first race of each runner
    df['Race Count'] = df['Race Count'].fillna(0)
    df['Cumulative Distance KM'] = df['Cumulative Distance KM'].fillna(0)
    #df['Avg Winner Percentage'] = df['Avg Winner Percentage'].fillna(0)
    
    return df

def extract_finishers(df):
    def parse_finishers(finishers_str):
        if pd.isna(finishers_str):
            return None, None, None
        
        try:
            finishers_str = str(finishers_str).strip()
            match = re.match(r'(\d+)\s*\((\d+)\s*M,\s*(\d+)\s*F\)', finishers_str)
            if match:
                total = int(match.group(1))
                male = int(match.group(2))
                female = int(match.group(3))
                return total, male, female
            else:
                print(f"Unmatched finishers string: {finishers_str}")
                return None, None, None
        except Exception as e:
            print(f"Error processing finishers string: {finishers_str}")
            print(f"Error: {str(e)}")
            return None, None, None

    # Apply the function and handle any errors
    result = df['Finishers'].apply(parse_finishers)
    
    # Split the result into separate columns
    df[['Total Finishers', 'Male Finishers', 'Female Finishers']] = pd.DataFrame(result.tolist(), index=df.index)
    
    return df



def add_elevation_gain_per_km(df):
    # Calculate elevation gain per km
    df['Elevation Gain per KM'] = df.apply(
        lambda row: row['Elevation Gain'] / row['Distance KM'] 
        if pd.notnull(row['Elevation Gain']) and pd.notnull(row['Distance KM']) and row['Distance KM'] != 0 
        else np.nan, 
        axis=1
    )
    
    # Calculate the median elevation gain per km for each terrain type
    median_elevation_by_terrain = df.groupby('Terrain')['Elevation Gain per KM'].median()
    
    # Fill NaN values with the median for the corresponding terrain type
    df['Elevation Gain per KM'] = df.apply(
        lambda row: median_elevation_by_terrain[row['Terrain']] 
        if pd.isnull(row['Elevation Gain per KM']) and row['Terrain'] in median_elevation_by_terrain
        else row['Elevation Gain per KM'],
        axis=1
    )
    
    return df

def calculate_performance_ratio(df):
    def calculate_ratio(row):
        if pd.notnull(row['Time Seconds Finish']) and pd.notnull(row['Time Seconds Winner']):
            # For time-based events, a lower time is better
            if row['Time Seconds Finish'] == 0:
                return None  # Avoid divide by zero
            return row['Time Seconds Winner'] / row['Time Seconds Finish']
        elif pd.notnull(row['Distance Finish']) and pd.notnull(row['Distance Winner']):
            # For distance-based events, a higher distance is better
            if row['Distance Winner'] == 0:
                return None  # Avoid divide by zero
            return row['Distance Finish'] / row['Distance Winner']
        else:
            return None

    df['Performance Ratio'] = df.apply(calculate_ratio, axis=1)
    return df



def clean_data(df):
    df = split_runner_name(df)
    df = split_distance_column(df)
    df = process_performance(df, 'Performance', 'Time Seconds Finish', 'Distance Finish')
    df = process_performance(df, 'Winner Time', 'Time Seconds Winner', 'Distance Winner')
    df = calculate_performance_ratio(df)
    df = extract_finishers(df)
    
    df['Terrain'] = df['Terrain'].apply(standardize_terrain)
    df['Distance KM'] = df['Distance/Time'].apply(convert_miles_to_km).round(0)
    df['Distance KM'] = df['Distance KM'].replace(0, pd.NA)
    df['Average Speed'] = df['Time Seconds Finish'] / df['Distance KM']
    df['Race Location'] = df['Event'].apply(extract_location)
    df['Gender'] = df['M/F']
    
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
    df['Total Finishers'] = pd.to_numeric(df['Total Finishers'], errors='coerce')

    df['Finish Percentage'] = np.where(
        df['Total Finishers'] > 0,
        (df['Rank'] / df['Total Finishers']),
        np.nan
    )
    df['Finish Percentage'] = df['Finish Percentage'].round(2)
    df['Distance KM'] = df['Distance KM'].replace(0, pd.NA)
    df['Average Speed'] = df['Time Seconds Finish'] / df['Distance KM']
    df['Gender'] = df['M/F']
    df['Race Location'] = df['Event'].apply(extract_location)
    df['YOB'] = pd.to_numeric(df['YOB'], errors='coerce').astype('Int64')
    df['Date'] = pd.to_datetime(df['Date'].apply(parse_date_range), format='%d.%m.%Y')
    df['Avg.Speed km/h'] = df['Avg.Speed km/h'].astype(float)
    df['Age'] = df['Date'].dt.year - df['YOB']
    df = add_age_group(df)
    df['Elevation Gain'] = pd.to_numeric(df['Elevation Gain'].replace({'Hm': '', 'm': ''}, regex=True), errors='coerce')
    df = df.fillna({'M/F': 'Unknown', 'Cat': 'Unknown'})
    df['Club'] = df['Club'].str.strip().str.replace(r'[^\w\s]', '', regex=True)
    df['Nat.'] = df['Nat.'].str.strip().str.upper()
    df = add_runner_statistics(df)
    df = add_elevation_gain_per_km(df)
    
    df = df.sort_values(by=['Date', 'Race Location', 'Event', 'Rank'])

    return df

def main(input_folder, output_dir):
    all_files = glob.glob(os.path.join(input_folder, "all_events_data_*.csv"))
    
    for file_path in all_files:
        year = re.search(r'all_events_data_(\d{4})\.csv', os.path.basename(file_path))
        if year:
            year = year.group(1)
            print(f"Processing data for year {year}")
            
            result_df = pd.read_csv(file_path, low_memory=False)
            df_clean = clean_data(result_df)

            columns_to_keep = ['Runner ID','First Name','Surname','Nat.','Gender','Age','Age Group','Cat','YOB',
                               'Race Count','Cumulative Distance KM',
                               'Event ID','Event','Event Type','Date','Race Location','Elevation Gain','Elevation Gain per KM',
                               'Total Finishers','Male Finishers','Female Finishers',
                               'Rank','Rank M/F','Cat. Rank','Finish Percentage','Winner Percentage',
                               'Distance/Time','Distance KM','Terrain',
                               'Time Seconds Finish','Distance Finish','Average Speed','Avg.Speed km/h']
            df_clean = df_clean[columns_to_keep]

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, f"processed_data_{year}.csv")
            df_clean.to_csv(output_file, index=False)
            print(f"Processed data for year {year} saved to {output_file}")
        else:
            print(f"Skipping file {file_path} as it doesn't match the expected naming pattern.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV files from a folder and save output by year.")
    parser.add_argument("input_folder", help="Path to the folder containing CSV files")
    parser.add_argument("--output_dir", default="processed_data", help="Output directory for processed files (default: processed_data)")
    args = parser.parse_args()

    main(args.input_folder, args.output_dir)

# python scr/clean_data.py /output --output_dir /output/cleaned