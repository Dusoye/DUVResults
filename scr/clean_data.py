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
    data['Surname'] = ''
    data['First Name'] = ''
    
    if 'Surname, first name' in data.columns:
        mask = data['Surname, first name'].notna()
        temp = data.loc[mask, 'Surname, first name'].str.split(',', expand=True)
        data.loc[mask, 'Surname'] = temp[0].str.strip()
        data.loc[mask, 'First Name'] = temp[1].str.strip() if temp.shape[1] > 1 else ''
    
    original_name_column = 'Original name\nSurname, first name'
    if original_name_column in data.columns:
        mask = (data['Surname'] == '') & (data[original_name_column].notna())
        temp = data.loc[mask, original_name_column].str.split(',', expand=True)
        data.loc[mask, 'Surname'] = temp[0].str.strip()
        data.loc[mask, 'First Name'] = temp[1].str.strip() if temp.shape[1] > 1 else ''
    
    return data

def parse_performance(data, column_name, time_column, distance_column):
    time_regex = re.compile(r'(?:(\d+)d )?(\d{1,2}):(\d{2}):(\d{2}) h')
    distance_regex = re.compile(r'(\d+\.?\d*) km')
    
    def parse_entry(entry):
        if 'km' in entry:
            match = distance_regex.search(entry)
            return (None, float(match.group(1)) if match else None)
        else:
            match = time_regex.search(entry)
            if match:
                days, hours, minutes, seconds = match.groups(default='0')
                total_seconds = timedelta(days=int(days), hours=int(hours), minutes=int(minutes), seconds=int(seconds)).total_seconds()
                return (total_seconds, None)
        return (None, None)
    
    data[time_column], data[distance_column] = zip(*data[column_name].apply(parse_entry))
    return data

def split_distance_column(df):
    distance_pattern = re.compile(r'(\d+\.?\d*)\s*(km|mi|h)')
    race_type_pattern = re.compile(r'(\d+\.?\d*\s*(km|mi|h))\s*(.*)')
    
    def split_distance(entry):
        distance_match = distance_pattern.search(entry)
        race_type_match = race_type_pattern.search(entry)
        
        if distance_match:
            distance = distance_match.group(0)
            unit = distance_match.group(2)
            race_type = 'Time' if unit == 'h' else 'Distance'
        else:
            distance = None
            race_type = None
        
        terrain = race_type_match.group(3).strip() if race_type_match and len(race_type_match.groups()) > 2 else None
        
        return distance, terrain, race_type
    
    df['Distance/Time'], df['Terrain'], df['Event Type'] = zip(*df['Distance'].apply(split_distance))
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
    
    if re.match(r'\d{2}\.\d{2}\.\d{4}', date_str):
        return date_str
    
    if re.match(r'\d{1,2}\.\d{1,2}\.?$', date_str):
        return f"{date_str.rstrip('.')}1970"
    
    if '-' in date_str:
        start, end = date_str.split('-')
        start = start.strip()
        end = end.strip()
        
        start_parts = re.findall(r'\d+', start)
        end_parts = re.findall(r'\d+', end)
        
        if len(start_parts) < 2:
            return None
        
        year = end_parts[-1] if len(end_parts) == 3 else str(datetime.now().year)
        
        return f"{start_parts[0].zfill(2)}.{start_parts[1].zfill(2)}.{year}"
    
    if re.match(r'\d{1,2}\.\d{1,2}\.\d{4}', date_str):
        parts = date_str.split('.')
        return f"{parts[0].zfill(2)}.{parts[1].zfill(2)}.{parts[2]}"
    
    return None

def extract_location(event):
    match = re.search(r'\((\w+)\)$', event)
    return match.group(1) if match else 'Unknown'

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
    df = df.sort_values(['Runner ID', 'Date'])
    grouped = df.groupby('Runner ID')
    
    df['Race Count'] = grouped.cumcount()
    
    df['Distance For Cumulative'] = df.apply(
        lambda row: row['Distance Finish'] if row['Event Type'] == 'Time' else row['Distance KM'],
        axis=1
    )
    
    df['Cumulative Distance KM'] = grouped['Distance For Cumulative'].transform(
        lambda x: x.shift().cumsum()
    )
    
    df = df.drop('Distance For Cumulative', axis=1)
    
    df['Race Count'] = df['Race Count'].fillna(0)
    df['Cumulative Distance KM'] = df['Cumulative Distance KM'].fillna(0)
    
    return df

def extract_finishers(df):
    def parse_finishers(finishers_str):
        match = re.match(r'(\d+)\s*\((\d+)\s*M,\s*(\d+)\s*F\)', finishers_str)
        if match:
            total = int(match.group(1))
            male = int(match.group(2))
            female = int(match.group(3))
            return total, male, female
        else:
            return None, None, None

    df[['Total Finishers', 'Male Finishers', 'Female Finishers']] = df['Finishers'].apply(parse_finishers).apply(pd.Series)
    return df

def add_elevation_gain_per_km(df):
    df['Elevation Gain per KM'] = df.apply(
        lambda row: row['Elevation Gain'] / row['Distance KM'] 
        if pd.notnull(row['Elevation Gain']) and pd.notnull(row['Distance KM']) and row['Distance KM'] != 0 
        else np.nan, 
        axis=1
    )
    
    median_elevation_by_terrain = df.groupby('Terrain')['Elevation Gain per KM'].median()
    
    df['Elevation Gain per KM'] = df.apply(
        lambda row: median_elevation_by_terrain[row['Terrain']] 
        if pd.isnull(row['Elevation Gain per KM']) and row['Terrain'] in median_elevation_by_terrain
        else row['Elevation Gain per KM'],
        axis=1
    )
    
    return df

def calculate_winner_percentage(df):
    distance_mask = df['Event Type'] == 'Distance'
    df.loc[distance_mask, 'Winner Percentage'] = (1 - df.loc[distance_mask, 'Time Seconds Winner'] / df.loc[distance_mask, 'Time Seconds Finish']).round(2)

    time_mask = df['Event Type'] == 'Time'
    df.loc[time_mask, 'Winner Percentage'] = (1 - df.loc[time_mask, 'Distance Finish'] / df.loc[time_mask, 'Distance Winner']).round(2)

    return df

def clean_data(df):
    df = split_runner_name(df)
    df = split_distance_column(df)
    df = parse_performance(df, 'Performance', 'Time Seconds Finish', 'Distance Finish')
    df = parse_performance(df, 'Winner Time', 'Time Seconds Winner', 'Distance Winner')
    df = calculate_winner_percentage(df)
    df = extract_finishers(df)
    
    df['Terrain'] = df['Terrain'].apply(standardize_terrain)
    df['Distance KM'] = df['Distance/Time'].apply(convert_miles_to_km).round(0)
    df['Finish Percentage'] = (df['Rank'] / df['Total Finishers']).round(2)
    df['Distance KM'] = df['Distance KM'].replace(0, pd.NA)
    df['Average Speed'] = df['Time Seconds Finish'] / df['Distance KM']
    df['Race Location'] = df['Event'].apply(extract_location)
    df['Gender'] = df['M/F']

    df['Date'] = pd.to_datetime(df['Date'].apply(parse_date_range), format='%d.%m.%Y')
    df['YOB'] = pd.to_numeric(df['YOB'], errors='coerce').astype('Int64')
    df['Avg.Speed km/h'] = df['Avg.Speed km/h'].astype(float)
    df['Elevation Gain'] = pd.to_numeric(df['Elevation Gain'].replace({'Hm': '', 'm': ''}, regex=True), errors='coerce')

    df = df.fillna({'M/F': 'Unknown', 'Cat': 'Unknown'})
    df['Club'] = df['Club'].str.strip().str.replace(r'[^\w\s]', '', regex=True)
    df['Nat.'] = df['Nat.'].str.strip().str.upper()
    df['Age'] = df['Date'].dt.year - df['YOB']
    df = add_age_group(df)
    
    df = add_runner_statistics(df)
    df = add_elevation_gain_per_km(df)
    
    df = df.sort_values(by=['Date', 'Race Location', 'Event', 'Rank'])

    return df

def main(input_folder, output_file):
    result_df = load_and_concat_csv(input_folder, chunksize=100000)
    df_clean = clean_data(result_df)

    columns_to_keep = ['Runner ID','First Name','Surname','Nat.','Gender','Age','Age Group','Cat','YOB',
                       'Race Count','Cumulative Distance KM',
                       'Event ID','Event','Event Type','Date','Race Location','Elevation Gain','Elevation Gain per KM',
                       'Total Finishers','Male Finishers','Female Finishers',
                       'Rank','Rank M/F','Cat. Rank','Finish Percentage','Winner Percentage',
                       'Distance/Time','Distance KM','Terrain',
                       'Time Seconds Finish','Distance Finish','Average Speed','Avg.Speed km/h']
    df_clean = df_clean[columns_to_keep]

    df_clean.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV files from a folder.")
    parser.add_argument("input_folder", help="Path to the folder containing CSV files")
    parser.add_argument("--output", default="processed_data.csv", help="Output file name (default: processed_data.csv)")
    args = parser.parse_args()

    main(args.input_folder, args.output)