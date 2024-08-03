from datetime import datetime, timedelta
from geopy.distance import geodesic
import numpy as np
import os
import pandas as pd
from typing import List, Tuple

def write_site_ids(read_directory,working_folder_path):

    # Iterate over all hourly data files in read_directory
    for name in os.scandir(read_directory):
        
        # Get file path for current file name in read_directory
        path = name.path
        df = pd.read_csv(path)
        
        # Get string of file name to write working hourly data df
        name_str = str(name)
        write_name = name_str[11:-2]
        
        # Make a unique site identifier from state code, county code, and site number
        df['State Code'] = df['State Code'].apply(str)
        df['County Code'] = df['County Code'].apply(str)
        df['Site Num'] = df['Site Num'].apply(str)
        df["Site ID"] = df[["State Code", "County Code", "Site Num"]].apply("-".join, axis=1) 
    
        # Write df that includes site IDs to working hourly data directory
        write_path = os.path.join(working_folder_path,write_name)
        df.to_csv(write_path)

def create_site_id_files():
        # Create a working folder if no folder exists. 
        # This folder will be deleted after completion of data processing
        current_directory = os.getcwd()
        read_directory = os.path.join(current_directory,'hourly_data')
        working_folder_path = os.path.join(current_directory,'working_hourly_data')
        if not os.path.exists(working_folder_path):
            os.makedirs(working_folder_path)
            
        # Create working hourly data files with Site ID added
        write_site_ids(read_directory,working_folder_path)

def get_candidate_target_sites(read_directory: str, years: List[int]) -> List[int]:
    """
    Get a list of unique Site IDs from the first year's dataset.

    Parameters:
        read_directory (str): The directory path where the data files are stored.
        years (List[int]): List of years to be processed.

    Returns:
        List[int]: A list of unique Site IDs.
    """
    year = years[0]
    filename = f'hourly_88101_{year}.csv'
    filepath = os.path.join(read_directory, filename)

    df = pd.read_csv(filepath)
    unique_site_ids = df['Site ID'].unique()

    return list(unique_site_ids)

def check_completeness(years: List[int], read_directory: str, target_candidates: List[int], pct_completeness_thr: float, pct_negative_thr: float) -> Tuple[List[int], pd.DataFrame]:
    """
    Check the completeness of data for each site over all specified years.

    Parameters:
        years (List[int]): List of years to be processed.
        read_directory (str): The directory path where the data files are stored.
        target_candidates (List[int]): List of target site IDs.

    Returns:
        Tuple[List[int], pd.DataFrame]: A tuple containing the cull set of sites to be removed and a DataFrame with site completeness details.
    """
    cull_set = set()  
    key_pairs = []

    for year in years:
        filename = f'hourly_88101_{year}.csv'
        filepath = os.path.join(read_directory, filename)
        df = pd.read_csv(filepath, parse_dates={'DateTime': ['Date GMT', 'Time GMT']})

        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)

        # Filter out February 29th
        df = df[~((df.index.month == 2) & (df.index.day == 29))]

        for site in target_candidates:
            df_site = df[df['Site ID'] == site]

            if not df_site.empty:
                lat = df_site['Latitude'].iloc[0]
                long = df_site['Longitude'].iloc[0]

                numerics = df_site.select_dtypes('number').resample('h').mean()
                strings = df_site.select_dtypes('object').astype(str).resample('h').agg(lambda x: ','.join(set(x.dropna())))
                df_site_resampled = numerics.join(strings)

                sample_count = df_site_resampled['Sample Measurement'].count()
                low_negatives_count = (df_site_resampled['Sample Measurement'] < -2).sum()

                percent_completeness = sample_count / 8760
                percent_low_negative = low_negatives_count / 8760

                if percent_completeness < pct_completeness_thr or percent_low_negative > (1-pct_negative_thr):
                    cull_set.add(site)

                key_pair = {
                    'Site ID': site,
                    'Sample Count': sample_count,
                    'Sample Count <-2': low_negatives_count,
                    'Latitude': lat,
                    'Longitude': long,
                    'Year': year
                }
                key_pairs.append(key_pair)
            else:
                cull_set.add(site)

    key_pairs_df = pd.DataFrame(key_pairs)
    return list(cull_set), key_pairs_df
        
def export_completeness_csv_files(key_pairs: pd.DataFrame, years: List[int]):
    """
    Export site completeness data to CSV files for each year.

    Parameters:
        key_pairs (pd.DataFrame): DataFrame containing site completeness details.
        years (List[int]): List of years to be processed.
    """
    key_pairs.to_csv('site_completeness_all_years.csv', index=False)
    
    for year in years:
        year_df = key_pairs[key_pairs['Year'] == year].drop_duplicates()
        year_df.to_csv(f'site_completeness_{year}.csv', index=False)

def get_closest_sites(cities: pd.DataFrame, key_pairs: pd.DataFrame, unique_cull_set: List[int]) -> pd.DataFrame:
    """
    Find the closest monitoring sites to specified city coordinates.

    Parameters:
        cities (pd.DataFrame): DataFrame containing city location data.
        key_pairs (pd.DataFrame): DataFrame containing site completeness details.
        unique_cull_set (List[int]): List of Site IDs to be culled.

    Returns:
        pd.DataFrame: DataFrame containing the closest sites for each city.
    """
    matching_dict = []

    # Filter out culled sites
    available_sites = key_pairs[~key_pairs['Site ID'].isin(unique_cull_set)]

    for _, city_row in cities.iterrows():
        city, target_lat, target_lon = city_row['Location'], city_row['Latitude'], city_row['Longitude']

        available_sites['Distance'] = available_sites.apply(
            lambda row: geodesic((target_lat, target_lon), (row['Latitude'], row['Longitude'])).miles, axis=1
        )

        closest_site = available_sites.nsmallest(1, 'Distance').iloc[0]

        matching_dict_pairs = {
            'City': city,
            'City Latitude': target_lat,
            'City Longitude': target_lon,
            'Closest Site ID': closest_site['Site ID'],
            'Closest Site Lat': closest_site['Latitude'],
            'Closest Site Lon': closest_site['Longitude'],
            'Distance': closest_site['Distance']
        }
        matching_dict.append(matching_dict_pairs)

    matching_df = pd.DataFrame(matching_dict)
    return matching_df

def export_distance_matrix(matching_df: pd.DataFrame, key_pairs: pd.DataFrame, unique_cull_set: List[int]) -> pd.DataFrame:
    """
    Create and export a matrix of distances between sites.

    Parameters:
        matching_df (pd.DataFrame): DataFrame containing closest sites for each city.
        key_pairs (pd.DataFrame): DataFrame containing site completeness details.
        unique_cull_set (List[int]): List of Site IDs to be culled.

    Returns:
        pd.DataFrame: A distance matrix DataFrame.
    """
    distance_matrix = pd.DataFrame()

    # Filter out culled sites
    available_sites = key_pairs[~key_pairs['Site ID'].isin(unique_cull_set)]

    for _, match_row in matching_df.iterrows():
        target_site_id = match_row['Closest Site ID']
        target_lat = match_row['Closest Site Lat']
        target_lon = match_row['Closest Site Lon']

        available_sites['Distance'] = available_sites.apply(
            lambda row: geodesic((target_lat, target_lon), (row['Latitude'], row['Longitude'])).miles, axis=1
        )

        site_distances = available_sites[['Site ID', 'Distance']].set_index('Site ID').rename(columns={'Distance': target_site_id})
        
        distance_matrix = distance_matrix.join(site_distances, how='outer')

    # distance_matrix.to_csv('distance_matrix.csv')
    return distance_matrix

def weather_fill(site_df, dist_mat):
    path = os.getcwd()
    read_directory = os.path.join(path,'working_hourly_data')
    output_directory = os.path.join(path,'annual_data_complete')

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Create list of target site IDs
    site_id_vals = dist_mat.columns

    # Define years
    years = [2018]

    years = np.sort(years)
    earliest_year = years[0] - 1

    # Loop over site IDs
    for site_id in site_id_vals:
        # Get target location info
        target_location_info = site_df.loc[site_id]
        target_location = target_location_info['City']
        target_latitude = target_location_info['Closest Site Lat']
        target_longitude = target_location_info['Closest Site Lon']
        
        # Get and sort distances for Site ID
        distances = dist_mat[site_id].sort_values()

        # Loop over years
        for year in years:
            # Load hourly data
            filename = f'hourly_88101_{year}.csv'
            file_path = os.path.join(read_directory, filename)
            df = pd.read_csv(file_path)
            
            # Create DateTime column
            df['DateTime'] = pd.to_datetime(df['Date GMT'] + ' ' + df['Time GMT'])
            
            # Filter data for current Site ID
            df_filter = df[df['Site ID'] == site_id].copy()
            df_filter.set_index('DateTime', inplace=True)
            
            # Calculate time difference
            df_filter['delta'] = df_filter.index.to_series().diff().dt.total_seconds() / 60
            
            # Resample to hourly
            numerics = df_filter.select_dtypes('number').resample('h').mean()
            strings = df_filter.select_dtypes('object').astype(str).resample('h').agg(lambda x: ','.join(set(x.dropna())))
            new_df_filter = numerics.join(strings)

            # Create ideal time index for the year
            ideal_time = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31 23:00:00', freq='H')
            new_df_filter = new_df_filter.reindex(ideal_time)

            # Load previous year data for missing start data
            if year > earliest_year:
                filename_prev = f'hourly_88101_{year - 1}.csv'
                path_prev = os.path.join(read_directory, filename_prev)
                df_prev = pd.read_csv(path_prev)
                df_prev['DateTime'] = pd.to_datetime(df_prev['Date GMT'] + ' ' + df_prev['Time GMT'])
                df_prev.set_index('DateTime', inplace=True)
                
            numerics = df_prev[df_prev['Site ID'] == site_id].select_dtypes('number').resample('h').mean()
            strings = df_prev[df_prev['Site ID'] == site_id].select_dtypes('object').astype(str).resample('h').agg(lambda x: ','.join(set(x.dropna())))
            df_prev_filtered = numerics.join(strings)

            # Fill start data from previous year
            time_init = new_df_filter.index[0]
            time_final_prev = df_filter.index[0] - timedelta(hours=1)
            if len(df_prev_filtered) > 0:
                new_df_filter.loc[time_init:time_final_prev, ['Sample Measurement', 'Latitude', 'Longitude']] = df_prev_filtered.loc[time_init:time_final_prev, ['Sample Measurement', 'Latitude', 'Longitude']]

            # Iterate to fill missing data from previous year file
            j = 1
            while new_df_filter['Sample Measurement'].loc[time_init:time_final_prev].isna().sum() > 0:
                
                df_filter['delta'] = df_filter.index.to_series().diff().dt.total_seconds() / 60
                
                closest_site_id = distances.index[j]
                closest_df = df_prev[df_prev['Site ID'] == closest_site_id].copy()
                if len(closest_df) > 0:
                    # closest_df.set_index('DateTime', inplace=True)
                    
                    numerics = closest_df.select_dtypes('number').resample('h').mean()
                    strings = closest_df.select_dtypes('object').astype(str).resample('h').agg(lambda x: ','.join(set(x.dropna())))
                    closest_df = numerics.join(strings)
    
                    new_df_filter.loc[time_init:time_final_prev, ['Sample Measurement', 'Latitude', 'Longitude']] = closest_df.loc[time_init:time_final_prev, ['Sample Measurement', 'Latitude', 'Longitude']]
                    df_filter = new_df_filter.dropna(subset=['Sample Measurement'])
                    
                if distances[j] > 500:
                    break
                j += 1

            # Iterate to fill missing data from current year file
            j = 1
            while new_df_filter['Sample Measurement'].isna().sum() > 0:
                
                df_filter['delta'] = df_filter.index.to_series().diff().dt.total_seconds() / 60

                closest_site_id = distances.index[j]
                closest_df = df[df['Site ID'] == closest_site_id].copy()
                closest_df.set_index('DateTime', inplace=True)
                
                numerics = closest_df.select_dtypes('number').resample('h').mean()
                strings = closest_df.select_dtypes('object').astype(str).resample('h').agg(lambda x: ','.join(set(x.dropna())))
                closest_df = numerics.join(strings)

                for i in range(1, len(df_filter)):
                    delta = df_filter['delta'].iloc[i]
                    if delta > 60 and delta <= 240:
                        # Interpolate for gaps <= 4 hours
                        new_df_filter['Sample Measurement'].iloc[i-1:i+1] = new_df_filter['Sample Measurement'].iloc[i-1:i+1].interpolate(method='linear')
                    elif delta > 240:
                        # Use data from closest station for gaps > 4 hours
                        time_prev = df_filter.index[i-1]
                        time_current = df_filter.index[i]
                        new_df_filter.loc[time_prev:time_current, ['Sample Measurement', 'Latitude', 'Longitude']] = closest_df.loc[time_prev:time_current, ['Sample Measurement', 'Latitude', 'Longitude']]

                # Fill start and end data
                # Check if new_df_filter is still missing data from beginning
                
                new_df_filter.loc[:df_filter.index[0] - timedelta(hours=1), ['Sample Measurement', 'Latitude', 'Longitude']] = closest_df.loc[:df_filter.index[0] - timedelta(hours=1), ['Sample Measurement', 'Latitude', 'Longitude']]
                new_df_filter.loc[df_filter.index[-1] + timedelta(hours=1):, ['Sample Measurement', 'Latitude', 'Longitude']] = closest_df.loc[df_filter.index[-1] + timedelta(hours=1):, ['Sample Measurement', 'Latitude', 'Longitude']]

                df_filter = new_df_filter.dropna(subset=['Sample Measurement'])
                j += 1

            complete_df = new_df_filter.copy()

            # Add interpolated check
            complete_df['Interpolated Check'] = np.where(complete_df['Latitude'].isna() & complete_df['Sample Measurement'].notna(), 1, 0)

            # Calculate distance from target
            complete_df['Distance from Target'] = complete_df.apply(
                lambda row: geodesic((target_latitude, target_longitude), (row['Latitude'], row['Longitude'])).miles
                if not pd.isna(row['Latitude']) else np.nan, axis=1
            )

            # Remove leap day
            complete_df = complete_df[~((complete_df.index.month == 2) & (complete_df.index.day == 29))]
            
            # Save to CSV
            file_name = f'{target_location}_{year}.csv'
            file_path = os.path.join(output_directory, file_name)
            complete_df.to_csv(file_path, columns=['Latitude', 'Longitude', 'Sample Measurement', 'Interpolated Check', 'Distance from Target'])

def process_all_steps(years,percent_completeness_threshold,percent_negative_threshold):

    # Creates a unique Site ID for each weather station 
    create_site_id_files()
    
    # Filter data belonging to potential target sites
    #  - Data must be >90% complete for each year
    #  - Data must have >98% PM2.5 values greater than -2
    
    # Specify hourly data directory
    # This folder should contain all years you are interested in processing, as well as one year prior,
    # e.g. for years = [2018,2019] you should include data for 2017.
    path = os.getcwd()
    read_directory = os.path.join(path,'working_hourly_data')
    
    # Get_candidate_target_sites (uses unique())
    target_candidates = get_candidate_target_sites(read_directory,years)
    
    # Check completeness of all sites over all years
    cull_set,key_pairs = check_completeness(years,read_directory,target_candidates,percent_completeness_threshold,percent_negative_threshold)
    unique_cull_set = list(set(cull_set))
    
    # Export completeness info for use later
    export_completeness_csv_files(key_pairs,years)
    
    # Import target city coordinates and find closest sites
    cities = pd.read_csv(os.path.join(path,'target_sites.csv'))
    # cities = pd.read_csv(r'/Users/cdgreen/Desktop/Weather Files for Theresa/2024-04-10 Workflow/target_sites.csv')
    matching_df = get_closest_sites(cities,key_pairs,unique_cull_set)
    # matching_df.to_csv('closest_sites.csv')

    # Create and export distance matrix
    distance_matrix =  export_distance_matrix(matching_df,key_pairs,unique_cull_set)
    
    # Fill weather data
    weather_fill(matching_df,distance_matrix)
    
    
    
if __name__ == '__main__':

    # Specify which years will be processed
    years = [2018]
    
    # Specify required completeness of data sets included in the candidate set. 
    percent_completeness_threshold = 0.90
    
    # Specify the amount of values required to be greater than -2
    percent_negative_threshold = 0.98
    
    process_all_steps(years,percent_completeness_threshold,percent_negative_threshold)

    
