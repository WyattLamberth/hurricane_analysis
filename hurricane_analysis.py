import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_hurdat_data(filename):
    """
    Load and process HURDAT data
    """
    df = pd.read_csv(filename)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'].astype(str).str.zfill(4), 
                                  format='%Y-%m-%d %H%M')
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    return df

def generate_sst_data(start_year, end_year):
    """
    Generate realistic SST data based on Gulf of Mexico climatology
    """
    print("Generating climatological SST data...")
    
    # Define grid
    lats = np.arange(15, 35, 0.5)
    lons = np.arange(-100, -80, 0.5)
    years = range(start_year, end_year + 1)
    months = range(1, 13)
    
    # Gulf of Mexico climatological parameters
    base_temps = {
        # Monthly climatological means for Gulf of Mexico
        1: 23.5,  # January
        2: 23.0,  # February
        3: 23.8,  # March
        4: 25.2,  # April
        5: 26.8,  # May
        6: 28.4,  # June
        7: 29.2,  # July
        8: 29.6,  # August
        9: 29.2,  # September
        10: 27.8, # October
        11: 25.9, # November
        12: 24.3  # December
    }
    
    sst_data = []
    
    for year in years:
        for month in months:
            base_temp = base_temps[month]
            
            for lat in lats:
                for lon in lons:
                    # Temperature variations
                    lat_effect = -0.15 * abs(lat - 25)  # Temperature decreases with latitude
                    lon_effect = -0.05 * abs(lon + 90)  # Slight east-west gradient
                    
                    # Add interannual variability
                    year_effect = np.random.normal(0, 0.3)
                    
                    # Calculate final SST
                    sst = base_temp + lat_effect + lon_effect + year_effect
                    
                    sst_data.append({
                        'year': year,
                        'month': month,
                        'lat': lat,
                        'lon': lon,
                        'sst': sst
                    })
    
    df = pd.DataFrame(sst_data)
    print(f"Generated {len(df)} SST records")
    return df

def calculate_environmental_indices(sst_df):
    """
    Calculate environmental indices from SST data
    """
    print("Calculating environmental indices...")
    
    # Calculate spatial means for each month
    monthly_means = sst_df.groupby(['year', 'month']).agg({
        'sst': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    # Flatten multi-level columns
    monthly_means.columns = ['year', 'month', 'sst_mean', 'sst_std', 'sst_min', 'sst_max']
    
    # Calculate 3-month running mean (proxy for seasonal patterns)
    monthly_means['sst_3month_mean'] = monthly_means['sst_mean'].rolling(window=3, center=True).mean()
    
    # Calculate annual cycle and anomalies (proxy for interannual variability)
    monthly_means['annual_cycle'] = monthly_means.groupby('month')['sst_mean'].transform('mean')
    monthly_means['sst_anomaly'] = monthly_means['sst_mean'] - monthly_means['annual_cycle']
    
    return monthly_means

def analyze_hurricane_sst_relationships(combined_data):
    """
    Analyze relationships between SST and hurricane characteristics
    """
    # Monthly statistics
    monthly_stats = combined_data.groupby('month').agg({
        'Wind': ['count', 'mean', 'max'],
        'sst_mean': 'mean',
        'sst_anomaly': 'mean'
    }).round(2)
    
    # Correlation analysis
    correlations = combined_data[[
        'Wind', 'Pressure', 'sst_mean', 'sst_3month_mean', 'sst_anomaly'
    ]].corr()
    
    # Intensity categories analysis
    intensity_by_sst = combined_data.groupby(
        pd.qcut(combined_data['sst_mean'], q=4)
    ).agg({
        'Wind': ['count', 'mean', 'max']
    }).round(2)
    
    return monthly_stats, correlations, intensity_by_sst

def main():
    try:
        # Load hurricane data
        print("Loading HURDAT data...")
        hurdat_df = load_hurdat_data('filtered_hurricane_data.csv')
        
        # Get date range
        start_year = hurdat_df['year'].min()
        end_year = hurdat_df['year'].max()
        print(f"Processing data from {start_year} to {end_year}")
        
        # Generate SST data
        sst_data = generate_sst_data(start_year, end_year)
        
        # Calculate environmental indices
        env_indices = calculate_environmental_indices(sst_data)
        
        # Combine datasets
        print("Combining datasets...")
        combined_data = pd.merge(
            hurdat_df,
            env_indices,
            on=['year', 'month'],
            how='left'
        )
        
        # Analyze relationships
        print("\nAnalyzing relationships...")
        monthly_stats, correlations, intensity_by_sst = analyze_hurricane_sst_relationships(combined_data)
        
        # Print results
        print("\nCorrelations with Hurricane Intensity:")
        print(correlations['Wind'].round(3))
        
        print("\nMonthly Statistics:")
        print(monthly_stats)
        
        print("\nHurricane Intensity by SST Quartile:")
        print(intensity_by_sst)
        
        # Save results
        print("\nSaving results...")
        combined_data.to_csv('hurricane_environmental_data.csv', index=False)
        
        return combined_data
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    combined_data = main()
    if combined_data is not None:
        print("\nSuccess! Analysis complete.")
        print(f"\nProcessed {len(combined_data)} records")
        print("\nEnvironmental variables included:", 
              [col for col in combined_data.columns if 'sst' in col])
        
        # Print some validation statistics
        print("\nSST Statistics:")
        print(combined_data[['sst_mean', 'sst_anomaly']].describe().round(2))
    else:
        print("\nFailed to complete analysis. Please check the error messages above.")