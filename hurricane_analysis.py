import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_hurdat_data(filename):
    """
    Load and process HURDAT2 data
    """
    df = pd.read_csv(filename)
    
    # Basic processing
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    
    return df

def download_ersst_data():
    """
    Download ERSST data directly using pandas
    """
    # ERSST monthly data URL (example)
    base_url = "https://www.ncei.noaa.gov/data/sea-surface-temperature-ersst/v5/access/ersst.v5.{year}.csv"
    
    years = range(1995, 2024)
    dfs = []
    
    for year in years:
        try:
            url = base_url.format(year=year)
            df = pd.read_csv(url)
            dfs.append(df)
        except Exception as e:
            print(f"Could not download data for {year}: {e}")
    
    return pd.concat(dfs)

def calculate_climate_indices(sst_data):
    """
    Calculate El Niño and AMO indices from SST data
    """
    # Example calculation (simplified)
    # You'll need to adjust the exact regions and calculation methods
    
    # El Niño 3.4 region (5°N-5°S, 170°W-120°W)
    nino34_mask = (
        (sst_data['lat'] >= -5) & 
        (sst_data['lat'] <= 5) & 
        (sst_data['lon'] >= 190) & 
        (sst_data['lon'] <= 240)
    )
    
    nino34 = sst_data[nino34_mask].groupby('time')['sst'].mean()
    
    # AMO region (0-60°N, 80°W-0°)
    amo_mask = (
        (sst_data['lat'] >= 0) & 
        (sst_data['lat'] <= 60) & 
        (sst_data['lon'] >= 280) & 
        (sst_data['lon'] <= 360)
    )
    
    amo = sst_data[amo_mask].groupby('time')['sst'].mean()
    
    return nino34, amo

def main():
    # Load hurricane data
    hurdat = load_hurdat_data('filtered_hurricane_data.csv')
    
    # Load SST data
    sst_data = download_ersst_data()
    
    # Calculate climate indices
    nino34, amo = calculate_climate_indices(sst_data)
    
    # Combine data
    combined_data = hurdat.copy()
    
    # Add climate indices to hurricane data
    for date in combined_data['datetime']:
        month_start = date.replace(day=1)
        combined_data.loc[combined_data['datetime'] == date, 'nino34'] = nino34.get(month_start, np.nan)
        combined_data.loc[combined_data['datetime'] == date, 'amo'] = amo.get(month_start, np.nan)
    
    return combined_data

if __name__ == "__main__":
    environmental_data = main()
    
    # Basic analysis
    print("\nCorrelations with hurricane intensity:")
    correlations = environmental_data[['Wind', 'nino34', 'amo']].corr()['Wind']
    print(correlations)
    
    # Save results
    environmental_data.to_csv('hurricane_environmental_data.csv', index=False)