import requests
from pathlib import Path
from tqdm import tqdm
import os

def download_sst_data(start_year, end_year, output_dir='climate_data'):
    """
    Download NOAA OISST daily mean data files for specified years
    
    Parameters:
    -----------
    start_year : int
        First year to download
    end_year : int
        Last year to download
    output_dir : str
        Directory to save files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    base_url = "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres"
    
    for year in range(start_year, end_year + 1):
        filename = f"sst.day.mean.{year}.nc"
        url = f"{base_url}/{filename}"
        output_path = os.path.join(output_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(output_path):
            print(f"{filename} already exists, skipping...")
            continue
        
        print(f"Downloading {filename}...")
        
        # Stream the download with progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
            print(f"Successfully downloaded {filename}")
        else:
            print(f"Failed to download {filename}")

def download_oni_data(output_dir='climate_data'):
    """
    Download Oceanic Niño Index (El Niño/La Niña) data
    Source: NOAA Climate Prediction Center
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    output_path = os.path.join(output_dir, 'oni_data.txt')
    
    print("Downloading ONI data...")
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print("Successfully downloaded ONI data")
    else:
        print("Failed to download ONI data")

def download_amo_data(output_dir='climate_data'):
    """
    Download Atlantic Multidecadal Oscillation (AMO) data
    Source: NOAA ESRL
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    url = "https://psl.noaa.gov/data/correlation/amon.us.data"
    output_path = os.path.join(output_dir, 'amo_data.txt')
    
    print("Downloading AMO data...")
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print("Successfully downloaded AMO data")
    else:
        print("Failed to download AMO data")

def download_wind_shear_data(start_year, end_year, output_dir='climate_data'):
    """
    Download vertical wind shear data from NCEP/NCAR Reanalysis
    Source: NOAA PSL
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    base_url = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/pressure"
    variables = ['uwnd', 'vwnd']  # u and v wind components
    
    for year in range(start_year, end_year + 1):
        for var in variables:
            filename = f"{var}.{year}.nc"
            url = f"{base_url}/{filename}"
            output_path = os.path.join(output_dir, filename)
            
            if os.path.exists(output_path):
                print(f"{filename} already exists, skipping...")
                continue
            
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            if response.status_code == 200:
                with open(output_path, 'wb') as f, tqdm(
                    desc=filename,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for data in response.iter_content(chunk_size=1024):
                        size = f.write(data)
                        pbar.update(size)
                print(f"Successfully downloaded {filename}")
            else:
                print(f"Failed to download {filename}")

def download_relative_humidity_data(start_year, end_year, output_dir='climate_data'):
    """
    Download relative humidity data from NCEP/NCAR Reanalysis
    Source: NOAA PSL
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    base_url = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/pressure"
    
    for year in range(start_year, end_year + 1):
        filename = f"rhum.{year}.nc"
        url = f"{base_url}/{filename}"
        output_path = os.path.join(output_dir, filename)
        
        if os.path.exists(output_path):
            print(f"{filename} already exists, skipping...")
            continue
        
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
            print(f"Successfully downloaded {filename}")
        else:
            print(f"Failed to download {filename}")

def download_all_climate_data(start_year, end_year):
    """
    Download all climate data
    """
    # Create main output directory
    base_dir = 'hurricane_climate_data'
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    
    # Download each dataset
    print("\nDownloading SST data...")
    download_sst_data(start_year, end_year, os.path.join(base_dir, 'sst'))
    
    print("\nDownloading ONI data...")
    download_oni_data(os.path.join(base_dir, 'oni'))
    
    print("\nDownloading AMO data...")
    download_amo_data(os.path.join(base_dir, 'amo'))
    
    print("\nDownloading wind shear data...")
    download_wind_shear_data(start_year, end_year, os.path.join(base_dir, 'wind'))
    
    print("\nDownloading relative humidity data...")
    download_relative_humidity_data(start_year, end_year, os.path.join(base_dir, 'humidity'))
    
    print("\nAll downloads complete!")

# Example usage:
if __name__ == "__main__":
    # Download all climate data for start_year to end_year
    download_all_climate_data(1999, 2000)