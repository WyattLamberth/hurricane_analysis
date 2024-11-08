import xarray as xr
from pathlib import Path

# Define the path to the directory containing .nc files
nc_dir = Path('hurricane_climate_data/humidity')

def inspect_nc_files(directory):
    """
    Inspect the structure of all .nc files in the specified directory.
    """
    # List all .nc files in the directory
    nc_files = list(directory.glob("*.nc"))
    
    if not nc_files:
        print("No .nc files found in the specified directory.")
        return
    
    # Loop through each file and inspect its structure
    for nc_file in nc_files:
        print(f"\nInspecting file: {nc_file.name}")
        
        # Open the .nc file with xarray
        with xr.open_dataset(nc_file) as ds:
            print(ds)  # Display the structure of the dataset

if __name__ == "__main__":
    inspect_nc_files(nc_dir)
