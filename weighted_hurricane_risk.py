import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
import psutil
import gc  # Added gc for garbage collection

os.environ["DASK_NUM_WORKERS"] = "2"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

class MemoryTracker:
    @staticmethod
    def print_memory_usage():
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Current memory usage: {memory_mb:.2f} MB")
    
    @staticmethod
    def check_memory_threshold(threshold_mb=1000):
        """Check if memory usage is above threshold and cleanup if needed"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        if memory_mb > threshold_mb:
            print(f"Memory usage high ({memory_mb:.2f} MB), cleaning up...")
            gc.collect()
            return True
        return False

class WeightedHurricaneRiskAssessor:
    def __init__(self, data_dir='hurricane_climate_data'):
        self.data_dir = Path(data_dir)
        self.model = None
        self.scaler = StandardScaler()
        
        # Define feature weights (redistributing Saharan dust weight proportionally)
        # Original weights: SST(.45), Nino/Nina(.25), AMO(.15), Wind(.10), Dust(.05)
        # New weights without dust: SST(.474), Nino/Nina(.263), AMO(.158), Wind(.105)
        self.weights = {
            'sst': 0.474,
            'oni': 0.263,
            'amo': 0.158,
            'wind_shear': 0.105
        }
        
    def calculate_weighted_features(self, df):
        """
        Calculate weighted feature importance
        """
        # Normalize each feature first
        normalized_features = {}
        for feature in self.weights.keys():
            if feature in df.columns:
                # Min-max normalization
                min_val = df[feature].min()
                max_val = df[feature].max()
                normalized_features[feature] = (df[feature] - min_val) / (max_val - min_val)
        
        # Calculate weighted sum
        weighted_risk = np.zeros(len(df))
        for feature, weight in self.weights.items():
            if feature in normalized_features:
                # For wind shear, higher values indicate lower risk
                if feature == 'wind_shear':
                    weighted_risk += weight * (1 - normalized_features[feature])
                else:
                    weighted_risk += weight * normalized_features[feature]
        
        return weighted_risk
    
    def _load_sst_data(self):
        """Load and process SST data with memory-efficient approach"""
        print("Starting SST data processing...")
        sst_files = sorted(self.data_dir.glob('sst/sst.day.mean.*.nc'))
        
        if not sst_files:
            raise FileNotFoundError("No SST data files found in the specified directory")
        
        # Process files one at a time instead of using open_mfdataset
        all_data = []
        
        for file in sst_files:
            try:
                print(f"Processing {file.name}...")
                
                # Open single file with optimal chunking
                ds = xr.open_dataset(
                    file,
                    chunks={'time': 1, 'lat': None, 'lon': None}  # Let xarray choose optimal chunks
                )
                
                # Get all times for this file
                times = ds.time.values
                
                # Process each time step individually
                for time in times:
                    try:
                        # Select single time step and region
                        time_slice = ds.sel(
                            time=time,
                            lat=slice(18, 31),
                            lon=slice(-98, -80)
                        )
                        
                        # Calculate mean SST for the region
                        mean_sst = float(time_slice.sst.mean().compute())
                        
                        # Store results
                        all_data.append({
                            'Date': pd.Timestamp(time),
                            'sst': mean_sst
                        })
                        
                        # Clear memory
                        del time_slice
                        
                    except Exception as e:
                        print(f"Error processing time {time}: {str(e)}")
                        continue
                
                # Explicitly close dataset
                ds.close()
                
                # Force garbage collection
                import gc
                gc.collect()
                
            except Exception as e:
                print(f"Error processing {file.name}: {str(e)}")
                continue
        
        if not all_data:
            raise ValueError("No SST data could be processed successfully")
        
        # Convert list of dictionaries to DataFrame
        combined_df = pd.DataFrame(all_data)
        
        # Sort by date and remove duplicates
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        combined_df = combined_df.sort_values('Date').drop_duplicates(subset=['Date'])
        
        print("SST data processing complete!")
        return combined_df
        """Load and prepare data with memory tracking"""
        memory_tracker = MemoryTracker()
        
        try:
            print("Loading hurricane data...")
            memory_tracker.print_memory_usage()
            hurricane_df = pd.read_csv(hurricane_data_path)
            hurricane_df['Date'] = pd.to_datetime(hurricane_df['Date'])
            
            print("Creating risk categories...")
            hurricane_df['risk_category'] = pd.cut(
                hurricane_df['Wind'],
                bins=[0, 34, 64, 83, 96, 113, float('inf')],
                labels=['TD', 'TS', 'Cat1', 'Cat2', 'Cat3', 'Cat4-5']
            )
            
            memory_tracker.check_memory_threshold()
            
            try:
                print("Loading SST data...")
                memory_tracker.print_memory_usage()
                sst_data = self._load_sst_data()
            except Exception as e:
                print(f"Error loading SST data: {str(e)}")
                print("Using placeholder SST data...")
                sst_data = pd.DataFrame({'Date': hurricane_df['Date'].unique(), 'sst': 28.0})
            
            memory_tracker.check_memory_threshold()
            
            try:
                print("Loading wind data...")
                memory_tracker.print_memory_usage()
                wind_data = self._load_wind_data()
            except Exception as e:
                print(f"Error loading wind data: {str(e)}")
                print("Using placeholder wind data...")
                wind_data = pd.DataFrame({'Date': hurricane_df['Date'].unique(), 'wind_shear': 10.0})
            
            memory_tracker.check_memory_threshold()
            
            try:
                print("Loading ONI data...")
                oni_data = self._load_oni_data()
            except Exception as e:
                print(f"Error loading ONI data: {str(e)}")
                print("Using placeholder ONI data...")
                oni_data = pd.DataFrame({'Date': hurricane_df['Date'].unique(), 'oni': 0.0})
            
            try:
                print("Loading AMO data...")
                amo_data = self._load_amo_data()
            except Exception as e:
                print(f"Error loading AMO data: {str(e)}")
                print("Using placeholder AMO data...")
                amo_data = pd.DataFrame({'Date': hurricane_df['Date'].unique(), 'amo': 0.0})
            
            print("Merging datasets...")
            combined_df = hurricane_df.merge(
                sst_data, on='Date', how='left'
            ).merge(
                wind_data, on='Date', how='left'
            ).merge(
                oni_data, on='Date', how='left'
            ).merge(
                amo_data, on='Date', how='left'
            )
            
            print("Handling missing values...")
            combined_df = self._handle_missing_values(combined_df)
            
            print("Calculating weighted risk...")
            combined_df['weighted_risk'] = self.calculate_weighted_features(combined_df)
            
            print("Data preparation complete!")
            return combined_df
        
        except Exception as e:
            print(f"Error in data preparation: {str(e)}")
            raise
            
    def _load_wind_data(self):
        """Load and process wind shear data with memory-efficient approach"""
        print("Starting wind data processing...")
        u_files = sorted(self.data_dir.glob('wind/uwnd.*.nc'))
        v_files = sorted(self.data_dir.glob('wind/vwnd.*.nc'))
        
        if not u_files or not v_files:
            raise FileNotFoundError("Missing wind data files")
        
        all_data = []
        
        # Process one pair of files at a time
        for u_file, v_file in zip(u_files, v_files):
            try:
                print(f"Processing {u_file.name} and {v_file.name}...")
                
                # Open files with minimal chunking
                ds_u = xr.open_dataset(
                    u_file,
                    chunks={'time': 1}  # Process one time step at a time
                )
                
                ds_v = xr.open_dataset(
                    v_file,
                    chunks={'time': 1}
                )
                
                # Get all times
                times = ds_u.time.values
                
                # Process each time step individually
                for time in times:
                    try:
                        # Select single time step and region for u-wind
                        u_slice = ds_u.sel(
                            time=time,
                            lat=slice(18, 31),
                            lon=slice(-98, -80),
                            level=[200, 850]
                        )
                        
                        # Select single time step and region for v-wind
                        v_slice = ds_v.sel(
                            time=time,
                            lat=slice(18, 31),
                            lon=slice(-98, -80),
                            level=[200, 850]
                        )
                        
                        # Calculate wind shear components
                        u_shear = (u_slice.uwnd.sel(level=200) - u_slice.uwnd.sel(level=850)).compute()
                        v_shear = (v_slice.vwnd.sel(level=200) - v_slice.vwnd.sel(level=850)).compute()
                        
                        # Calculate total shear and mean
                        total_shear = np.sqrt(u_shear**2 + v_shear**2)
                        mean_shear = float(total_shear.mean().values)
                        
                        # Store results
                        all_data.append({
                            'Date': pd.Timestamp(time),
                            'wind_shear': mean_shear
                        })
                        
                        # Clear memory
                        del u_slice, v_slice, u_shear, v_shear, total_shear
                        
                    except Exception as e:
                        print(f"Error processing time step {time}: {str(e)}")
                        continue
                
                # Close datasets
                ds_u.close()
                ds_v.close()
                
                # Force garbage collection
                import gc
                gc.collect()
                
            except Exception as e:
                print(f"Error processing files {u_file.name} and {v_file.name}: {str(e)}")
                continue
        
        if not all_data:
            raise ValueError("No wind data could be processed successfully")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Sort by date and remove duplicates
        df = df.sort_values('Date').drop_duplicates(subset=['Date'])
        
        print("Wind data processing complete!")
        return df
    
    def _load_oni_data(self):
        """Load ONI (El Niño/La Niña) data"""
        try:
            print("Starting ONI data processing...")
            # Read the file directly first to inspect its content
            with open(self.data_dir / 'oni/oni_data.txt', 'r') as f:
                lines = f.readlines()[:5]
                print("First few lines of ONI file:")
                print(''.join(lines))
            
            oni_df = pd.read_csv(
                self.data_dir / 'oni/oni_data.txt',
                delim_whitespace=True,
                na_values=['***']
            )
            
            print("Column names found:", oni_df.columns.tolist())
            
            # Assuming we have SEAS, YR, and ANOM columns
            if 'SEAS' not in oni_df.columns or 'YR' not in oni_df.columns:
                raise ValueError("Required columns not found in ONI data")
            
            # Map seasons to months (using middle month of season)
            season_to_month = {
                'DJF': 1,  # January
                'JFM': 2,  # February
                'FMA': 3,  # March
                'MAM': 4,  # April
                'AMJ': 5,  # May
                'MJJ': 6,  # June
                'JJA': 7,  # July
                'JAS': 8,  # August
                'ASO': 9,  # September
                'SON': 10, # October
                'OND': 11, # November
                'NDJ': 12  # December
            }
            
            oni_df['Month'] = oni_df['SEAS'].map(season_to_month)
            
            # Create dates using the middle month
            oni_df['Date'] = pd.to_datetime(
                oni_df['YR'].astype(str) + '-' + 
                oni_df['Month'].astype(str) + '-01'
            )
            
            result = oni_df[['Date', 'ANOM']].rename(columns={'ANOM': 'oni'})
            print(f"Processed {len(result)} ONI records")
            return result
        
        except Exception as e:
            print(f"Error processing ONI data: {str(e)}")
            return pd.DataFrame(columns=['Date', 'oni'])

    def _load_amo_data(self):
        """Load AMO data"""
        try:
            print("Starting AMO data processing...")
            
            # Read the file content first
            with open(self.data_dir / 'amo/amo_data.txt', 'r') as f:
                lines = f.readlines()[:5]
                print("First few lines of AMO file:")
                print(''.join(lines))
            
            # Skip the first line (year range) and use fixed-width format
            amo_df = pd.read_fwf(
                self.data_dir / 'amo/amo_data.txt',
                skiprows=1,
                widths=[6] + [8] * 12,  # Adjust widths based on your file format
                na_values=['-99.99']
            )
            
            # Rename columns
            amo_df.columns = ['Year'] + [f'Month_{i}' for i in range(1, 13)]
            
            # Melt the dataframe
            amo_long = pd.melt(
                amo_df,
                id_vars=['Year'],
                var_name='Month',
                value_name='amo'
            )
            
            # Extract month number
            amo_long['Month'] = amo_long['Month'].str.extract('(\d+)').astype(int)
            
            # Create proper dates
            amo_long['Date'] = pd.to_datetime(
                amo_long['Year'].astype(str) + '-' + 
                amo_long['Month'].astype(str) + '-01'
            )
            
            result = amo_long[['Date', 'amo']].dropna()
            print(f"Processed {len(result)} AMO records")
            return result
            
        except Exception as e:
            print(f"Error processing AMO data: {str(e)}")
            return pd.DataFrame(columns=['Date', 'amo'])
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset with improved validation"""
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        print("\nHandling missing values...")
        print("Before cleaning:")
        print(df.isnull().sum())
        
        # Fill missing climate data with column medians first, then fallback values
        climate_cols = ['sst', 'wind_shear', 'oni', 'amo']
        for col in climate_cols:
            if col in df.columns:
                median_val = df[col].median()
                if pd.isna(median_val):  # If median is NaN, use default values
                    if col == 'sst':
                        df[col] = df[col].fillna(28.0)
                    elif col == 'wind_shear':
                        df[col] = df[col].fillna(10.0)
                    else:  # oni and amo
                        df[col] = df[col].fillna(0.0)
                else:
                    df[col] = df[col].fillna(median_val)
        
        # Handle coordinate data
        for col in ['Latitude', 'Longitude']:
            if col in df.columns:
                df[col] = df[col].interpolate(method='linear')
                # If any NaN remain, fill with mean
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].mean())
        
        # Handle wind radii with 0s
        wind_cols = [col for col in df.columns if col.startswith('Wind')]
        for col in wind_cols:
            df[col] = df[col].fillna(0)
        
        # Handle MaxWindRadius
        if 'MaxWindRadius' in df.columns:
            median_radius = df['MaxWindRadius'].median()
            if pd.isna(median_radius):
                df['MaxWindRadius'] = df['MaxWindRadius'].fillna(30)
            else:
                df['MaxWindRadius'] = df['MaxWindRadius'].fillna(median_radius)
        
        # Handle Pressure
        if 'Pressure' in df.columns:
            df['Pressure'] = df['Pressure'].interpolate(method='linear')
            if df['Pressure'].isnull().any():
                df['Pressure'] = df['Pressure'].fillna(df['Pressure'].median())
        
        # Fill RecordID with empty string
        if 'RecordID' in df.columns:
            df['RecordID'] = df['RecordID'].fillna('')
        
        print("\nAfter cleaning:")
        print(df.isnull().sum())
        print(f"\nRows before cleaning: {len(df)}")
        print(f"Rows after cleaning: {len(df)}")
        
        # Verify no NaN values remain in essential columns
        essential_cols = ['Date', 'Latitude', 'Longitude', 'Wind', 'Status']
        if df[essential_cols].isnull().any().any():
            raise ValueError("NaN values remain in essential columns after cleaning")
        
        return df
    
    def load_and_prepare_data(self, hurricane_data_path='filtered_hurricane_data.csv'):
        """Load and prepare data with validation"""
        memory_tracker = MemoryTracker()
        
        try:
            print("Loading hurricane data...")
            memory_tracker.print_memory_usage()
            hurricane_df = pd.read_csv(hurricane_data_path)
            hurricane_df['Date'] = pd.to_datetime(hurricane_df['Date'])
            
            print(f"Loaded {len(hurricane_df)} hurricane records")
            print("Date range:", hurricane_df['Date'].min(), "to", hurricane_df['Date'].max())
            
            # Create combined dataset starting with hurricane data
            combined_df = hurricane_df.copy()
            
            # Load and merge each dataset
            datasets = {
                'SST': self._load_sst_data,
                'Wind': self._load_wind_data,
                'ONI': self._load_oni_data,
                'AMO': self._load_amo_data
            }
            
            for name, load_func in datasets.items():
                try:
                    print(f"\nLoading {name} data...")
                    data = load_func()
                    print(f"{name} data shape: {data.shape}")
                    if len(data) > 0:
                        combined_df = combined_df.merge(data, on='Date', how='left')
                        print(f"After {name} merge: {len(combined_df)} records")
                    else:
                        print(f"No {name} data available, using default values")
                        if name == 'SST':
                            combined_df['sst'] = 28.0
                        elif name == 'Wind':
                            combined_df['wind_shear'] = 10.0
                        elif name == 'ONI':
                            combined_df['oni'] = 0.0
                        elif name == 'AMO':
                            combined_df['amo'] = 0.0
                except Exception as e:
                    print(f"Error processing {name} data: {str(e)}")
                    # Set default values for missing data
                    if name == 'SST':
                        combined_df['sst'] = 28.0
                    elif name == 'Wind':
                        combined_df['wind_shear'] = 10.0
                    elif name == 'ONI':
                        combined_df['oni'] = 0.0
                    elif name == 'AMO':
                        combined_df['amo'] = 0.0
            
            print("\nCreating risk categories...")
            combined_df['risk_category'] = pd.cut(
                combined_df['Wind'],
                bins=[0, 34, 64, 83, 96, 113, float('inf')],
                labels=['TD', 'TS', 'Cat1', 'Cat2', 'Cat3', 'Cat4-5']
            )
            
            print("\nHandling missing values...")
            combined_df = self._handle_missing_values(combined_df)
            
            if len(combined_df) == 0:
                raise ValueError("No data remained after merging and cleaning")
            
            print("\nCalculating weighted risk...")
            combined_df['weighted_risk'] = self.calculate_weighted_features(combined_df)
            
            print("\nFinal dataset summary:")
            print(f"Shape: {combined_df.shape}")
            print("Date range:", combined_df['Date'].min(), "to", combined_df['Date'].max())
            print("\nRisk category distribution:")
            print(combined_df['risk_category'].value_counts())
            
            return combined_df
        
        except Exception as e:
            print(f"Error in data preparation: {str(e)}")
            raise
    
    def train_model(self, data):
        """
        Train the SVM model with improved handling of imbalanced data and feature engineering
        """
        print("\nEngineering features...")
        # Add derived features
        data = self._engineer_features(data)
        
        # Prepare features
        feature_cols = [
            'sst', 'wind_shear', 'oni', 'amo', 'Latitude', 'Longitude',
            'weighted_risk', 'sst_anomaly', 'wind_pressure_ratio',
            'location_risk'
        ]
        
        X = data[feature_cols]
        y = data['risk_category']
        
        # Print NaN check before cleaning
        print("\nChecking for NaN values before cleaning:")
        print(X.isnull().sum())
        
        # Handle NaN values explicitly
        print("\nHandling missing values...")
        for col in X.columns:
            if X[col].isnull().any():
                if col in ['sst', 'sst_anomaly']:
                    X[col] = X[col].fillna(28.0)
                elif col == 'wind_shear':
                    X[col] = X[col].fillna(10.0)
                elif col == 'wind_pressure_ratio':
                    X[col] = X[col].fillna(X[col].mean())
                elif col == 'weighted_risk':
                    X[col] = X[col].fillna(0.5)
                else:
                    X[col] = X[col].fillna(0.0)
        
        # Verify no NaN values remain
        print("\nChecking for NaN values after cleaning:")
        print(X.isnull().sum())
        
        # Calculate class weights based on inverse frequency
        class_counts = y.value_counts()
        total_samples = len(y)
        class_weights = {
            cls: total_samples / (len(class_counts) * count) 
            for cls, count in class_counts.items()
        }
        
        print("\nClass weights:", class_weights)
        
        # Split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Simplified parameter grid to start
        param_grid = {
            'C': [1, 10],
            'gamma': ['scale'],
            'kernel': ['rbf'],
            'class_weight': ['balanced'],
            'decision_function_shape': ['ovr']
        }
        
        print("\nTraining SVM model with parameters:", param_grid)
        
        # Use SVC with probability estimates
        svm = SVC(probability=True)
        
        # Perform grid search with cross-validation and error handling
        grid_search = GridSearchCV(
            svm, 
            param_grid, 
            cv=5, 
            scoring='f1_weighted', 
            n_jobs=-1,
            verbose=1,
            error_score='raise'  # Raise errors for debugging
        )
        
        try:
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            print("\nBest parameters:", grid_search.best_params_)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            return self.model
            
        except Exception as e:
            print("\nError during model training:", str(e))
            print("\nFeature statistics:")
            print(X.describe())
            raise

    def _engineer_features(self, data):
        """
        Engineer additional features with careful NaN handling
        """
        print("Adding engineered features...")
        df = data.copy()
        
        # SST anomaly from typical hurricane-supporting temperatures
        df['sst_anomaly'] = df['sst'] - 26.5
        
        # Wind-Pressure relationship with error handling
        df['wind_pressure_ratio'] = np.where(
            df['Pressure'] > 0,
            df['Wind'] / df['Pressure'],
            np.nan
        )
        
        # Location-based risk
        df['location_risk'] = df.apply(
            lambda row: self._calculate_location_risk(row['Latitude'], row['Longitude']),
            axis=1
        )
        
        # Print feature statistics
        print("\nNew feature statistics:")
        print(df[['sst_anomaly', 'wind_pressure_ratio', 'location_risk']].describe())
        
        return df

    def _calculate_location_risk(self, lat, lon):
        """Calculate location-based risk based on historical hurricane patterns"""
        # Gulf of Mexico bounds
        gulf_bounds = {
            'lat': (23.5, 30.5),
            'lon': (-98.0, -80.0)
        }
        
        # Caribbean bounds
        caribbean_bounds = {
            'lat': (17.0, 23.5),
            'lon': (-88.0, -75.0)
        }
        
        try:
            if (gulf_bounds['lat'][0] <= lat <= gulf_bounds['lat'][1] and 
                gulf_bounds['lon'][0] <= lon <= gulf_bounds['lon'][1]):
                return 0.8
            elif (caribbean_bounds['lat'][0] <= lat <= caribbean_bounds['lat'][1] and 
                caribbean_bounds['lon'][0] <= lon <= caribbean_bounds['lon'][1]):
                return 0.6
            else:
                return 0.4
        except:
            return 0.4  # Default risk level if calculation fails
    
    def perform_spatial_correlation_analysis(self, data):
        """
        Perform spatial correlation analysis between climate factors and hurricane activity
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing hurricane and climate data
        """
        # Create grid of Gulf of Mexico region
        lat_grid = np.arange(18, 31, 1)  # 1-degree resolution
        lon_grid = np.arange(-98, -80, 1)
        
        # Initialize correlation matrices
        correlations = {
            'sst': np.zeros((len(lat_grid), len(lon_grid))),
            'wind_shear': np.zeros((len(lat_grid), len(lon_grid))),
            'oni': np.zeros((len(lat_grid), len(lon_grid))),
            'amo': np.zeros((len(lat_grid), len(lon_grid)))
        }
        
        # Calculate hurricane intensity at each grid point
        for i, lat in enumerate(lat_grid):
            for j, lon in enumerate(lon_grid):
                # Find hurricanes within radius of grid point
                radius = 300  # km
                nearby_storms = data[
                    (data['Latitude'] - lat)**2 + 
                    (data['Longitude'] - lon)**2 <= (radius/111)**2  # Convert km to degrees
                ]
                
                if len(nearby_storms) > 0:
                    # Calculate correlations with each climate factor
                    for factor in correlations.keys():
                        if factor in nearby_storms.columns:
                            correlation = np.corrcoef(
                                nearby_storms[factor],
                                nearby_storms['Wind']
                            )[0,1]
                            correlations[factor][i,j] = correlation
        
        # Plot correlation maps
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Spatial Correlations between Climate Factors and Hurricane Intensity')
        
        for (factor, corr_matrix), ax in zip(correlations.items(), axes.flat):
            im = ax.pcolormesh(
                lon_grid, lat_grid, corr_matrix,
                cmap='RdBu', vmin=-1, vmax=1
            )
            ax.set_title(f'{factor.upper()} Correlation')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig('spatial_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate and return overall correlation statistics
        correlation_stats = {
            factor: {
                'mean_correlation': np.nanmean(corr_matrix),
                'max_correlation': np.nanmax(corr_matrix),
                'min_correlation': np.nanmin(corr_matrix),
                'std_correlation': np.nanstd(corr_matrix)
            }
            for factor, corr_matrix in correlations.items()
        }
        
        return correlation_stats

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.model.classes_,
            yticklabels=self.model.classes_
        )
        plt.title('Hurricane Risk Classification Confusion Matrix')
        plt.ylabel('True Category')
        plt.xlabel('Predicted Category')
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, X, y):
        """Plot relative importance of each feature"""
        plt.figure(figsize=(10, 6))
        importance = pd.Series(self.weights).sort_values(ascending=True)
        importance.plot(kind='barh')
        plt.title('Feature Weights in Risk Assessment')
        plt.xlabel('Weight')
        plt.tight_layout()
        plt.show()
    
    def assess_location_risk(self, lat, lon, climate_conditions=None):
        """
        Assess hurricane risk for a specific location with weighted features
        """
        if self.model is None:
            raise ValueError("Model needs to be trained first")
        
        # Set default climate conditions if none provided
        if climate_conditions is None:
            climate_conditions = {
                'sst': 28.0,
                'wind_shear': 10.0,
                'oni': 0.0,
                'amo': 0.0
            }
        
        # Calculate derived features
        climate_conditions['sst_anomaly'] = climate_conditions['sst'] - 26.5
        climate_conditions['location_risk'] = self._calculate_location_risk(lat, lon)
        climate_conditions['wind_pressure_ratio'] = 0.05  # Default average value
        climate_conditions['weighted_risk'] = 0.5  # Default mid-range value
        
        # Create feature vector with all required features in correct order
        features_dict = {
            'sst': climate_conditions['sst'],
            'wind_shear': climate_conditions['wind_shear'],
            'oni': climate_conditions['oni'],
            'amo': climate_conditions['amo'],
            'Latitude': lat,
            'Longitude': lon,
            'weighted_risk': climate_conditions['weighted_risk'],
            'sst_anomaly': climate_conditions['sst_anomaly'],
            'wind_pressure_ratio': climate_conditions['wind_pressure_ratio'],
            'location_risk': climate_conditions['location_risk']
        }
        
        # Convert to DataFrame to ensure correct feature order
        features_df = pd.DataFrame([features_dict])
        
        # Scale features
        try:
            features_scaled = self.scaler.transform(features_df)
        except Exception as e:
            print(f"Error during feature scaling: {e}")
            return {
                'location': {'lat': lat, 'lon': lon},
                'risk_level': 'Unknown',
                'predicted_category': 'Unknown',
                'confidence_score': 0.0,
                'category_probabilities': {},
                'error': str(e)
            }
        
        # Get predictions and probabilities
        try:
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Determine risk level based on probabilities
            max_prob = np.max(probabilities)
            if max_prob >= 0.8:
                risk_level = 'Very High'
            elif max_prob >= 0.6:
                risk_level = 'High'
            elif max_prob >= 0.4:
                risk_level = 'Moderate'
            elif max_prob >= 0.2:
                risk_level = 'Low'
            else:
                risk_level = 'Very Low'
            
            # Create assessment dictionary
            assessment = {
                'location': {'lat': lat, 'lon': lon},
                'overall_risk_level': risk_level,
                'predicted_category': prediction,
                'confidence_score': float(max_prob),
                'category_probabilities': dict(zip(self.model.classes_, probabilities.tolist())),
                'contributing_factors': {
                    'SST': {
                        'value': climate_conditions['sst'],
                        'anomaly': climate_conditions['sst_anomaly'],
                        'impact': 'high' if climate_conditions['sst'] > 26.5 else 'moderate'
                    },
                    'Wind Shear': {
                        'value': climate_conditions['wind_shear'],
                        'impact': 'high' if climate_conditions['wind_shear'] < 20 else 'low'
                    },
                    'Location Risk': {
                        'value': climate_conditions['location_risk'],
                        'impact': 'high' if climate_conditions['location_risk'] > 0.6 else 'moderate'
                    }
                }
            }
            
            return assessment
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {
                'location': {'lat': lat, 'lon': lon},
                'overall_risk_level': 'Error',
                'predicted_category': 'Unknown',
                'confidence_score': 0.0,
                'category_probabilities': {},
                'error': str(e)
            }

def main():
    """Main function to test the risk assessor"""
    # Initialize and train the model
    assessor = WeightedHurricaneRiskAssessor()
    data = assessor.load_and_prepare_data()
    assessor.train_model(data)
    
    # Test risk assessment for New Orleans
    assessment = assessor.assess_location_risk(
        29.9511, -90.0715,
        climate_conditions={
            'sst': 28.5,
            'wind_shear': 8.0,
            'oni': 0.5,
            'amo': 0.2
        }
    )
    
    print("\nRisk Assessment Results:")
    print(f"Location: {assessment['location']}")
    print(f"Overall Risk Level: {assessment['overall_risk_level']}")
    print(f"Predicted Category: {assessment['predicted_category']}")
    print(f"Confidence Score: {assessment['confidence_score']:.2f}")
    print("\nCategory Probabilities:")
    for category, prob in assessment['category_probabilities'].items():
        print(f"  {category}: {prob:.2%}")
    print("\nContributing Factors:")
    for factor, details in assessment['contributing_factors'].items():
        print(f"  {factor}:")
        for key, value in details.items():
            print(f"    {key}: {value}")
            
if __name__ == "__main__":
    main()