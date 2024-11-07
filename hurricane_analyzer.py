import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler
import xarray as xr
from geopy.distance import geodesic
from datetime import datetime

class HurricaneRiskAnalyzer:
    def __init__(self):
        # Define factor weights
        self.weights = {
            'sst': 0.45,
            'enso': 0.25,
            'amo': 0.15,
            'wind': 0.10,
            'dust': 0.05
        }
        
        # Initialize scalers for normalization
        self.scalers = {factor: MinMaxScaler() for factor in self.weights.keys()}
        
    def load_hurricane_data(self, hurdat_df):
        """
        Load and preprocess HURDAT2 data
        """
        self.hurricane_data = hurdat_df.copy()
        # Convert wind speeds to normalized intensity (0-1)
        self.hurricane_data['normalized_intensity'] = self.hurricane_data['Wind'] / 185  # Max wind speed in dataset
        
    def calculate_sst_risk(self, location, sst_data):
        """
        Calculate risk based on sea surface temperature
        """
        lat, lon = location
        # Extract SST for hurricane season (June-November)
        seasonal_sst = sst_data.sel(
            time=sst_data.time.dt.month.isin([6,7,8,9,10,11])
        ).mean(dim='time')
        
        # Get SST at location
        location_sst = seasonal_sst.sel(lat=lat, lon=lon, method='nearest').values
        
        # Normalize SST (higher SST = higher risk)
        return self.scalers['sst'].fit_transform([[location_sst]])[0][0]
    
    def calculate_enso_risk(self, oni_data):
        """
        Calculate risk based on El Niño/La Niña patterns
        """
        # Calculate average ONI during hurricane season
        seasonal_oni = oni_data[oni_data.index.month.isin([6,7,8,9,10,11])].mean()
        
        # Convert to risk (negative ONI/La Niña = higher risk)
        return self.scalers['enso'].fit_transform([[-seasonal_oni]])[0][0]
    
    def calculate_trajectory_density(self, location, radius_km=500):
        """
        Calculate hurricane trajectory density near location
        """
        lat, lon = location
        
        # Filter trajectories within radius
        nearby_storms = []
        for _, storm in self.hurricane_data.groupby('Date'):
            if geodesic((lat, lon), (storm['Latitude'].iloc[0], storm['Longitude'].iloc[0])).km <= radius_km:
                nearby_storms.append(storm)
        
        if not nearby_storms:
            return 0
            
        # Calculate kernel density estimation
        positions = np.vstack([storm['Latitude'].values for storm in nearby_storms])
        kernel = gaussian_kde(positions)
        
        return kernel.evaluate(np.array([lat]))[0]
    
    def calculate_historical_risk(self, location):
        """
        Calculate risk based on historical hurricane tracks
        """
        density = self.calculate_trajectory_density(location)
        return self.scalers['historical'].fit_transform([[density]])[0][0]
    
    def calculate_combined_risk(self, location, climate_data):
        """
        Calculate combined risk score using all factors
        """
        # Calculate individual risk components
        sst_risk = self.calculate_sst_risk(location, climate_data['sst'])
        enso_risk = self.calculate_enso_risk(climate_data['oni'])
        amo_risk = climate_data['amo']  # Assuming normalized AMO value
        wind_risk = climate_data['wind']  # Assuming normalized wind pattern risk
        dust_risk = climate_data['dust']  # Assuming normalized dust level risk
        
        # Calculate weighted sum
        total_risk = (
            sst_risk * self.weights['sst'] +
            enso_risk * self.weights['enso'] +
            amo_risk * self.weights['amo'] +
            wind_risk * self.weights['wind'] +
            dust_risk * self.weights['dust']
        )
        
        return total_risk
    
    def assess_city_risks(self, cities, climate_data):
        """
        Assess hurricane risks for multiple cities
        
        Parameters:
        -----------
        cities : dict
            Dictionary of city names and their (lat, lon) coordinates
        climate_data : dict
            Dictionary containing climate factor data
            
        Returns:
        --------
        DataFrame with risk scores for each city
        """
        results = []
        
        for city, coords in cities.items():
            # Calculate risks using both methods
            climate_risk = self.calculate_combined_risk(coords, climate_data)
            historical_risk = self.calculate_historical_risk(coords)
            
            # Average the two risk assessment methods
            combined_risk = (climate_risk + historical_risk) / 2
            
            results.append({
                'city': city,
                'latitude': coords[0],
                'longitude': coords[1],
                'climate_risk': climate_risk,
                'historical_risk': historical_risk,
                'combined_risk': combined_risk
            })
        
        return pd.DataFrame(results).sort_values('combined_risk', ascending=False)

def format_risk_report(risk_df):
    """
    Format risk assessment results into a readable report
    """
    report = "Hurricane Risk Assessment Report\n"
    report += "=" * 30 + "\n\n"
    
    # Add risk categories
    risk_df['risk_category'] = pd.qcut(risk_df['combined_risk'], 
                                     q=5, 
                                     labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
    
    # Group cities by risk category
    for category in ['Very High', 'High', 'Moderate', 'Low', 'Very Low']:
        cities = risk_df[risk_df['risk_category'] == category]
        report += f"{category} Risk Cities:\n"
        report += "-" * 20 + "\n"
        
        for _, city in cities.iterrows():
            report += f"{city['city']}: {city['combined_risk']:.3f}\n"
        report += "\n"
    
    return report