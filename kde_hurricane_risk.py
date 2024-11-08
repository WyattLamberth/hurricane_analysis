from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

class KDEHurricaneRiskAssessor:
    def __init__(self, bandwidth_method='scott'):
        """
        Initialize KDE-based hurricane risk assessor
        
        Parameters:
        -----------
        bandwidth_method : str
            Method for bandwidth selection ('scott' or 'silverman')
        """
        self.bandwidth_method = bandwidth_method
        self.spatial_kde = None
        self.intensity_kde = None
        self.wind_range = None
        self.location_weights = None
    
    def fit(self, hurricane_data):
        """
        Fit KDE models to hurricane data
        
        Parameters:
        -----------
        hurricane_data : pd.DataFrame
            DataFrame containing hurricane tracks with columns:
            'Latitude', 'Longitude', 'Wind'
        """
        # Filter out missing or invalid data
        valid_data = hurricane_data.dropna(subset=['Latitude', 'Longitude', 'Wind'])
        
        # Prepare spatial data (lat, lon points)
        spatial_data = np.vstack([
            valid_data['Latitude'].values,
            valid_data['Longitude'].values
        ])
        
        # Prepare intensity data (wind speeds)
        intensity_data = valid_data['Wind'].values.reshape(1, -1)
        
        # Store wind speed range for normalization
        self.wind_range = (valid_data['Wind'].min(), valid_data['Wind'].max())
        
        # Fit spatial KDE
        self.spatial_kde = gaussian_kde(
            spatial_data,
            bw_method=self.bandwidth_method
        )
        
        # Fit intensity KDE
        self.intensity_kde = gaussian_kde(
            intensity_data,
            bw_method=self.bandwidth_method
        )
        
        # Calculate location weights based on historical hurricane frequency
        self._calculate_location_weights(valid_data)
        
        return self
    
    def _calculate_location_weights(self, data, grid_size=1.0):
        """
        Calculate location weights based on historical hurricane frequency
        """
        # Create grid covering Gulf of Mexico region
        lat_grid = np.arange(17.0, 31.5, grid_size)
        lon_grid = np.arange(-98.0, -80.0, grid_size)
        
        # Count hurricanes in each grid cell
        weights = np.zeros((len(lat_grid), len(lon_grid)))
        
        for i, lat in enumerate(lat_grid):
            for j, lon in enumerate(lon_grid):
                # Count storms within grid cell
                storms_in_cell = data[
                    (data['Latitude'] >= lat) & 
                    (data['Latitude'] < lat + grid_size) &
                    (data['Longitude'] >= lon) &
                    (data['Longitude'] < lon + grid_size)
                ]
                weights[i, j] = len(storms_in_cell)
        
        # Normalize weights
        weights = weights / np.max(weights)
        
        self.location_weights = {
            'grid': (lat_grid, lon_grid),
            'weights': weights
        }
    
    def _get_location_weight(self, lat, lon):
        """Get weight for specific location"""
        if self.location_weights is None:
            return 1.0
            
        lat_grid, lon_grid = self.location_weights['grid']
        weights = self.location_weights['weights']
        
        # Find nearest grid cell
        lat_idx = np.abs(lat_grid - lat).argmin()
        lon_idx = np.abs(lon_grid - lon).argmin()
        
        return weights[lat_idx, lon_idx]
    
    def assess_risk(self, lat, lon, season_month=None):
        """
        Assess hurricane risk for a specific location
        """
        if self.spatial_kde is None:
            raise ValueError("Model needs to be fitted first")
        
        try:
            # Calculate spatial density
            coords = np.array([[lat], [lon]])
            spatial_density = float(self.spatial_kde(coords)[0])
            
            # Get location weight
            location_weight = float(self._get_location_weight(lat, lon))
            
            # Calculate seasonal factor
            if season_month is not None:
                seasonal_factor = float(self._calculate_seasonal_factor(season_month))
            else:
                seasonal_factor = 1.0
            
            # Generate intensity distribution
            wind_speeds = np.linspace(
                self.wind_range[0],
                self.wind_range[1],
                100
            )
            intensity_distribution = self.intensity_kde(wind_speeds.reshape(1, -1)).ravel()
            
            # Calculate probability of different hurricane categories
            category_probs = self._calculate_category_probabilities(wind_speeds, intensity_distribution)
            
            # Calculate overall risk score (0-1)
            overall_risk = float(
                0.4 * spatial_density / self.spatial_kde([[25], [-85]])[0] +
                0.3 * location_weight +
                0.3 * seasonal_factor
            )
            
            # Ensure overall_risk is between 0 and 1
            overall_risk = max(0.0, min(1.0, overall_risk))
            
            # Determine risk level
            risk_level = self._determine_risk_level(overall_risk)
            
            return {
                'location': {'lat': float(lat), 'lon': float(lon)},
                'spatial_density': float(spatial_density),
                'intensity_distribution': {
                    'wind_speeds': wind_speeds.tolist(),
                    'densities': intensity_distribution.tolist()
                },
                'category_probabilities': category_probs,
                'seasonal_factor': float(seasonal_factor),
                'location_weight': float(location_weight),
                'overall_risk_score': float(overall_risk),
                'risk_level': risk_level
            }
            
        except Exception as e:
            print(f"Error in assess_risk for location ({lat}, {lon}): {str(e)}")
            return {
                'location': {'lat': float(lat), 'lon': float(lon)},
                'spatial_density': 0.0,
                'intensity_distribution': {
                    'wind_speeds': [],
                    'densities': []
                },
                'category_probabilities': self._get_default_category_probs(),
                'seasonal_factor': 0.0,
                'location_weight': 0.0,
                'overall_risk_score': 0.0,
                'risk_level': 'Error'
            }
    
    def _get_default_category_probs(self):
        """Return default category probabilities when calculation fails"""
        return {
            'TD': 0.0,
            'TS': 0.0,
            'Cat1': 0.0,
            'Cat2': 0.0,
            'Cat3': 0.0,
            'Cat4': 0.0,
            'Cat5': 0.0
        }

    def _calculate_seasonal_factor(self, month):
        """Calculate risk factor based on month of hurricane season"""
        # Peak season weights (based on historical Atlantic hurricane frequency)
        seasonal_weights = {
            1: 0.1,  # January
            2: 0.1,  # February
            3: 0.1,  # March
            4: 0.1,  # April
            5: 0.2,  # May
            6: 0.4,  # June
            7: 0.6,  # July
            8: 0.9,  # August
            9: 1.0,  # September
            10: 0.8, # October
            11: 0.4, # November
            12: 0.2  # December
        }
        return seasonal_weights.get(month, 0.1)
    
    def _calculate_category_probabilities(self, wind_speeds, densities):
        """Calculate probabilities for each hurricane category"""
        # Saffir-Simpson scale boundaries
        categories = {
            'TD': (0, 34),
            'TS': (34, 64),
            'Cat1': (64, 83),
            'Cat2': (83, 96),
            'Cat3': (96, 113),
            'Cat4': (113, 137),
            'Cat5': (137, float('inf'))
        }
        
        # Calculate probability for each category
        probs = {}
        total_density = np.sum(densities)
        
        for category, (lower, upper) in categories.items():
            mask = (wind_speeds >= lower) & (wind_speeds < upper)
            prob = np.sum(densities[mask]) / total_density if total_density > 0 else 0
            probs[category] = float(prob)
        
        return probs
    
    def _determine_risk_level(self, risk_score):
        """Convert risk score to categorical risk level"""
        if risk_score >= 0.8:
            return 'Very High'
        elif risk_score >= 0.6:
            return 'High'
        elif risk_score >= 0.4:
            return 'Moderate'
        elif risk_score >= 0.2:
            return 'Low'
        else:
            return 'Very Low'
    
    def plot_risk_density(self, lat_range=(17, 31), lon_range=(-98, -80), resolution=0.5):
        """
        Plot spatial risk density map
        """
        if self.spatial_kde is None:
            raise ValueError("Model needs to be fitted first")
            
        # Create grid
        lats = np.arange(lat_range[0], lat_range[1], resolution)
        lons = np.arange(lon_range[0], lon_range[1], resolution)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Calculate density for each point
        positions = np.vstack([lat_grid.ravel(), lon_grid.ravel()])
        density = self.spatial_kde.evaluate(positions)
        density = density.reshape(lon_grid.shape)
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.contourf(lon_grid, lat_grid, density, levels=20, cmap='YlOrRd')
        plt.colorbar(label='Hurricane Track Density')
        plt.title('Hurricane Risk Density Map (KDE-based)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        return plt.gcf()