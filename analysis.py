import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from get_climate_data import download_all_climate_data
from hurricane_analyzer import HurricaneRiskAnalyzer, format_risk_report

def load_cities():
    """
    Create dictionary of cities and their coordinates
    """
    cities_data = """
    New Orleans, USA,29.951065,-90.071533
    Houston, USA,29.760427,-95.369804
    Tampa, USA,27.950575,-82.457176
    Miami, USA,25.761681,-80.191788
    Corpus Christi, USA,27.800583,-97.396378
    Pensacola, USA,30.421309,-87.216911
    Mobile, USA,37.358402,-94.723251
    Galveston, USA,29.301348,-94.797699
    Biloxi, USA,30.396032,-88.885307
    Key West, USA,24.555059,-81.779984
    Veracruz, Mexico,19.173773,-96.134224
    Tampico, Mexico,22.247049,-97.861992
    Campeche, Mexico,18.931225,-90.26181
    Cancún, Mexico,21.097589,-86.85479
    Mérida, Mexico,20.967779,-89.62426
    Ciudad del Carmen, Mexico,18.649679,-91.822121
    Progreso, Mexico,21.28208,-89.663307
    Coatzacoalcos, Mexico,18.13566,-94.440224
    Tuxpan, Mexico,20.96134,-97.400887
    Havana, Cuba,23.113592,-82.366592
    Varadero, Cuba,23.134581,-81.283653
    Cienfuegos, Cuba,22.144449,-80.440292
    Belize City, Belize,17.49469,-88.189728
    George Town, Cayman Islands,19.300249,-81.375999
    Nassau, Bahamas,25.047983,-77.355415
    """
    
    # Convert to dictionary
    cities = {}
    for line in cities_data.strip().split('\n'):
        city, country, lat, lon = line.strip().split(',')
        city_name = f"{city.strip()} ({country.strip()})"
        cities[city_name] = (float(lat), float(lon))
    
    return cities

def analyze_hurricane_season(analyzer, cities, year):
    """
    Analyze hurricane risk for entire hurricane season
    
    Parameters:
    -----------
    analyzer : HurricaneRiskAnalyzer
        Initialized analyzer object
    cities : dict
        Dictionary of city names and coordinates
    year : int
        Year to analyze
    
    Returns:
    --------
    DataFrame with average risk scores for the season
    """
    # Hurricane season months (June through November)
    season_months = range(6, 12)
    monthly_results = []
    
    for month in season_months:
        # Analyze mid-month conditions
        date = np.datetime64(f'{year}-{month:02d}-15')
        monthly_risk = analyzer.assess_city_risks(cities, date)
        monthly_results.append(monthly_risk.set_index('city'))
    
    # Calculate season average
    season_avg = pd.concat(monthly_results).groupby('city').mean()
    return season_avg.reset_index()

def plot_risk_map(risk_results, cities):
    """
    Create a map visualization of hurricane risks
    """
    plt.figure(figsize=(15, 10))
    
    # Create scatter plot of cities colored by risk
    plt.scatter(
        [coords[1] for coords in cities.values()],
        [coords[0] for coords in cities.values()],
        c=risk_results['risk_score'],
        cmap='RdYlBu_r',
        s=100
    )
    
    # Add city labels
    for city, coords in cities.items():
        plt.annotate(city, (coords[1], coords[0]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    plt.colorbar(label='Risk Score')
    plt.title('Hurricane Risk Assessment Map')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Save plot
    plt.savefig('hurricane_risk_map.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create output directory
    output_dir = Path('hurricane_analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    # Download climate data if needed
    print("Checking and downloading climate data...")
    download_all_climate_data(1999, 2023)
    
    # Load cities
    print("Loading city data...")
    cities = load_cities()
    
    # Initialize analyzer
    print("Initializing analyzer...")
    analyzer = HurricaneRiskAnalyzer()
    
    # Analyze current hurricane season
    current_year = datetime.now().year
    print(f"Analyzing {current_year} hurricane season...")
    
    # Load climate data for analysis year
    analyzer.load_climate_data(current_year)
    
    # Analyze entire season
    season_results = analyze_hurricane_season(analyzer, cities, current_year)
    
    # Generate and save report
    report = format_risk_report(season_results)
    report_path = output_dir / 'hurricane_risk_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Create visualization
    print("Generating risk map...")
    plot_risk_map(season_results, cities)
    
    # Save detailed results to CSV
    results_path = output_dir / 'hurricane_risk_analysis.csv'
    season_results.to_csv(results_path, index=False)
    
    print("\nAnalysis complete!")
    print(f"Report saved to: {report_path}")
    print(f"Detailed results saved to: {results_path}")
    print("Risk map saved as: hurricane_risk_map.png")

if __name__ == "__main__":
    main()