import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from get_climate_data import download_all_climate_data
from weighted_hurricane_risk import WeightedHurricaneRiskAssessor
from kde_hurricane_risk import KDEHurricaneRiskAssessor

def load_cities():
    """Create dictionary of cities and their coordinates"""
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
    
    cities = {}
    for line in cities_data.strip().split('\n'):
        city, country, lat, lon = line.strip().split(',')
        city_name = f"{city.strip()} ({country.strip()})"
        cities[city_name] = (float(lat), float(lon))
    
    return cities

def analyze_hurricane_season(assessor, cities, year):
    """
    Analyze hurricane risk for entire hurricane season using SVM model
    """
    # Hurricane season months (June through November)
    season_months = range(6, 12)
    monthly_results = []
    
    for month in season_months:
        print(f"Analyzing {year}-{month:02d}...")
        results = []
        
        for city, (lat, lon) in cities.items():
            try:
                # Get risk assessment for each city
                assessment = assessor.assess_location_risk(
                    lat, lon,
                    climate_conditions={
                        'sst': 28.5,  # Default values if actual data not available
                        'wind_shear': 8.0,
                        'oni': 0.5,
                        'amo': 0.2
                    }
                )
                
                # Format results with error handling
                result = {
                    'city': city,
                    'latitude': lat,
                    'longitude': lon,
                    'risk_level': assessment.get('risk_level', 'Unknown'),
                    'predicted_category': assessment.get('predicted_category', 'Unknown'),
                    'confidence': assessment.get('confidence_score', 0.0),
                    'risk_score': assessment.get('confidence_score', 0.0)
                }
                
                # Add individual category probabilities if available
                probabilities = assessment.get('category_probabilities', {})
                for category, prob in probabilities.items():
                    result[f'prob_{category}'] = prob
                
                results.append(result)
                
            except Exception as e:
                print(f"Warning: Error analyzing {city}: {str(e)}")
                # Add placeholder result for failed analysis
                results.append({
                    'city': city,
                    'latitude': lat,
                    'longitude': lon,
                    'risk_level': 'Error',
                    'predicted_category': 'Unknown',
                    'confidence': 0.0,
                    'risk_score': 0.0
                })
        
        monthly_results.append(pd.DataFrame(results))
    
    # Calculate season average with error handling
    try:
        season_avg = pd.concat(monthly_results).groupby('city').mean().reset_index()
    except Exception as e:
        print(f"Warning: Error calculating season average: {str(e)}")
        # Return the last month's results if averaging fails
        season_avg = monthly_results[-1]
    
    return season_avg

def format_risk_report(results):
    """Format detailed risk assessment report"""
    report = ["Hurricane Risk Assessment Report", "=" * 30, ""]
    
    # Sort cities by risk score
    sorted_results = results.sort_values('risk_score', ascending=False)
    
    report.append("City Risk Rankings:")
    report.append("-" * 20)
    
    for _, row in sorted_results.iterrows():
        report.append(f"\nCity: {row['city']}")
        report.append(f"Risk Level: {row['risk_level']}")
        report.append(f"Predicted Category: {row['predicted_category']}")
        report.append(f"Confidence Score: {row['confidence']:.2f}")
        report.append(f"Overall Risk Score: {row['risk_score']:.2f}")
        
        # Add category probabilities if available
        prob_cols = [col for col in row.index if col.startswith('prob_')]
        if prob_cols:
            report.append("\nCategory Probabilities:")
            for col in prob_cols:
                category = col.replace('prob_', '')
                report.append(f"  {category}: {row[col]:.1%}")
        
        report.append("-" * 20)
    
    return "\n".join(report)

def format_correlation_report(correlation_stats):
    """Format spatial correlation analysis report"""
    report = [
        "Spatial Correlation Analysis",
        "=========================",
        ""
    ]
    
    for factor, stats in correlation_stats.items():
        report.extend([
            f"{factor.upper()} Correlations:",
            f"  Mean correlation: {stats['mean_correlation']:.3f}",
            f"  Maximum correlation: {stats['max_correlation']:.3f}",
            f"  Minimum correlation: {stats['min_correlation']:.3f}",
            f"  Standard deviation: {stats['std_correlation']:.3f}",
            ""
        ])
    
    return "\n".join(report)

def analyze_hurricane_season_combined(svm_assessor, kde_assessor, cities, year):
    """
    Analyze hurricane risk using both SVM and KDE approaches
    """
    # Hurricane season months (June through November)
    season_months = range(6, 12)
    monthly_results = []
    
    for month in season_months:
        print(f"Analyzing {year}-{month:02d}...")
        results = []
        
        for city, (lat, lon) in cities.items():
            try:
                # Get SVM assessment
                svm_assessment = svm_assessor.assess_location_risk(
                    lat, lon,
                    climate_conditions={
                        'sst': 28.5,
                        'wind_shear': 8.0,
                        'oni': 0.5,
                        'amo': 0.2
                    }
                )
                
                # Get KDE assessment
                kde_assessment = kde_assessor.assess_risk(lat, lon, season_month=month)
                
                # Combine results
                result = {
                    'city': city,
                    'latitude': lat,
                    'longitude': lon,
                    'svm_risk_level': svm_assessment.get('risk_level', 'Unknown'),
                    'kde_risk_level': kde_assessment['risk_level'],
                    'svm_confidence': svm_assessment.get('confidence_score', 0.0),
                    'kde_risk_score': kde_assessment['overall_risk_score'],
                    'combined_risk_score': (
                        0.5 * svm_assessment.get('confidence_score', 0.0) +
                        0.5 * kde_assessment['overall_risk_score']
                    )
                }
                
                # Add individual category probabilities from both models
                svm_probs = svm_assessment.get('category_probabilities', {})
                kde_probs = kde_assessment['category_probabilities']
                
                for category in set(list(svm_probs.keys()) + list(kde_probs.keys())):
                    result[f'svm_prob_{category}'] = svm_probs.get(category, 0.0)
                    result[f'kde_prob_{category}'] = kde_probs.get(category, 0.0)
                
                results.append(result)
                
            except Exception as e:
                print(f"Warning: Error analyzing {city}: {str(e)}")
                results.append({
                    'city': city,
                    'latitude': lat,
                    'longitude': lon,
                    'svm_risk_level': 'Error',
                    'kde_risk_level': 'Error',
                    'svm_confidence': 0.0,
                    'kde_risk_score': 0.0,
                    'combined_risk_score': 0.0
                })
        
        monthly_results.append(pd.DataFrame(results))
    
    # Calculate season average
    try:
        season_avg = pd.concat(monthly_results).groupby('city').mean().reset_index()
    except Exception as e:
        print(f"Warning: Error calculating season average: {str(e)}")
        season_avg = monthly_results[-1]
    
    return season_avg

def plot_combined_risk_map(risk_results, cities):
    """Create a map visualization showing both SVM and KDE risks"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot SVM risk
    scatter1 = ax1.scatter(
        [coords[1] for coords in cities.values()],
        [coords[0] for coords in cities.values()],
        c=risk_results['svm_confidence'],
        cmap='RdYlBu_r',
        s=100
    )
    ax1.set_title('SVM-based Risk Assessment')
    plt.colorbar(scatter1, ax=ax1, label='Risk Score (SVM)')
    
    # Plot KDE risk
    scatter2 = ax2.scatter(
        [coords[1] for coords in cities.values()],
        [coords[0] for coords in cities.values()],
        c=risk_results['kde_risk_score'],
        cmap='RdYlBu_r',
        s=100
    )
    ax2.set_title('KDE-based Risk Assessment')
    plt.colorbar(scatter2, ax=ax2, label='Risk Score (KDE)')
    
    # Add city labels to both plots
    for city, coords in cities.items():
        ax1.annotate(
            city, 
            (coords[1], coords[0]), 
            xytext=(5, 5),
            textcoords='offset points', 
            fontsize=8
        )
        ax2.annotate(
            city, 
            (coords[1], coords[0]), 
            xytext=(5, 5),
            textcoords='offset points', 
            fontsize=8
        )
    
    # Set labels
    for ax in (ax1, ax2):
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    
    plt.suptitle('Hurricane Risk Assessment Comparison')
    plt.tight_layout()
    
    # Save plot
    plt.savefig('hurricane_risk_comparison_map.png', dpi=300, bbox_inches='tight')
    plt.close()

def format_combined_risk_report(results):
    """Format detailed risk assessment report including both methods"""
    report = ["Hurricane Risk Assessment Report (SVM & KDE Combined)", "=" * 50, ""]
    
    # Sort cities by combined risk score
    sorted_results = results.sort_values('combined_risk_score', ascending=False)
    
    report.append("City Risk Rankings:")
    report.append("-" * 30)
    
    for _, row in sorted_results.iterrows():
        report.extend([
            f"\nCity: {row['city']}",
            f"Combined Risk Score: {row['combined_risk_score']:.2f}",
            "\nSVM Assessment:",
            f"  Risk Level: {row['svm_risk_level']}",
            f"  Confidence Score: {row['svm_confidence']:.2f}",
            "\nKDE Assessment:",
            f"  Risk Level: {row['kde_risk_level']}",
            f"  Risk Score: {row['kde_risk_score']:.2f}",
            "\nCategory Probabilities:"
        ])
        
        # Add category probabilities from both models
        svm_probs = {col.replace('svm_prob_', ''): row[col] 
                    for col in row.index if col.startswith('svm_prob_')}
        kde_probs = {col.replace('kde_prob_', ''): row[col] 
                    for col in row.index if col.startswith('kde_prob_')}
        
        for category in set(list(svm_probs.keys()) + list(kde_probs.keys())):
            report.append(
                f"  {category}:"
                f" SVM: {svm_probs.get(category, 0):.1%},"
                f" KDE: {kde_probs.get(category, 0):.1%}"
            )
        
        report.append("-" * 30)
    
    return "\n".join(report)

def main():
    # Create output directory
    output_dir = Path('hurricane_analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    # Download climate data for recent years
    print("Downloading climate data for 2018-2023...")
    download_all_climate_data(2018, 2023)
    
    # Load cities
    print("Loading city data...")
    cities = load_cities()
    
    # Initialize and train both assessors
    print("Initializing and training assessors...")
    svm_assessor = WeightedHurricaneRiskAssessor()
    kde_assessor = KDEHurricaneRiskAssessor()
    
    # Load and prepare data
    data = svm_assessor.load_and_prepare_data()
    
    # Train both models
    print("Training SVM model...")
    svm_assessor.train_model(data)
    
    print("Training KDE model...")
    kde_assessor.fit(data)
    
    # Analyze current season with both models
    current_year = 2023
    print(f"Analyzing {current_year} hurricane season...")
    season_results = analyze_hurricane_season_combined(
        svm_assessor, 
        kde_assessor, 
        cities, 
        current_year
    )
    
    # Generate and save combined report
    print("Generating combined risk report...")
    report = format_combined_risk_report(season_results)
    report_path = output_dir / 'hurricane_risk_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Create visualization comparing both methods
    print("Generating comparison risk maps...")
    plot_combined_risk_map(season_results, cities)
    
    # Plot KDE density map
    print("Generating KDE density map...")
    kde_assessor.plot_risk_density()
    plt.savefig(output_dir / 'kde_density_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results to CSV
    results_path = output_dir / 'hurricane_risk_analysis.csv'
    season_results.to_csv(results_path, index=False)
    
    print("\nAnalysis complete!")
    print(f"Report saved to: {report_path}")
    print(f"Detailed results saved to: {results_path}")
    print("Risk comparison map saved as: hurricane_risk_comparison_map.png")
    print("KDE density map saved as: kde_density_map.png")

if __name__ == "__main__":
    main()