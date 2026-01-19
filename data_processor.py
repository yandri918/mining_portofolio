import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def load_and_process_data(filepath):
    """
    Loads the GDP CSV, cleans it, and reshapes it for analysis.
    """
    df = pd.read_csv(filepath, header=0)
    
    # Rename the first column (sectors)
    df.rename(columns={df.columns[0]: 'Sector'}, inplace=True)
    
    # Clean sector names
    df['Sector'] = df['Sector'].str.strip() # Remove indentation whitespace
    df['Sector'] = df['Sector'].str.replace('&amp;', '&', regex=False) # Fix HTML encoded ampersands
    
    target_sectors = {
        'Mining & Quarrying': 'Mining',
        'Agriculture & Hunting': 'Agriculture', 
        'Manufacturing': 'Manufacturing',
        'Construction': 'Construction',
        'GROSS DOMESTIC PRODUCT (GDP)': 'Total GDP'
    }
    
    # Filter rows
    filtered_df = df[df['Sector'].isin(target_sectors.keys())].copy()
    
    # Remap names
    filtered_df['Sector'] = filtered_df['Sector'].map(target_sectors)
    
    # Clean numeric data (remove commas)
    year_cols = filtered_df.columns[1:]
    
    for col in year_cols:
        filtered_df[col] = filtered_df[col].astype(str).str.replace(',', '').astype(float)
        
    # Melt for Plotly (Long Format)
    df_long = filtered_df.melt(id_vars=['Sector'], var_name='Year', value_name='GDP')
    
    # Remove non-numeric characters from Year (e.g. "2014/p" -> "2014")
    df_long['Year'] = df_long['Year'].astype(str).str.replace(r'\D+', '', regex=True).astype(int)
    
    # Drop any rows where GDP parsing failed (NaNs)
    df_long = df_long.dropna(subset=['GDP'])
    
    return df_long

def get_mining_stats(df_long):
    """Calculates specific stats for the Mining sector."""
    mining_df = df_long[df_long['Sector'] == 'Mining'].sort_values('Year')
    
    if mining_df.empty:
        return None
        
    start_val = mining_df.iloc[0]['GDP']
    end_val = mining_df.iloc[-1]['GDP']
    
    total_growth = ((end_val - start_val) / start_val) * 100
    
    # CAGR
    n_years = mining_df['Year'].max() - mining_df['Year'].min()
    cagr = ((end_val / start_val) ** (1/n_years) - 1) * 100
    
    return {
        'current_gdp': end_val,
        'total_growth_pct': total_growth,
        'cagr_pct': cagr
    }

def calculate_sector_contribution(df):
    """Calculate sector contribution to total GDP"""
    result = []
    for year in df['Year'].unique():
        year_data = df[df['Year'] == year]
        total_gdp = year_data[year_data['Sector'] == 'Total GDP']['GDP'].values
        
        if len(total_gdp) == 0:
            continue
            
        total_gdp = total_gdp[0]
        
        for sector in ['Mining', 'Agriculture', 'Manufacturing', 'Construction']:
            sector_gdp = year_data[year_data['Sector'] == sector]['GDP'].values
            if len(sector_gdp) > 0:
                contribution_pct = (sector_gdp[0] / total_gdp) * 100
                
                result.append({
                    'Year': year,
                    'Sector': sector,
                    'GDP': sector_gdp[0],
                    'Contribution_%': contribution_pct
                })
    
    return pd.DataFrame(result)

def calculate_volatility_metrics(df, sector_name):
    """Calculate volatility metrics for a specific sector"""
    sector_data = df[df['Sector'] == sector_name].sort_values('Year')
    
    if len(sector_data) == 0:
        return None
    
    gdp_values = sector_data['GDP'].values
    
    # Standard deviation
    std_dev = np.std(gdp_values)
    
    # Coefficient of variation (CV)
    mean_gdp = np.mean(gdp_values)
    cv = (std_dev / mean_gdp) * 100 if mean_gdp > 0 else 0
    
    # Peak and trough
    peak_value = np.max(gdp_values)
    trough_value = np.min(gdp_values)
    peak_idx = np.argmax(gdp_values)
    trough_idx = np.argmin(gdp_values)
    peak_year = sector_data.iloc[peak_idx]['Year']
    trough_year = sector_data.iloc[trough_idx]['Year']
    
    # Peak-to-trough drop (if peak comes before trough)
    if peak_idx < trough_idx:
        drop_pct = ((trough_value - peak_value) / peak_value) * 100
    else:
        drop_pct = 0
    
    return {
        'std_dev': std_dev,
        'cv': cv,
        'peak_value': peak_value,
        'peak_year': int(peak_year),
        'trough_value': trough_value,
        'trough_year': int(trough_year),
        'drop_pct': drop_pct,
        'mean_gdp': mean_gdp
    }

def decompose_growth(df, sector_name):
    """Decompose growth into year-over-year contributions"""
    sector_data = df[df['Sector'] == sector_name].sort_values('Year')
    
    if len(sector_data) < 2:
        return None
    
    result = []
    for i in range(1, len(sector_data)):
        prev_gdp = sector_data.iloc[i-1]['GDP']
        curr_gdp = sector_data.iloc[i]['GDP']
        year = sector_data.iloc[i]['Year']
        
        absolute_growth = curr_gdp - prev_gdp
        pct_growth = ((curr_gdp - prev_gdp) / prev_gdp) * 100 if prev_gdp > 0 else 0
        
        result.append({
            'Year': year,
            'Absolute_Growth': absolute_growth,
            'Pct_Growth': pct_growth,
            'GDP': curr_gdp
        })
    
    return pd.DataFrame(result)

def calculate_correlation_matrix(df):
    """Calculate correlation matrix between major sectors"""
    # Pivot to wide format
    pivot_df = df.pivot_table(values='GDP', index='Year', columns='Sector')
    
    # Select major sectors only
    major_sectors = ['Mining', 'Agriculture', 'Manufacturing', 'Construction']
    
    available_sectors = [s for s in major_sectors if s in pivot_df.columns]
    
    if len(available_sectors) < 2:
        return None
    
    corr_matrix = pivot_df[available_sectors].corr()
    
    return corr_matrix

def generate_forecast(df, sector_name, periods=5):
    """Generate multi-method forecast for a sector"""
    sector_data = df[df['Sector'] == sector_name].sort_values('Year')
    
    if len(sector_data) < 3:
        return None
    
    X = sector_data['Year'].values.reshape(-1, 1)
    y = sector_data['GDP'].values
    
    # Linear Regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Future years
    last_year = sector_data['Year'].max()
    future_years = np.arange(last_year + 1, last_year + periods + 1).reshape(-1, 1)
    
    # Predictions
    linear_pred = model.predict(future_years)
    
    # Exponential smoothing (simple)
    alpha = 0.3
    exp_smooth = [y[-1]]
    for i in range(periods):
        next_val = alpha * exp_smooth[-1] + (1 - alpha) * y[-1]
        exp_smooth.append(next_val)
    exp_smooth = exp_smooth[1:]
    
    # Confidence interval (±20% for demonstration)
    ci_upper = linear_pred * 1.2
    ci_lower = linear_pred * 0.8
    
    result = pd.DataFrame({
        'Year': future_years.flatten(),
        'Linear_Forecast': linear_pred,
        'Exp_Smooth_Forecast': exp_smooth,
        'CI_Upper': ci_upper,
        'CI_Lower': ci_lower
    })
    
    return result

def calculate_multiplier_effect(df):
    """Calculate mining sector multiplier effect on total GDP"""
    mining_data = df[df['Sector'] == 'Mining'].sort_values('Year')
    gdp_data = df[df['Sector'] == 'Total GDP'].sort_values('Year')
    
    if len(mining_data) < 2 or len(gdp_data) < 2:
        return None
    
    # Calculate changes
    mining_change = mining_data['GDP'].diff().dropna()
    gdp_change = gdp_data['GDP'].diff().dropna()
    
    # Multiplier = Change in Total GDP / Change in Mining GDP
    multipliers = gdp_change.values / mining_change.values
    
    # Remove infinite and NaN values
    multipliers = multipliers[np.isfinite(multipliers)]
    
    avg_multiplier = np.mean(multipliers) if len(multipliers) > 0 else 0
    
    return {
        'avg_multiplier': avg_multiplier,
        'multipliers': multipliers.tolist()
    }

def generate_policy_recommendations(volatility_metrics, contribution_data):
    """Generate policy recommendations based on analysis"""
    recommendations = []
    
    if volatility_metrics is None or contribution_data is None:
        return recommendations
    
    # High volatility warning
    if volatility_metrics['cv'] > 100:
        recommendations.append({
            'type': 'warning',
            'title': 'High Volatility Risk',
            'message': f"Mining sector shows extreme volatility (CV: {volatility_metrics['cv']:.1f}%). Recommend economic diversification."
        })
    
    # Boom-bust cycle
    if abs(volatility_metrics['drop_pct']) > 50:
        recommendations.append({
            'type': 'alert',
            'title': 'Boom-Bust Cycle Detected',
            'message': f"Sector experienced {abs(volatility_metrics['drop_pct']):.1f}% drop from peak. Implement stabilization policies."
        })
    
    # Low contribution
    mining_contrib = contribution_data[contribution_data['Sector'] == 'Mining']
    if len(mining_contrib) > 0:
        latest_contrib = mining_contrib.iloc[-1]['Contribution_%']
        if latest_contrib < 5:
            recommendations.append({
                'type': 'info',
                'title': 'Low GDP Contribution',
                'message': f"Mining contributes only {latest_contrib:.1f}% to GDP. Consider sector development strategies."
            })
    
    return recommendations
