import pandas as pd
import numpy as np

def load_and_process_data(filepath):
    """
    Loads the GDP CSV, cleans it, and reshapes it for analysis.
    """
    # Read CSV, skipping the first row which is often empty or a title in this dataset
    df = pd.read_csv(filepath, header=None)
    
    # Based on the view_file, row 1 (index 1) contains years "2007", "2008"...
    # Row 2 (index 2) starts with "AGRICULTURE..."
    
    # Let's extract years from the correct row (row 1 in 0-indexed pandas if we didn't skip, 
    # but let's assume standard read).
    # Re-reading with specific header behavior is safer:
    
    df = pd.read_csv(filepath)
    
    # The file has a complex structure. Let's manually reconstruct it based on the earlier 'view_file'.
    # Line 1: ,2007,2008...
    # Line 2: "AGRICULTURE...","1,226.10"...
    
    # Reload with header=0 to get years as columns, but the first column name is empty or unnamed
    df = pd.read_csv(filepath, header=0)
    
    # Rename the first column (sectors)
    df.rename(columns={df.columns[0]: 'Sector'}, inplace=True)
    
    # Identify relevant sector rows. 
    # We want "Mining & Quarrying", "Agriculture & Hunting", "Manufacturing", "Construction"
    # And potentially the Total GDP.
    
    target_sectors = {
        'Mining & Quarrying': 'Mining',
        'Agriculture & Hunting': 'Agriculture', 
        'Manufacturing': 'Manufacturing',
        'Construction': 'Construction',
        'GROSS DOMESTIC PRODUCT (GDP)': 'Total GDP'
    }
    
    # Filter rows
    df['Sector'] = df['Sector'].str.strip() # Remove indentation whitespace
    filtered_df = df[df['Sector'].isin(target_sectors.keys())].copy()
    
    # Remap names
    filtered_df['Sector'] = filtered_df['Sector'].map(target_sectors)
    
    # Clean numeric data (remove commas)
    # Get all year columns (columns 1 onwards)
    year_cols = filtered_df.columns[1:]
    
    for col in year_cols:
        filtered_df[col] = filtered_df[col].astype(str).str.replace(',', '').astype(float)
        
    # Melt for Plotly (Long Format)
    df_long = filtered_df.melt(id_vars=['Sector'], var_name='Year', value_name='GDP')
    
    # invalid literal for int() with base 10: '2014/p' fix
    # Remove non-numeric characters from Year (e.g. "2014/p" -> "2014")
    df_long['Year'] = df_long['Year'].astype(str).str.replace(r'\D+', '', regex=True).astype(int)
    
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
