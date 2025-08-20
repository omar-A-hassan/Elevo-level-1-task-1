import pandas as pd
import numpy as np
import os

def prepare_walmart_for_ts2vec():
    """
    Correct preprocessing to make Walmart data compatible with TS2Vec's forecast_csv loader.
    
    TS2Vec expects:
    - CSV with 'date' column as index
    - Time series data in columns (wide format)
    - Each column = different variable/time series
    """
    
    print("Loading Walmart sales data...")
    
    # Load datasets
    features = pd.read_csv('features.csv')
    stores = pd.read_csv('stores.csv')
    sales = pd.read_csv('train 2.csv')
    
    # Convert dates
    features['Date'] = pd.to_datetime(features['Date'])
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    print(f"Loaded {len(sales)} sales records, {len(features)} feature records")
    
    # Find common date range (sales data ends earlier than features)
    sales_dates = set(sales['Date'].unique())
    feature_dates = set(features['Date'].unique())
    common_dates = sorted(sales_dates.intersection(feature_dates))
    
    print(f"Common date range: {common_dates[0]} to {common_dates[-1]} ({len(common_dates)} weeks)")
    
    # Filter to common dates
    sales = sales[sales['Date'].isin(common_dates)]
    features = features[features['Date'].isin(common_dates)]
    
    # ==========================================
    # OPTION 1: Store-level aggregated sales (RECOMMENDED)
    # ==========================================
    print("\nCreating store-level time series...")
    
    # Aggregate sales by store and date (sum across departments)
    store_sales = sales.groupby(['Store', 'Date'])['Weekly_Sales'].sum().reset_index()
    
    # Create wide format: dates as rows, stores as columns
    sales_pivot = store_sales.pivot(index='Date', columns='Store', values='Weekly_Sales')
    
    # Rename columns to have 'Store_' prefix for clarity
    sales_pivot.columns = [f'Store_{col}' for col in sales_pivot.columns]
    
    # Reset index to make 'Date' a column and rename to 'date' (required by TS2Vec)
    sales_wide = sales_pivot.reset_index().rename(columns={'Date': 'date'})
    
    # Handle any missing values (forward fill then backward fill)
    for col in sales_wide.columns:
        if col != 'date':
            sales_wide[col] = sales_wide[col].fillna(method='ffill').fillna(method='bfill')
    
    print(f"Store-level data shape: {sales_wide.shape} (rows=dates, cols=date+stores)")
    
    # Save for univariate forecasting (multiple stores)
    os.makedirs('datasets', exist_ok=True)
    output_file = 'datasets/walmart_stores.csv'
    sales_wide.to_csv(output_file, index=False)
    print(f"✓ Saved store-level data to {output_file}")
    
    # ==========================================
    # OPTION 2: Single store multivariate (with features)
    # ==========================================
    print("\nCreating multivariate time series for Store 1...")
    
    # Get Store 1 data with features
    store1_sales = sales[sales['Store'] == 1].groupby('Date')['Weekly_Sales'].sum().reset_index()
    store1_features = features[features['Store'] == 1].copy()
    
    # Merge sales with features for Store 1
    store1_data = store1_sales.merge(store1_features, left_on='Date', right_on='Date', how='inner')
    
    # Handle markdown columns (convert "NA" to 0)
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    for col in markdown_cols:
        store1_data[col] = pd.to_numeric(store1_data[col], errors='coerce').fillna(0)
    
    # Convert boolean to int
    store1_data['IsHoliday'] = store1_data['IsHoliday'].astype(int)
    
    # Add store characteristics
    store1_info = stores[stores['Store'] == 1].iloc[0]
    store1_data['Size'] = store1_info['Size']
    store1_data['Type_A'] = 1 if store1_info['Type'] == 'A' else 0
    store1_data['Type_B'] = 1 if store1_info['Type'] == 'B' else 0
    store1_data['Type_C'] = 1 if store1_info['Type'] == 'C' else 0
    
    # Select features for multivariate model
    feature_columns = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size'] + \
                     markdown_cols + ['Type_A', 'Type_B', 'Type_C', 'IsHoliday']
    
    # Create final multivariate dataset
    multivar_data = store1_data[['Date'] + feature_columns].copy()
    multivar_data = multivar_data.rename(columns={'Date': 'date'})
    
    print(f"Multivariate data shape: {multivar_data.shape}")
    print(f"Features: {feature_columns}")
    
    # Save multivariate version
    multivar_file = 'datasets/walmart_multivar.csv'
    multivar_data.to_csv(multivar_file, index=False)
    print(f"✓ Saved multivariate data to {multivar_file}")
    
    # ==========================================
    # OPTION 3: Department-level for one store (many time series)
    # ==========================================
    print("\nCreating department-level time series for Store 1...")
    
    # Get all departments for Store 1
    store1_dept_sales = sales[sales['Store'] == 1].copy()
    
    # Pivot to wide format: dates as rows, departments as columns
    dept_pivot = store1_dept_sales.pivot(index='Date', columns='Dept', values='Weekly_Sales')
    
    # Rename columns
    dept_pivot.columns = [f'Dept_{col}' for col in dept_pivot.columns]
    
    # Reset index and rename Date to date
    dept_wide = dept_pivot.reset_index().rename(columns={'Date': 'date'})
    
    # Handle missing values
    for col in dept_wide.columns:
        if col != 'date':
            dept_wide[col] = dept_wide[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    print(f"Department-level data shape: {dept_wide.shape}")
    
    # Save department version
    dept_file = 'datasets/walmart_departments.csv'
    dept_wide.to_csv(dept_file, index=False)
    print(f"✓ Saved department-level data to {dept_file}")
    
    # ==========================================
    # Summary and validation
    # ==========================================
    print(f"\n{'='*60}")
    print("DATA PREPARATION COMPLETE!")
    print(f"{'='*60}")
    
    # Check for any remaining issues
    print(f"\nData validation:")
    print(f"1. Store-level data: {sales_wide.shape}, NaN count: {sales_wide.isnull().sum().sum()}")
    print(f"2. Multivariate data: {multivar_data.shape}, NaN count: {multivar_data.isnull().sum().sum()}")
    print(f"3. Department data: {dept_wide.shape}, NaN count: {dept_wide.isnull().sum().sum()}")
    
    return sales_wide, multivar_data, dept_wide

if __name__ == '__main__':
    try:
        sales_wide, multivar_data, dept_wide = prepare_walmart_for_ts2vec()
        
       
        print(f"\n{'='*60}")
        print("FILES CREATED:")
        print("- datasets/walmart_stores.csv (45 store time series)")
        print("- datasets/walmart_multivar.csv (1 store + features)")  
        print("- datasets/walmart_departments.csv (departments for Store 1)")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()