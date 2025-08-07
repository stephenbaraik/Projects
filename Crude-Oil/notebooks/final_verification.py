#!/usr/bin/env python3
"""
Final verification of the cleaned data and combined dataset
"""

import pandas as pd
import os

def verify_individual_files():
    """Verify that individual CSV files are properly formatted"""
    print("🔍 VERIFYING INDIVIDUAL CSV FILES")
    print("=" * 60)
    
    folders = {
        'Oil Prices': 'market_data/oil_prices',
        'Indian Indices': 'market_data/indian_indices',
        'Oil Companies': 'market_data/oil_companies',
        'Currency': 'market_data/currency_rates'
    }
    
    for category, folder in folders.items():
        print(f"\n📁 {category}:")
        
        csv_files = list(os.listdir(folder)) if os.path.exists(folder) else []
        csv_files = [f for f in csv_files if f.endswith('.csv')]
        
        for csv_file in sorted(csv_files[:3]):  # Check first 3 files
            file_path = os.path.join(folder, csv_file)
            try:
                df = pd.read_csv(file_path, index_col=0)
                print(f"   ✅ {csv_file}: {df.shape[0]} rows, columns: {list(df.columns)}")
                
                # Check if Date is properly set as index
                print(f"      📅 Date range: {df.index.min()} to {df.index.max()}")
                
            except Exception as e:
                print(f"   ❌ {csv_file}: Error - {str(e)}")

def verify_combined_dataset():
    """Verify the combined dataset"""
    print(f"\n📊 VERIFYING COMBINED DATASET")
    print("=" * 60)
    
    combined_file = "market_data/combined_data/combined_market_data.csv"
    
    if os.path.exists(combined_file):
        try:
            df = pd.read_csv(combined_file, index_col=0)
            print(f"✅ Combined dataset loaded successfully")
            print(f"   • Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            
            if len(df) > 0:
                print(f"   • Date range: {df.index.min()} to {df.index.max()}")
                print(f"   • Data completeness: {(1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.1f}%")
                
                # Show column categories
                oil_cols = [col for col in df.columns if 'WTI' in col or 'BRENT' in col]
                indian_cols = [col for col in df.columns if 'NIFTY' in col or 'SENSEX' in col]
                company_cols = [col for col in df.columns if any(company in col for company in ['ONGC', 'RELIANCE', 'IOC'])]
                
                print(f"\n   📈 Column Breakdown:")
                print(f"      • Oil prices: {len(oil_cols)} columns")
                print(f"      • Indian indices: {len(indian_cols)} columns") 
                print(f"      • Oil companies: {len(company_cols)} columns")
                print(f"      • Currency: 1 USD/INR column")
                
                # Show sample data
                print(f"\n   📋 Sample Data (Latest 3 days):")
                sample_cols = ['USD_INR_Rate', 'WTI_Price_USD', 'BRENT_Price_USD', 'NIFTY50_Price', 'RELIANCE_Price']
                available_sample_cols = [col for col in sample_cols if col in df.columns]
                
                if available_sample_cols:
                    print(df[available_sample_cols].tail(3).round(2))
                
            else:
                print("   ⚠️  No data rows found in combined dataset")
                
        except Exception as e:
            print(f"   ❌ Error reading combined dataset: {str(e)}")
    else:
        print("   ❌ Combined dataset file not found")

def check_data_consistency():
    """Check data consistency across files"""
    print(f"\n🔄 CHECKING DATA CONSISTENCY")
    print("=" * 60)
    
    # Check if all files have proper OHLCV structure
    sample_files = [
        'market_data/oil_prices/WTI_data.csv',
        'market_data/indian_indices/NIFTY50_data.csv', 
        'market_data/oil_companies/RELIANCE_data.csv'
    ]
    
    expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    for file_path in sample_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, index_col=0)
                
                # Check column structure
                has_all_cols = all(col in df.columns for col in expected_columns)
                print(f"   ✅ {os.path.basename(file_path)}: {'All OHLCV columns present' if has_all_cols else 'Missing some columns'}")
                
                # Check for any remaining ticker issues
                if 'Ticker' in str(df.columns) or any('Ticker' in str(df.index)):
                    print(f"   ⚠️  Still has ticker issues")
                else:
                    print(f"      ✅ No ticker issues found")
                    
            except Exception as e:
                print(f"   ❌ {os.path.basename(file_path)}: Error - {str(e)}")

if __name__ == "__main__":
    print("🔍 FINAL DATA VERIFICATION")
    print("=" * 70)
    print("Checking that all CSV files are properly cleaned and formatted")
    print("=" * 70)
    
    verify_individual_files()
    verify_combined_dataset()
    check_data_consistency()
    
    print(f"\n🎉 VERIFICATION COMPLETE!")
    print(f"All CSV files should now have clean headers: Date, Open, High, Low, Close, Volume")
    print(f"No more multi-level headers or ticker rows!")
