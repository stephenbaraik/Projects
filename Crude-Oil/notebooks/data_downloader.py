#!/usr/bin/env python3
"""
Comprehensive Oil-Indian Markets Data Downloader
=====================================================

This script downloads real market data for research analysis:
- WTI & Brent Crude Oil prices
- Top 5 Indian stock market indices
- Saves data to CSV files for fast loading in notebooks

Author: Stephen Baraik
Date: July 19, 2025
Purpose: Research paper data preparation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Configuration
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
DATA_DIR = "market_data"

def create_data_directory():
    """Create data directory if it doesn't exist"""
    Path(DATA_DIR).mkdir(exist_ok=True)
    print(f"📁 Data directory: {os.path.abspath(DATA_DIR)}")

def download_with_retry(ticker, max_retries=3):
    """Download data with retry mechanism"""
    for attempt in range(max_retries):
        try:
            print(f"   🔄 Attempt {attempt + 1}/{max_retries} for {ticker}")
            data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if len(data) > 0:
                print(f"   ✅ Success: {len(data)} data points")
                return data
            else:
                print(f"   ⚠️  No data returned for {ticker}")
        except Exception as e:
            print(f"   ❌ Attempt {attempt + 1} failed: {str(e)[:100]}...")
            if attempt < max_retries - 1:
                print(f"   ⏳ Waiting 5 seconds before retry...")
                import time
                time.sleep(5)
    
    print(f"   ❌ All attempts failed for {ticker}")
    return None

def download_oil_data():
    """Download crude oil data"""
    print("\n🛢️  DOWNLOADING OIL DATA")
    print("=" * 50)
    
    oil_tickers = {
        'WTI': 'CL=F',      # WTI Crude Oil (NYMEX)
        'BRENT': 'BZ=F'     # Brent Crude Oil (ICE)
    }
    
    oil_data = {}
    
    for oil_name, ticker in oil_tickers.items():
        print(f"\n📈 Downloading {oil_name} ({ticker})...")
        data = download_with_retry(ticker)
        
        if data is not None:
            # Save individual oil data
            filename = f"{DATA_DIR}/{oil_name}_data.csv"
            data.to_csv(filename)
            print(f"   💾 Saved to: {filename}")
            
            # Store for combined dataset
            oil_data[oil_name] = data['Close'].dropna()
        else:
            print(f"   ❌ Failed to download {oil_name}")
    
    return oil_data

def download_indian_market_data():
    """Download Indian stock market data"""
    print("\n🇮🇳 DOWNLOADING INDIAN MARKET DATA")
    print("=" * 50)
    
    indian_tickers = {
        'NIFTY50': '^NSEI',     # Nifty 50 - Premier large-cap
        'NIFTY100': '^CNX100',  # Nifty 100 - Extended large-cap
        'NIFTY500': '^CRSLDX',  # Nifty 500 - Broad market
        'SENSEX': '^BSESN',     # Sensex - BSE flagship
        'NIFTYBANK': '^NSEBANK' # Nifty Bank - Banking sector
    }
    
    indian_data = {}
    
    for index_name, ticker in indian_tickers.items():
        print(f"\n📊 Downloading {index_name} ({ticker})...")
        data = download_with_retry(ticker)
        
        if data is not None:
            # Save individual index data
            filename = f"{DATA_DIR}/{index_name}_data.csv"
            data.to_csv(filename)
            print(f"   💾 Saved to: {filename}")
            
            # Store for combined dataset
            indian_data[index_name] = data['Close'].dropna()
        else:
            print(f"   ❌ Failed to download {index_name}")
    
    return indian_data

def create_combined_dataset(oil_data, indian_data):
    """Create and save combined dataset"""
    print("\n🔄 CREATING COMBINED DATASET")
    print("=" * 50)
    
    # Create combined DataFrame
    combined_data = pd.DataFrame()
    
    # Add oil data
    for oil_name, data in oil_data.items():
        combined_data[f'{oil_name}_Price'] = data
    
    # Add Indian market data
    for index_name, data in indian_data.items():
        combined_data[f'{index_name}_Price'] = data
    
    # Clean data: remove rows with any missing values
    print(f"📊 Before cleaning: {len(combined_data)} rows")
    combined_data = combined_data.dropna()
    print(f"📊 After cleaning: {len(combined_data)} rows")
    
    # Save combined dataset
    combined_filename = f"{DATA_DIR}/combined_market_data.csv"
    combined_data.to_csv(combined_filename)
    print(f"💾 Combined dataset saved to: {combined_filename}")
    
    # Create summary statistics
    summary = combined_data.describe()
    summary_filename = f"{DATA_DIR}/data_summary.csv"
    summary.to_csv(summary_filename)
    print(f"📈 Summary statistics saved to: {summary_filename}")
    
    return combined_data

def validate_data(combined_data):
    """Validate downloaded data quality"""
    print("\n✅ DATA VALIDATION")
    print("=" * 50)
    
    print(f"📅 Date Range: {combined_data.index.min()} to {combined_data.index.max()}")
    print(f"📈 Total Trading Days: {len(combined_data)}")
    print(f"🔢 Total Instruments: {len(combined_data.columns)}")
    print(f"📊 Data Completeness: 100% (no missing values)")
    
    # Check correlations
    price_cols = [col for col in combined_data.columns if 'Price' in col]
    correlations = combined_data[price_cols].corr()
    
    # Oil-oil correlation
    if 'WTI_Price' in combined_data.columns and 'BRENT_Price' in combined_data.columns:
        oil_corr = correlations.loc['WTI_Price', 'BRENT_Price']
        print(f"🔗 WTI-Brent Correlation: {oil_corr:.3f}")
        
        if oil_corr > 0.8:
            print("   ✅ Strong positive correlation (expected)")
        else:
            print("   ⚠️  Correlation lower than expected")
    
    # Data range validation
    print(f"\n📊 PRICE RANGES:")
    for col in price_cols:
        if col in combined_data.columns:
            min_val = combined_data[col].min()
            max_val = combined_data[col].max()
            print(f"   {col}: ${min_val:.2f} - ${max_val:.2f}")
    
    # Save validation report
    validation_report = {
        'date_range': f"{combined_data.index.min()} to {combined_data.index.max()}",
        'total_days': len(combined_data),
        'instruments': len(combined_data.columns),
        'completeness': 100.0,
        'oil_correlation': oil_corr if 'WTI_Price' in combined_data.columns and 'BRENT_Price' in combined_data.columns else None
    }
    
    validation_df = pd.DataFrame([validation_report])
    validation_filename = f"{DATA_DIR}/validation_report.csv"
    validation_df.to_csv(validation_filename, index=False)
    print(f"📋 Validation report saved to: {validation_filename}")

def main():
    """Main execution function"""
    print("🚀 COMPREHENSIVE MARKET DATA DOWNLOADER")
    print("=" * 80)
    print("📊 Downloading: WTI, Brent vs Top 5 Indian Indices")
    print("🎯 Purpose: Research paper data preparation")
    print("📅 Period: 2015-2024 (10 years)")
    print("=" * 80)
    
    # Create data directory
    create_data_directory()
    
    # Download data
    oil_data = download_oil_data()
    indian_data = download_indian_market_data()
    
    # Check if we have any data
    if not oil_data and not indian_data:
        print("\n❌ ERROR: No data downloaded successfully!")
        print("   Check your internet connection and try again.")
        sys.exit(1)
    
    # Create combined dataset
    if oil_data or indian_data:
        combined_data = create_combined_dataset(oil_data, indian_data)
        validate_data(combined_data)
        
        print(f"\n🎉 DATA DOWNLOAD COMPLETE!")
        print(f"📁 All files saved in: {os.path.abspath(DATA_DIR)}")
        print(f"📊 Ready for notebook analysis!")
        
        # Display preview
        print(f"\n📋 Data Preview (Latest 5 Days):")
        price_cols = [col for col in combined_data.columns if 'Price' in col]
        print(combined_data[price_cols].tail().round(2))
        
    else:
        print("\n❌ ERROR: Failed to download any market data!")

if __name__ == "__main__":
    main()
