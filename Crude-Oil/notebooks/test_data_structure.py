#!/usr/bin/env python3
"""
Test script to verify the data structure and folder organization
"""

import os
import pandas as pd
from pathlib import Path

# Check the organized folder structure
DATA_DIR = "market_data"
FOLDERS = {
    'oil': f"{DATA_DIR}/oil_prices",
    'indices': f"{DATA_DIR}/indian_indices", 
    'companies': f"{DATA_DIR}/oil_companies",
    'currency': f"{DATA_DIR}/currency_rates",
    'combined': f"{DATA_DIR}/combined_data",
    'reports': f"{DATA_DIR}/reports"
}

def check_folder_structure():
    """Check if all folders exist and list their contents"""
    print("🗂️  CHECKING FOLDER STRUCTURE")
    print("=" * 50)
    
    for folder_name, folder_path in FOLDERS.items():
        print(f"\n📁 {folder_name.upper()} ({folder_path}):")
        
        if os.path.exists(folder_path):
            files = list(Path(folder_path).glob("*.csv"))
            if files:
                for file in sorted(files):
                    print(f"   ✅ {file.name}")
            else:
                print("   📭 No CSV files found")
        else:
            print("   ❌ Folder does not exist")

def check_combined_data():
    """Check the combined dataset"""
    print(f"\n📊 CHECKING COMBINED DATASET")
    print("=" * 50)
    
    combined_file = f"{FOLDERS['combined']}/combined_market_data.csv"
    if os.path.exists(combined_file):
        try:
            df = pd.read_csv(combined_file, index_col=0)
            print(f"✅ Combined dataset loaded successfully")
            print(f"   • Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"   • Date range: {df.index.min()} to {df.index.max()}")
            print(f"   • Sample columns: {list(df.columns[:5])}")
            
            # Check for oil companies
            oil_company_cols = [col for col in df.columns if any(company in col for company in ['ONGC', 'RELIANCE', 'IOC', 'BPCL', 'HPCL'])]
            print(f"   • Oil companies found: {len(oil_company_cols)}")
            
            # Check data completeness
            if len(df) > 0:
                print(f"   • Data completeness: {(1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.1f}%")
            else:
                print("   ⚠️  No data rows found")
                
        except Exception as e:
            print(f"   ❌ Error reading combined data: {str(e)}")
    else:
        print("   ❌ Combined dataset file not found")

def check_reports():
    """Check the generated reports"""
    print(f"\n📋 CHECKING REPORTS")
    print("=" * 50)
    
    reports = [
        'validation_report.csv',
        'data_summary.csv', 
        'oil_companies_summary.csv'
    ]
    
    for report in reports:
        report_path = f"{FOLDERS['reports']}/{report}"
        if os.path.exists(report_path):
            try:
                df = pd.read_csv(report_path)
                print(f"   ✅ {report}: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                print(f"   ❌ Error reading {report}: {str(e)}")
        else:
            print(f"   ❌ {report}: Not found")

if __name__ == "__main__":
    print("🧪 DATA STRUCTURE VERIFICATION")
    print("=" * 60)
    
    check_folder_structure()
    check_combined_data()
    check_reports()
    
    print(f"\n✅ Structure verification complete!")
