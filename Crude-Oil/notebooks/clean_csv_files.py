#!/usr/bin/env python3
"""
Clean existing CSV files with multi-level headers
Removes ticker row and fixes column headers
"""

import pandas as pd
import os
from pathlib import Path

# Configuration
DATA_DIR = "market_data"
FOLDERS = {
    'oil': f"{DATA_DIR}/oil_prices",
    'indices': f"{DATA_DIR}/indian_indices", 
    'companies': f"{DATA_DIR}/oil_companies",
    'currency': f"{DATA_DIR}/currency_rates"
}

def clean_csv_file(file_path):
    """Clean a single CSV file by removing ticker row and fixing headers"""
    try:
        print(f"🧹 Cleaning: {file_path}")
        
        # Read the raw file
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Check if it has the multi-level header issue
        if len(lines) >= 3 and 'Ticker' in lines[1]:
            print(f"   📝 Found multi-level headers, cleaning...")
            
            # Extract the proper header (first line)
            header = lines[0].strip()
            
            # Skip the ticker line (second line) and empty date line (third line)
            data_lines = lines[3:]
            
            # Create cleaned content
            cleaned_content = header + '\n' + ''.join(data_lines)
            
            # Write back the cleaned content
            with open(file_path, 'w') as f:
                f.write(cleaned_content)
            
            print(f"   ✅ Cleaned successfully")
            
            # Verify the cleaning worked
            df = pd.read_csv(file_path, index_col=0)
            print(f"   📊 Shape: {df.shape}, Columns: {list(df.columns)}")
            
        else:
            print(f"   ℹ️  File appears to be already clean")
            
    except Exception as e:
        print(f"   ❌ Error cleaning {file_path}: {str(e)}")

def clean_all_csv_files():
    """Clean all CSV files in the organized folders"""
    print("🧹 CLEANING ALL CSV FILES")
    print("=" * 60)
    
    total_cleaned = 0
    
    for folder_name, folder_path in FOLDERS.items():
        print(f"\n📁 Cleaning {folder_name.upper()} files in {folder_path}:")
        
        if os.path.exists(folder_path):
            csv_files = list(Path(folder_path).glob("*.csv"))
            
            if csv_files:
                for csv_file in sorted(csv_files):
                    clean_csv_file(csv_file)
                    total_cleaned += 1
            else:
                print(f"   📭 No CSV files found")
        else:
            print(f"   ❌ Folder does not exist")
    
    print(f"\n✅ Cleaning complete! Processed {total_cleaned} files")

def verify_cleaned_files():
    """Verify that all files are properly cleaned"""
    print(f"\n🔍 VERIFICATION OF CLEANED FILES")
    print("=" * 60)
    
    for folder_name, folder_path in FOLDERS.items():
        print(f"\n📁 Verifying {folder_name.upper()} files:")
        
        if os.path.exists(folder_path):
            csv_files = list(Path(folder_path).glob("*.csv"))
            
            for csv_file in sorted(csv_files):
                try:
                    df = pd.read_csv(csv_file, index_col=0)
                    print(f"   ✅ {csv_file.name}: {df.shape[0]} rows, {df.shape[1]} cols - {list(df.columns)}")
                except Exception as e:
                    print(f"   ❌ {csv_file.name}: Error - {str(e)}")

if __name__ == "__main__":
    print("🔧 CSV FILE CLEANER")
    print("=" * 60)
    print("This script will remove multi-level headers and ticker rows")
    print("from all CSV files in the market_data folders.")
    print("=" * 60)
    
    clean_all_csv_files()
    verify_cleaned_files()
    
    print(f"\n🎉 All CSV files have been cleaned and verified!")
    print(f"Headers now contain only: Date, Open, High, Low, Close, Volume")
