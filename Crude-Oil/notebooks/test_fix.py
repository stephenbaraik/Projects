#!/usr/bin/env python3

# Quick test to verify the fix works
from main import OilNiftyForecaster
import warnings
warnings.filterwarnings('ignore')

print("🧪 Testing the fix...")

try:
    # Initialize the forecaster
    forecaster = OilNiftyForecaster()
    
    # Run just the initial steps to test the fix
    print("Step 1: Collecting data...")
    forecaster.collect_data()
    
    print("Step 2: Preprocessing data...")
    forecaster.preprocess_data()
    
    print("Step 3: Preparing features...")
    forecaster.prepare_features()
    
    print("Step 4: Training basic models...")
    forecaster.train_models()
    
    print("Step 5: Testing analyze_model_performance...")
    forecaster.analyze_model_performance()
    
    print("✅ Fix successful! The analyze_model_performance method works correctly.")
    
except Exception as e:
    print(f"❌ Error still exists: {e}")
    import traceback
    traceback.print_exc()
