import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class OilNiftyForecaster:
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.results = None
        self.predictions = None
        
    def collect_data(self, start_date="2010-01-01", end_date="2025-06-30"):
        """
        📦 Data Collection: Fetch Brent crude oil and Nifty50 data
        """
        print("📦 Collecting data...")
        
        # Fetch Brent crude oil data (BZ=F is Brent crude futures)
        brent = yf.download("BZ=F", start=start_date, end=end_date)
        brent = brent[['Close']].rename(columns={'Close': 'brent_price'})
        
        # Fetch Nifty50 data (^NSEI is Nifty50 index)
        nifty = yf.download("^NSEI", start=start_date, end=end_date)
        nifty = nifty[['Close']].rename(columns={'Close': 'nifty_price'})
        
        # Convert Brent from USD to INR (approximate conversion rate)
        # For simplicity, using a fixed rate - in practice, you'd fetch USD/INR data
        usd_to_inr = 83.0  # Approximate current rate
        brent['brent_price_inr'] = brent['brent_price'] * usd_to_inr
        
        # Merge datasets on date
        self.data = pd.merge(brent, nifty, left_index=True, right_index=True, how='inner')
        self.data = self.data.dropna()
        
        print(f"✅ Data collected: {len(self.data)} records from {self.data.index[0]} to {self.data.index[-1]}")
        return self.data
    
    def preprocess_data(self):
        """
        🧹 Data Preprocessing: Feature engineering and target creation
        """
        print("🧹 Preprocessing data...")
        
        df = self.data.copy()
        
        # Calculate daily percent changes
        df['brent_pct_change'] = df['brent_price_inr'].pct_change()
        df['nifty_pct_change'] = df['nifty_price'].pct_change()
        
        # Rolling averages and volatility
        df['brent_rolling_mean_7'] = df['brent_price_inr'].rolling(window=7).mean()
        df['brent_rolling_mean_30'] = df['brent_price_inr'].rolling(window=30).mean()
        df['brent_rolling_std_7'] = df['brent_price_inr'].rolling(window=7).std()
        df['brent_rolling_std_30'] = df['brent_price_inr'].rolling(window=30).std()
        
        # Lagged features
        for lag in [1, 3, 5, 7, 14]:
            df[f'brent_lag_{lag}'] = df['brent_pct_change'].shift(lag)
            df[f'nifty_lag_{lag}'] = df['nifty_pct_change'].shift(lag)
        
        # Target variables
        df['nifty_return_next_day'] = df['nifty_price'].pct_change().shift(-1)
        df['nifty_return_30d'] = (df['nifty_price'].shift(-30) - df['nifty_price']) / df['nifty_price']
        
        # Additional features
        df['brent_momentum_7'] = df['brent_price_inr'] / df['brent_rolling_mean_7']
        df['brent_momentum_30'] = df['brent_price_inr'] / df['brent_rolling_mean_30']
        df['volatility_ratio'] = df['brent_rolling_std_7'] / df['brent_rolling_std_30']
        
        # Remove rows with NaN values
        df = df.dropna()
        
        self.data = df
        print(f"✅ Preprocessing complete: {len(self.data)} records after cleaning")
        return self.data
    
    def prepare_features(self, target='nifty_return_30d'):
        """
        Prepare feature matrix and target variable
        """
        feature_cols = [
            'brent_pct_change', 'brent_rolling_mean_7', 'brent_rolling_std_7',
            'brent_lag_1', 'brent_lag_3', 'brent_lag_7', 'brent_lag_14',
            'brent_momentum_7', 'brent_momentum_30', 'volatility_ratio',
            'nifty_lag_1', 'nifty_lag_3', 'nifty_lag_7'
        ]
        
        X = self.data[feature_cols]
        y = self.data[target]
        
        # Train-test split (80-20, with most recent data as test)
        split_idx = int(len(X) * 0.8)
        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]
        
        print(f"📊 Train set: {len(self.X_train)} samples")
        print(f"📊 Test set: {len(self.X_test)} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """
        🧠 Train multiple models
        """
        print("🧠 Training models...")
        
        # Linear Regression
        lr = LinearRegression()
        lr.fit(self.X_train, self.y_train)
        self.models['Linear Regression'] = lr
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        self.feature_importance['Random Forest'] = pd.Series(
            rf.feature_importances_, index=self.X_train.columns
        ).sort_values(ascending=False)
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        xgb_model.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = xgb_model
        self.feature_importance['XGBoost'] = pd.Series(
            xgb_model.feature_importances_, index=self.X_train.columns
        ).sort_values(ascending=False)
        
        print("✅ Models trained successfully!")
        return self.models
    
    def train_improved_models(self):
        """
        🧠 Train improved models with better feature engineering
        """
        print("🧠 Training improved models...")
        
        # Feature scaling for better performance
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Linear Regression with regularization
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, self.y_train)
        self.models['Ridge Regression'] = ridge
        
        # Random Forest with better hyperparameters
        rf_improved = RandomForestRegressor(
            n_estimators=200, 
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42, 
            n_jobs=-1
        )
        rf_improved.fit(self.X_train, self.y_train)
        self.models['Random Forest (Improved)'] = rf_improved
        self.feature_importance['Random Forest (Improved)'] = pd.Series(
            rf_improved.feature_importances_, index=self.X_train.columns
        ).sort_values(ascending=False)
        
        # XGBoost with better hyperparameters
        xgb_improved = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        xgb_improved.fit(self.X_train, self.y_train)
        self.models['XGBoost (Improved)'] = xgb_improved
        self.feature_importance['XGBoost (Improved)'] = pd.Series(
            xgb_improved.feature_importances_, index=self.X_train.columns
        ).sort_values(ascending=False)
        
        print("✅ Improved models trained successfully!")
        return self.models
    
    def evaluate_models(self):
        """
        📈 Evaluate model performance
        """
        print("📊 Evaluating models...")
        
        results = {}
        predictions = {}
        
        for name, model in self.models.items():
            # Make predictions
            if name == 'Ridge Regression':
                # Use scaled features for Ridge regression
                y_pred = model.predict(self.scaler.transform(self.X_test))
            else:
                y_pred = model.predict(self.X_test)
            predictions[name] = y_pred
            
            # Calculate metrics
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            r2 = r2_score(self.y_test, y_pred)
            
            # Directional accuracy
            actual_direction = (self.y_test > 0).astype(int)
            pred_direction = (y_pred > 0).astype(int)
            directional_accuracy = (actual_direction == pred_direction).mean()
            
            results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'Directional Accuracy': directional_accuracy
            }
            
            print(f"\n{name}:")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  Directional Accuracy: {directional_accuracy:.4f}")
        
        self.results = results
        self.predictions = predictions
        return results
    
    def analyze_model_performance(self):
        """
        📊 Analyze why models might be performing poorly
        """
        print("\n🔍 Model Performance Analysis")
        print("=" * 50)
        
        # Check correlation between features and target
        # Create a combined dataset with features and target for correlation analysis
        feature_data = self.X_train.copy()
        feature_data['nifty_return_30d'] = self.y_train
        correlations = feature_data.corr()['nifty_return_30d'].abs().sort_values(ascending=False)
        
        print("📊 Feature-Target Correlations (absolute):")
        for feature, corr in correlations.head(10).items():
            if feature != 'nifty_return_30d':
                print(f"  {feature}: {corr:.4f}")
        
        # Check data distribution
        print(f"\n📊 Target Variable Statistics:")
        print(f"  Mean: {self.y_train.mean():.4f}")
        print(f"  Std: {self.y_train.std():.4f}")
        print(f"  Min: {self.y_train.min():.4f}")
        print(f"  Max: {self.y_train.max():.4f}")
        
        # Check for data leakage or issues
        print(f"\n🔍 Data Quality Checks:")
        print(f"  Missing values in features: {self.X_train.isnull().sum().sum()}")
        print(f"  Missing values in target: {self.y_train.isnull().sum()}")
        print(f"  Infinite values in features: {np.isinf(self.X_train).sum().sum()}")
        
        # Simple baseline model performance
        baseline_pred = np.full_like(self.y_test, self.y_train.mean())
        baseline_r2 = r2_score(self.y_test, baseline_pred)
        print(f"  Baseline R² (mean prediction): {baseline_r2:.4f}")
        
        return correlations
    
    def plot_results(self):
        """
        📊 Visualize results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted comparison
        ax1 = axes[0, 0]
        for name, pred in self.predictions.items():
            ax1.scatter(self.y_test, pred, alpha=0.6, label=name)
        ax1.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual 30-day Returns')
        ax1.set_ylabel('Predicted 30-day Returns')
        ax1.set_title('Actual vs Predicted Returns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Feature importance (XGBoost)
        ax2 = axes[0, 1]
        if 'XGBoost' in self.feature_importance:
            feat_imp = self.feature_importance['XGBoost'].head(10)
            feat_imp.plot(kind='barh', ax=ax2)
            ax2.set_title('Feature Importance (XGBoost)')
            ax2.set_xlabel('Importance')
        
        # 3. Model performance comparison
        ax3 = axes[1, 0]
        metrics_df = pd.DataFrame(self.results).T
        metrics_df[['MAE', 'RMSE']].plot(kind='bar', ax=ax3)
        ax3.set_title('Model Performance Comparison')
        ax3.set_ylabel('Error')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Time series of predictions vs actual
        ax4 = axes[1, 1]
        test_dates = self.data.index[len(self.X_train):]
        ax4.plot(test_dates, self.y_test, label='Actual', linewidth=2)
        for name, pred in self.predictions.items():
            ax4.plot(test_dates, pred, label=f'{name} Predicted', alpha=0.8)
        ax4.set_title('Time Series: Actual vs Predicted')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('30-day Returns')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def scenario_analysis(self, brent_change_pct=10, model_name='XGBoost'):
        """
        🔍 Scenario Analysis: What if Brent oil changes by X%?
        """
        print(f"🔍 Scenario Analysis: Brent oil {brent_change_pct:+.1f}% change impact")
        
        # Get the latest data point
        latest_data = self.data.iloc[-1:].copy()
        
        # Create scenario where Brent increases by specified percentage
        scenario_data = latest_data.copy()
        
        # Simulate gradual increase over 7 days
        for i in range(7):
            daily_change = brent_change_pct / 7 / 100  # Daily change percentage
            scenario_data['brent_pct_change'] = daily_change
            
            # Update other features accordingly
            scenario_data['brent_momentum_7'] = 1 + (brent_change_pct / 100)
            scenario_data['brent_momentum_30'] = 1 + (brent_change_pct / 200)  # Less impact on 30-day
            scenario_data['volatility_ratio'] = scenario_data['volatility_ratio'] * 1.2  # Increase volatility
        
        # Make prediction
        model = self.models[model_name]
        features = scenario_data[self.X_train.columns]
        predicted_return = model.predict(features)[0]
        
        # Calculate predicted price - fix the Series formatting issue
        current_nifty = float(self.data['nifty_price'].iloc[-1])
        predicted_price = current_nifty * (1 + predicted_return)
        
        print(f"\n📊 Scenario Results:")
        print(f"  Current Nifty50: {current_nifty:.2f}")
        print(f"  Predicted 30-day return: {predicted_return:.4f} ({predicted_return*100:.2f}%)")
        print(f"  Predicted Nifty50 price: {predicted_price:.2f}")
        print(f"  Expected price change: {predicted_price - current_nifty:.2f}")
        
        return predicted_return, predicted_price
    
    def run_complete_analysis(self):
        """
        🚀 Run the complete analysis pipeline
        """
        print("🚀 Starting Complete Oil-Nifty Forecasting Analysis")
        print("=" * 60)
        
        # Step 1: Data Collection
        self.collect_data()
        
        # Step 2: Data Preprocessing
        self.preprocess_data()
        
        # Step 3: Feature Preparation
        self.prepare_features()
        
        # Step 4: Model Training
        self.train_models()
        
        # Step 4.5: Improved Model Training
        self.train_improved_models()
        
        # Step 5: Model Evaluation
        self.evaluate_models()
        
        # Step 6: Performance Analysis
        self.analyze_model_performance()
        
        # Step 7: Visualizations
        self.plot_results()
        
        # Step 8: Scenario Analysis
        print("\n" + "=" * 60)
        self.scenario_analysis(brent_change_pct=10)
        self.scenario_analysis(brent_change_pct=-10)
        
        print("\n🎉 Analysis Complete!")
        
        return self

# Example usage
if __name__ == "__main__":
    # Initialize the forecaster
    forecaster = OilNiftyForecaster()
    
    # Run complete analysis
    forecaster.run_complete_analysis()
    
    # Additional scenario testing
    print("\n" + "=" * 60)
    print("🔮 Additional Scenario Testing")
    
    scenarios = [5, 15, -5, -15, 20]
    for change in scenarios:
        forecaster.scenario_analysis(brent_change_pct=change)
        print("-" * 40)