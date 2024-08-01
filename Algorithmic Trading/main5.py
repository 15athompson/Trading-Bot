import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import talib
from datetime import datetime, timedelta
import requests
from textblob import TextBlob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class AdvancedMultiAssetAlgoTradingSystem:
    def __init__(self, symbols, start_date, end_date, risk_free_rate=0.02):
        # ... [Previous initialization code remains the same] ...

    # ... [Previous methods remain the same] ...

    def walk_forward_optimization(self, train_window=252, test_window=63):
        full_index = self.data[self.symbols[0]].index
        for train_end in tqdm(range(train_window, len(full_index) - test_window, test_window)):
            train_start = train_end - train_window
            test_start = train_end
            test_end = test_start + test_window

            # Train on the training window
            self.train_models(train_start, train_end)
            self.optimize_portfolio(train_start, train_end)

            # Generate signals and apply risk management for the test window
            self.generate_signals(test_start, test_end)
            self.apply_risk_management(test_start, test_end)

    def train_models(self, start_date, end_date):
        features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'ATR', 'BB_upper', 'BB_lower', 'OBV', 'ADX', 
                    'P/E', 'P/B', 'Dividend_Yield', 'Sentiment', 'Price_to_SMA50', 'Momentum']

        for symbol, df in self.data.items():
            train_data = df.loc[start_date:end_date]
            X = train_data[features]
            y = train_data['Returns'].shift(-1)  # Predict next day's returns
            
            X_scaled = self.scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled[:-1], y[:-1], test_size=0.2, random_state=42)
            
            # Random Forest
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            rf.fit(X_train, y_train)
            
            # Gradient Boosting
            gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            gb.fit(X_train, y_train)
            
            self.models[symbol] = {'RF': rf, 'GB': gb}

    def generate_signals(self, start_date, end_date):
        features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'ATR', 'BB_upper', 'BB_lower', 'OBV', 'ADX', 
                    'P/E', 'P/B', 'Dividend_Yield', 'Sentiment', 'Price_to_SMA50', 'Momentum']

        for symbol, df in self.data.items():
            test_data = df.loc[start_date:end_date]
            X = test_data[features]
            X_scaled = self.scaler.transform(X)
            
            df.loc[start_date:end_date, 'RF_Pred'] = self.models[symbol]['RF'].predict(X_scaled)
            df.loc[start_date:end_date, 'GB_Pred'] = self.models[symbol]['GB'].predict(X_scaled)
            df.loc[start_date:end_date, 'Ensemble_Pred'] = (df.loc[start_date:end_date, 'RF_Pred'] + df.loc[start_date:end_date, 'GB_Pred']) / 2
            
            # Combine model predictions with mean reversion and momentum signals
            df.loc[start_date:end_date, 'MR_Signal'] = np.where(df.loc[start_date:end_date, 'Price_to_SMA50'] < 0.95, 1, 
                                                                np.where(df.loc[start_date:end_date, 'Price_to_SMA50'] > 1.05, -1, 0))
            df.loc[start_date:end_date, 'Mom_Signal'] = np.where(df.loc[start_date:end_date, 'Momentum'] > 0, 1, -1)
            
            df.loc[start_date:end_date, 'Signal'] = np.sign(df.loc[start_date:end_date, 'Ensemble_Pred'] + 
                                                            0.5 * df.loc[start_date:end_date, 'MR_Signal'] + 
                                                            0.5 * df.loc[start_date:end_date, 'Mom_Signal'])

    def optimize_portfolio(self, start_date, end_date):
        # ... [Similar to previous optimize_portfolio method, but using data from start_date to end_date] ...

    def apply_risk_management(self, start_date, end_date, max_portfolio_var=0.02):
        # ... [Similar to previous apply_risk_management method, but using data from start_date to end_date] ...

    def calculate_transaction_costs(self, commission_rate=0.001):
        for symbol, df in self.data.items():
            position_changes = df['Position'].diff().abs()
            df['Transaction_Costs'] = position_changes * df['Close'] * commission_rate

    def implement_trailing_stop(self, trailing_stop_pct=0.05):
        for symbol, df in self.data.items():
            df['Trailing_Stop'] = df['Close'].cummax() * (1 - trailing_stop_pct)
            df.loc[df['Close'] < df['Trailing_Stop'], 'Signal'] = -1

    def monte_carlo_var(self, n_simulations=10000, confidence_level=0.95):
        portfolio_weights = self.portfolio['Weight'].values
        returns = np.array([df['Returns'].values for df in self.data.values()])
        
        cov_matrix = np.cov(returns)
        mean_returns = np.mean(returns, axis=1)
        
        simulated_returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_simulations)
        portfolio_simulated_returns = np.dot(portfolio_weights, simulated_returns.T)
        
        var = np.percentile(portfolio_simulated_returns, (1 - confidence_level) * 100)
        cvar = portfolio_simulated_returns[portfolio_simulated_returns <= var].mean()
        
        return var, cvar

    def backtest(self):
        portfolio_returns = self.calculate_portfolio_returns()
        cumulative_returns = (1 + portfolio_returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        # Calculate total transaction costs
        total_transaction_costs = sum(df['Transaction_Costs'].sum() for df in self.data.values())
        
        # Calculate VaR and CVaR using Monte Carlo simulation
        var, cvar = self.monte_carlo_var()
        
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns.index, cumulative_returns)
        plt.title('Cumulative Returns of Multi-Asset Trading Strategy')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.show()
        
        print(f"Total Return: {total_return:.4f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Max Drawdown: {max_drawdown:.4f}")
        print(f"Total Transaction Costs: {total_transaction_costs:.4f}")
        print(f"Value at Risk (95%): {var:.4f}")
        print(f"Conditional Value at Risk (95%): {cvar:.4f}")

    def run(self, sentiment_api_key):
        self.fetch_data()
        self.preprocess_data()
        self.fetch_fundamental_data()
        self.fetch_sentiment_data(sentiment_api_key)
        self.detect_market_regime()
        self.walk_forward_optimization()
        self.calculate_transaction_costs()
        self.implement_trailing_stop()
        self.backtest()

# Usage remains the same as before