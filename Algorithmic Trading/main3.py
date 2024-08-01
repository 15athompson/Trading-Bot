import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import talib
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta

class AdvancedAlgoTradingSystem:
    def __init__(self, data, risk_free_rate=0.02):
        self.data = data
        self.models = {}
        self.risk_free_rate = risk_free_rate
        self.scaler = StandardScaler()

    def preprocess_data(self):
        # Calculate basic technical indicators
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['RSI'] = talib.RSI(self.data['Close'])
        self.data['MACD'], _, _ = talib.MACD(self.data['Close'])
        self.data['ATR'] = talib.ATR(self.data['High'], self.data['Low'], self.data['Close'])
        
        # Calculate advanced technical indicators
        self.data['BB_upper'], self.data['BB_middle'], self.data['BB_lower'] = talib.BBANDS(self.data['Close'])
        self.data['OBV'] = talib.OBV(self.data['Close'], self.data['Volume'])
        self.data['ADX'] = talib.ADX(self.data['High'], self.data['Low'], self.data['Close'])
        
        # Calculate returns
        self.data['Returns'] = self.data['Close'].pct_change()
        
        # Remove NaN values
        self.data.dropna(inplace=True)

    def detect_market_regime(self, window=252):
        # Calculate rolling mean and standard deviation
        returns = self.data['Returns']
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        
        # Define market regimes
        self.data['Market_Regime'] = np.where(returns > (rolling_mean + rolling_std), 'Bull',
                                     np.where(returns < (rolling_mean - rolling_std), 'Bear', 'Neutral'))

    def train_models(self):
        features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'ATR', 'BB_upper', 'BB_lower', 'OBV', 'ADX']
        X = self.data[features]
        y = self.data['Returns'].shift(-1)  # Predict next day's returns
        
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled[:-1], y[:-1], test_size=0.2, random_state=42)
        
        # Random Forest
        rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
        rf = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5)
        rf.fit(X_train, y_train)
        self.models['RF'] = rf.best_estimator_
        
        # Gradient Boosting
        gb_params = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
        gb = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_params, cv=5)
        gb.fit(X_train, y_train)
        self.models['GB'] = gb.best_estimator_
        
        # Print model performance
        for name, model in self.models.items():
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            print(f"{name} Train R-squared: {train_score:.4f}")
            print(f"{name} Test R-squared: {test_score:.4f}")

    def generate_signals(self):
        features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'ATR', 'BB_upper', 'BB_lower', 'OBV', 'ADX']
        X = self.data[features]
        X_scaled = self.scaler.transform(X)
        
        self.data['RF_Pred'] = self.models['RF'].predict(X_scaled)
        self.data['GB_Pred'] = self.models['GB'].predict(X_scaled)
        self.data['Ensemble_Pred'] = (self.data['RF_Pred'] + self.data['GB_Pred']) / 2
        
        self.data['Signal'] = np.where(self.data['Ensemble_Pred'] > 0, 1, -1)

    def apply_risk_management(self, max_position_size=0.1, stop_loss=0.02, take_profit=0.05):
        self.data['Position'] = self.data['Signal'] * max_position_size
        
        # Apply stop-loss and take-profit
        current_position = 0
        entry_price = 0
        for i in range(1, len(self.data)):
            if current_position == 0 and self.data['Position'].iloc[i] != 0:
                current_position = self.data['Position'].iloc[i]
                entry_price = self.data['Close'].iloc[i]
            elif current_position != 0:
                price_change = (self.data['Close'].iloc[i] - entry_price) / entry_price
                if (price_change <= -stop_loss) or (price_change >= take_profit):
                    current_position = 0
                    entry_price = 0
            
            self.data.at[self.data.index[i], 'Position'] = current_position

    def calculate_portfolio_returns(self):
        self.data['Portfolio_Returns'] = self.data['Position'].shift(1) * self.data['Returns']

    def calculate_var(self, confidence_level=0.95, time_horizon=1):
        returns = self.data['Portfolio_Returns'].dropna()
        var = np.percentile(returns, (1 - confidence_level) * 100) * np.sqrt(time_horizon)
        return var

    def calculate_expected_shortfall(self, confidence_level=0.95):
        returns = self.data['Portfolio_Returns'].dropna()
        var = self.calculate_var(confidence_level)
        expected_shortfall = returns[returns <= var].mean()
        return expected_shortfall

    def optimize_portfolio(self, assets):
        # ... [Same as before] ...

    def backtest(self):
        self.data['Cumulative_Returns'] = (1 + self.data['Portfolio_Returns']).cumprod()
        total_return = self.data['Cumulative_Returns'].iloc[-1] - 1
        sharpe_ratio = np.sqrt(252) * self.data['Portfolio_Returns'].mean() / self.data['Portfolio_Returns'].std()
        max_drawdown = (self.data['Cumulative_Returns'] / self.data['Cumulative_Returns'].cummax() - 1).min()
        
        var = self.calculate_var()
        es = self.calculate_expected_shortfall()
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data['Cumulative_Returns'])
        plt.title('Cumulative Returns of Advanced Trading Strategy')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.show()
        
        print(f"Total Return: {total_return:.4f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Max Drawdown: {max_drawdown:.4f}")
        print(f"Value at Risk (95%): {var:.4f}")
        print(f"Expected Shortfall (95%): {es:.4f}")

    def run_paper_trading_simulation(self, start_date, end_date, initial_capital=100000):
        simulation_data = self.data.loc[start_date:end_date].copy()
        simulation_data['Capital'] = initial_capital
        simulation_data['Shares'] = 0
        
        for i in range(1, len(simulation_data)):
            prev_capital = simulation_data['Capital'].iloc[i-1]
            prev_shares = simulation_data['Shares'].iloc[i-1]
            signal = simulation_data['Signal'].iloc[i]
            price = simulation_data['Close'].iloc[i]
            
            if signal == 1 and prev_shares == 0:  # Buy
                shares_to_buy = prev_capital // price
                cost = shares_to_buy * price
                simulation_data.at[simulation_data.index[i], 'Shares'] = shares_to_buy
                simulation_data.at[simulation_data.index[i], 'Capital'] = prev_capital - cost
            elif signal == -1 and prev_shares > 0:  # Sell
                revenue = prev_shares * price
                simulation_data.at[simulation_data.index[i], 'Shares'] = 0
                simulation_data.at[simulation_data.index[i], 'Capital'] = prev_capital + revenue
            else:  # Hold
                simulation_data.at[simulation_data.index[i], 'Shares'] = prev_shares
                simulation_data.at[simulation_data.index[i], 'Capital'] = prev_capital
        
        simulation_data['Total_Value'] = simulation_data['Capital'] + simulation_data['Shares'] * simulation_data['Close']
        final_return = (simulation_data['Total_Value'].iloc[-1] - initial_capital) / initial_capital
        
        plt.figure(figsize=(12, 6))
        plt.plot(simulation_data.index, simulation_data['Total_Value'])
        plt.title('Paper Trading Simulation Results')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.show()
        
        print(f"Paper Trading Simulation Return: {final_return:.4f}")

    def run(self):
        self.preprocess_data()
        self.detect_market_regime()
        self.train_models()
        self.generate_signals()
        self.apply_risk_management()
        self.calculate_portfolio_returns()
        self.backtest()
        
        # Example of paper trading simulation
        start_date = self.data.index[-252]  # Last year of data
        end_date = self.data.index[-1]
        self.run_paper_trading_simulation(start_date, end_date)

# Usage:
data = pd.read_csv('stock_data.csv', index_col='Date', parse_dates=True)
trading_system = AdvancedAlgoTradingSystem(data)
trading_system.run()