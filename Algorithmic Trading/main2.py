import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import linregress
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class EnhancedAlgoTradingSystem:
    def __init__(self, data, risk_free_rate=0.02):
        self.data = data
        self.model = None
        self.risk_free_rate = risk_free_rate

    # ... [Previous methods remain the same: preprocess_data, calculate_rsi, calculate_macd, train_model] ...

    def generate_signals(self):
        features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Volatility']
        X = self.data[features]
        self.data['Predicted_Returns'] = self.model.predict(X)
        self.data['Raw_Signal'] = np.where(self.data['Predicted_Returns'] > 0, 1, -1)

    def apply_risk_management(self, max_position_size=0.1, stop_loss=0.02, take_profit=0.05):
        self.data['Position'] = self.data['Raw_Signal'] * max_position_size
        
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

    def calculate_sharpe_ratio(self):
        excess_returns = self.data['Portfolio_Returns'] - self.risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        return sharpe_ratio

    def optimize_portfolio(self, assets):
        def portfolio_volatility(weights, returns):
            return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

        def portfolio_return(weights, returns):
            return np.sum(returns.mean() * weights) * 252

        def portfolio_sharpe(weights, returns, risk_free_rate):
            p_ret = portfolio_return(weights, returns)
            p_vol = portfolio_volatility(weights, returns)
            return (p_ret - risk_free_rate) / p_vol

        returns = self.data[assets].pct_change().dropna()
        n = len(assets)
        args = (returns, self.risk_free_rate)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for asset in range(n))
        result = minimize(lambda weights, returns, rf: -portfolio_sharpe(weights, returns, rf),
                          n * [1./n], args=args, method='SLSQP', bounds=bounds, constraints=constraints)

        optimal_weights = result.x
        optimal_sharpe = -result.fun
        
        return optimal_weights, optimal_sharpe

    def backtest(self):
        cumulative_returns = (1 + self.data['Portfolio_Returns']).cumprod()
        sharpe_ratio = self.calculate_sharpe_ratio()
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns.index, cumulative_returns)
        plt.title('Cumulative Returns of Enhanced Trading Strategy')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.show()
        
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Total Return: {cumulative_returns.iloc[-1]:.4f}")
        print(f"Max Drawdown: {max_drawdown:.4f}")

    def run(self):
        self.preprocess_data()
        self.train_model()
        self.generate_signals()
        self.apply_risk_management()
        self.calculate_portfolio_returns()
        self.backtest()

        # Example of portfolio optimization
        assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']  # Assuming these are columns in your data
        optimal_weights, optimal_sharpe = self.optimize_portfolio(assets)
        print("Optimal Portfolio Weights:")
        for asset, weight in zip(assets, optimal_weights):
            print(f"{asset}: {weight:.4f}")
        print(f"Optimal Portfolio Sharpe Ratio: {optimal_sharpe:.4f}")

# Usage:
data = pd.read_csv('multi_stock_data.csv', index_col='Date', parse_dates=True)
trading_system = EnhancedAlgoTradingSystem(data)
trading_system.run()

# This CSV file includes stock data for three companies (AAPL, GOOGL, MSFT) across multiple dates with the columns: Date, Stock Symbol, Open, High, Low, Close, and Volume