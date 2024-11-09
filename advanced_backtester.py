import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategies import MovingAverageCrossover, RSIStrategy
from portfolio_optimization import ModernPortfolioTheory
import config

class AdvancedBacktester:
    def __init__(self, exchange, symbols, timeframe):
        self.exchange = exchange
        self.symbols = symbols
        self.timeframe = timeframe
        self.strategies = self._initialize_strategies()
        self.portfolio_optimizer = ModernPortfolioTheory()

    def _initialize_strategies(self):
        strategies = {
            'MA': MovingAverageCrossover(),
            'RSI': RSIStrategy()
        }
        strategies.update(config.CUSTOM_STRATEGIES)
        return strategies

    def fetch_historical_data(self, symbol, start_date, end_date):
        ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, since=int(start_date.timestamp()) * 1000, limit=1000)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df[(df.index >= start_date) & (df.index <= end_date)]

    def simulate_market_condition(self, data, condition):
        if condition == 'normal':
            return data
        elif condition == 'bull':
            return data * 1.2  # Simulate 20% overall growth
        elif condition == 'bear':
            return data * 0.8  # Simulate 20% overall decline
        elif condition == 'volatile':
            noise = np.random.normal(0, 0.02, len(data))  # Add 2% daily volatility
            return data * (1 + noise)
        else:
            raise ValueError(f"Unknown market condition: {condition}")

    def run(self, start_date, end_date, initial_balance=10000, market_condition='normal'):
        results = {}
        for symbol in self.symbols:
            data = self.fetch_historical_data(symbol, start_date, end_date)
            data['close'] = self.simulate_market_condition(data['close'], market_condition)
            
            symbol_results = {}
            for name, strategy in self.strategies.items():
                balance = initial_balance
                position = 0
                trades = []

                for i in range(len(data)):
                    signal = strategy.generate_signal(data.iloc[:i+1])
                    current_price = data.iloc[i]['close']

                    if signal == 1 and position <= 0:  # Buy signal
                        position = balance / current_price
                        balance = 0
                        trades.append(('buy', current_price, position))
                    elif signal == -1 and position > 0:  # Sell signal
                        balance = position * current_price
                        position = 0
                        trades.append(('sell', current_price, balance))

                final_balance = balance + position * data.iloc[-1]['close']
                returns = (final_balance - initial_balance) / initial_balance * 100
                max_drawdown = self.calculate_max_drawdown(trades)

                symbol_results[name] = {
                    'final_balance': final_balance,
                    'returns': returns,
                    'num_trades': len(trades),
                    'max_drawdown': max_drawdown
                }

            results[symbol] = symbol_results

        # Portfolio optimization
        portfolio_weights = self.portfolio_optimizer.optimize(data)
        optimized_returns = self.calculate_portfolio_returns(results, portfolio_weights)
        results['portfolio_optimization'] = {
            'weights': portfolio_weights,
            'returns': optimized_returns
        }

        return results

    def calculate_max_drawdown(self, trades):
        peak = 0
        max_drawdown = 0
        for trade in trades:
            if trade[0] == 'sell':
                peak = max(peak, trade[2])
                drawdown = (peak - trade[2]) / peak
                max_drawdown = max(max_drawdown, drawdown)
        return max_drawdown

    def calculate_portfolio_returns(self, results, weights):
        total_return = 0
        for symbol, weight in weights.items():
            symbol_return = np.mean([strategy['returns'] for strategy in results[symbol].values()])
            total_return += symbol_return * weight
        return total_return

    def stress_test(self, start_date, end_date, num_simulations=100):
        market_conditions = ['normal', 'bull', 'bear', 'volatile']
        stress_test_results = {}

        for condition in market_conditions:
            condition_results = []
            for _ in range(num_simulations):
                result = self.run(start_date, end_date, market_condition=condition)
                portfolio_return = result['portfolio_optimization']['returns']
                condition_results.append(portfolio_return)

            stress_test_results[condition] = {
                'mean_return': np.mean(condition_results),
                'std_dev': np.std(condition_results),
                'min_return': np.min(condition_results),
                'max_return': np.max(condition_results)
            }

        return stress_test_results

if __name__ == "__main__":
    from ccxt import binance
    exchange = binance({'enableRateLimit': True})
    backtester = AdvancedBacktester(exchange, config.SYMBOLS, config.TIMEFRAME)
    
    start_date = datetime.strptime(config.BACKTEST_START_DATE, "%Y-%m-%d")
    end_date = datetime.strptime(config.BACKTEST_END_DATE, "%Y-%m-%d")
    
    results = backtester.run(start_date, end_date)
    print("Backtest results:", results)
    
    stress_test_results = backtester.stress_test(start_date, end_date)
    print("Stress test results:", stress_test_results)