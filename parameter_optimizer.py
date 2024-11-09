import numpy as np
from scipy.optimize import differential_evolution
from advanced_backtester import AdvancedBacktester
import config

class ParameterOptimizer:
    def __init__(self, exchange, symbols, timeframe, strategy_class):
        self.backtester = AdvancedBacktester(exchange, symbols, timeframe)
        self.strategy_class = strategy_class
        self.symbols = symbols
        self.timeframe = timeframe

    def objective_function(self, params):
        strategy = self.strategy_class(*params)
        results = self.backtester.run(config.BACKTEST_START_DATE, config.BACKTEST_END_DATE, 
                                      initial_balance=10000, market_condition='normal')
        
        # Calculate the average return across all symbols
        avg_return = np.mean([results[symbol][strategy.__class__.__name__]['returns'] for symbol in self.symbols])
        
        # We want to maximize returns, so we return the negative of avg_return
        return -avg_return

    def optimize(self, param_bounds):
        result = differential_evolution(self.objective_function, param_bounds)
        return result.x, -result.fun

class GridSearch:
    def __init__(self, exchange, symbols, timeframe, strategy_class):
        self.backtester = AdvancedBacktester(exchange, symbols, timeframe)
        self.strategy_class = strategy_class
        self.symbols = symbols
        self.timeframe = timeframe

    def search(self, param_grid):
        best_params = None
        best_return = float('-inf')

        for params in self._generate_param_combinations(param_grid):
            strategy = self.strategy_class(*params)
            results = self.backtester.run(config.BACKTEST_START_DATE, config.BACKTEST_END_DATE, 
                                          initial_balance=10000, market_condition='normal')
            
            avg_return = np.mean([results[symbol][strategy.__class__.__name__]['returns'] for symbol in self.symbols])

            if avg_return > best_return:
                best_return = avg_return
                best_params = params

        return best_params, best_return

    def _generate_param_combinations(self, param_grid):
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        for combination in np.array(np.meshgrid(*values)).T.reshape(-1, len(keys)):
            yield combination

if __name__ == "__main__":
    from ccxt import binance
    from strategies import MovingAverageCrossover, RSIStrategy
    
    exchange = binance({'enableRateLimit': True})
    
    # Example usage of ParameterOptimizer
    ma_optimizer = ParameterOptimizer(exchange, config.SYMBOLS, config.TIMEFRAME, MovingAverageCrossover)
    ma_bounds = [(10, 50), (20, 200)]  # bounds for short_window and long_window
    best_ma_params, best_ma_return = ma_optimizer.optimize(ma_bounds)
    print(f"Best MA parameters: {best_ma_params}, Return: {best_ma_return}")

    # Example usage of GridSearch
    rsi_grid_search = GridSearch(exchange, config.SYMBOLS, config.TIMEFRAME, RSIStrategy)
    rsi_param_grid = {
        'period': [7, 14, 21],
        'overbought': [70, 75, 80],
        'oversold': [20, 25, 30]
    }
    best_rsi_params, best_rsi_return = rsi_grid_search.search(rsi_param_grid)
    print(f"Best RSI parameters: {best_rsi_params}, Return: {best_rsi_return}")