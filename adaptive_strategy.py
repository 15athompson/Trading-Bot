import pandas as pd
import numpy as np
from strategies import MovingAverageCrossover, RSIStrategy

class AdaptiveStrategySelector:
    def __init__(self, strategies, lookback_period=30):
        self.strategies = strategies
        self.lookback_period = lookback_period
        self.performance_history = {strategy.__class__.__name__: [] for strategy in strategies}

    def select_strategy(self, data):
        # Update performance history for each strategy
        for strategy in self.strategies:
            strategy_name = strategy.__class__.__name__
            signal = strategy.generate_signal(data)
            returns = data['close'].pct_change().iloc[-1]
            performance = signal * returns
            self.performance_history[strategy_name].append(performance)

            # Keep only the last 'lookback_period' performances
            self.performance_history[strategy_name] = self.performance_history[strategy_name][-self.lookback_period:]

        # Calculate average performance for each strategy
        avg_performances = {strategy_name: np.mean(performances) for strategy_name, performances in self.performance_history.items()}

        # Select the best performing strategy
        best_strategy_name = max(avg_performances, key=avg_performances.get)
        best_strategy = next(strategy for strategy in self.strategies if strategy.__class__.__name__ == best_strategy_name)

        return best_strategy

    def generate_signal(self, data):
        best_strategy = self.select_strategy(data)
        return best_strategy.generate_signal(data)

# Example usage
ma_crossover = MovingAverageCrossover(short_window=20, long_window=50)
rsi_strategy = RSIStrategy(period=14, overbought=70, oversold=30)

adaptive_selector = AdaptiveStrategySelector([ma_crossover, rsi_strategy])

# This can be used in the main trading loop to get signals
# signal = adaptive_selector.generate_signal(latest_data)