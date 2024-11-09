import ccxt
import time
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from strategies import MovingAverageCrossover, RSIStrategy
from risk_management import RiskManager
from portfolio_manager import PortfolioManager
from backtester import Backtester
from data_stream import DataStream
from ml_model import create_and_train_model, get_price_prediction
from portfolio_optimization import ModernPortfolioTheory
from notifications import notifier
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedTradingBot:
    def __init__(self):
        self.exchanges = self._initialize_exchanges()
        self.symbols = config.SYMBOLS
        self.timeframe = config.TIMEFRAME
        self.risk_manager = RiskManager(max_risk_per_trade=config.MAX_RISK_PER_TRADE)
        self.portfolio_managers = {ex: PortfolioManager(exchange, self.symbols) for ex, exchange in self.exchanges.items()}
        self.strategies = self._initialize_strategies()
        self.data_streams = {ex: DataStream(exchange, self.symbols, self.timeframe) for ex, exchange in self.exchanges.items()}
        self.backtester = Backtester(self.exchanges[config.EXCHANGES[0]], self.symbols, self.timeframe)
        self.ml_models = {symbol: {} for symbol in self.symbols}
        self.portfolio_optimizer = ModernPortfolioTheory()
        self.initial_portfolio_value = self._get_total_portfolio_value()
        self.last_notification_time = time.time()

    def _initialize_exchanges(self):
        exchanges = {}
        exchange_configs = {
            'binance': {'apiKey': config.BINANCE_API_KEY, 'secret': config.BINANCE_API_SECRET, 'options': {'defaultType': 'future'}},
            'kraken': {'apiKey': config.KRAKEN_API_KEY, 'secret': config.KRAKEN_API_SECRET},
            'coinbasepro': {'apiKey': config.COINBASE_API_KEY, 'secret': config.COINBASE_API_SECRET, 'password': config.COINBASE_API_PASSPHRASE}
        }
        
        for name in config.EXCHANGES:
            try:
                exchange_class = getattr(ccxt, name)
                exchanges[name] = exchange_class({**exchange_configs[name], 'enableRateLimit': True})
                exchanges[name].load_markets()
                logger.info(f"Successfully initialized {name} exchange")
            except Exception as e:
                logger.error(f"Failed to initialize {name} exchange: {e}")
        
        return exchanges

    def _initialize_strategies(self):
        strategies = {
            'MA': MovingAverageCrossover(),
            'RSI': RSIStrategy()
        }
        strategies.update(config.CUSTOM_STRATEGIES)
        return strategies

    def _get_total_portfolio_value(self):
        return sum(pm.get_total_value() for pm in self.portfolio_managers.values())

    def train_ml_models(self):
        for exchange in self.exchanges:
            for symbol in self.symbols:
                try:
                    data = self.data_streams[exchange].get_historical_data(symbol, '1d', 1000)
                    if data is not None and len(data) > 0:
                        self.ml_models[symbol][exchange] = create_and_train_model(data['close'])
                        logger.info(f"ML model trained for {symbol} on {exchange}")
                except Exception as e:
                    logger.error(f"Error training ML model for {symbol} on {exchange}: {e}")

    def execute_order(self, exchange, symbol, side, amount):
        try:
            order = self.exchanges[exchange].create_market_order(symbol, side, amount)
            logger.info(f"Order executed on {exchange}: {order}")
            self._check_large_order(exchange, symbol, side, amount)
            return order
        except ccxt.NetworkError as e:
            logger.error(f"Network error while executing order on {exchange}: {e}")
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error while executing order on {exchange}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error executing order on {exchange}: {e}")
        return None

    def _check_large_order(self, exchange, symbol, side, amount):
        threshold = config.LARGE_ORDER_THRESHOLD
        if amount > threshold:
            notifier.send_alert("Large Order", f"Large {side} order executed on {exchange} for {symbol}: {amount}")

    def _check_performance(self):
        current_value = self._get_total_portfolio_value()
        roi = (current_value - self.initial_portfolio_value) / self.initial_portfolio_value
        if roi >= 0.1:  # 10% ROI
            notifier.send_alert("Performance Milestone", f"Bot has achieved {roi:.2%} return on investment")
            self.initial_portfolio_value = current_value  # Reset for next milestone

    def _check_high_volatility(self, exchange, symbol, data):
        returns = data['close'].pct_change()
        volatility = returns.std()
        if volatility > config.HIGH_VOLATILITY_THRESHOLD:
            notifier.send_alert("High Volatility", f"{symbol} volatility has exceeded {config.HIGH_VOLATILITY_THRESHOLD:.2%} on {exchange}")

    def run(self):
        self.train_ml_models()
        while True:
            try:
                for exchange in self.exchanges:
                    exchange_data = {}
                    for symbol in self.symbols:
                        data = self.data_streams[exchange].get_latest_data(symbol)
                        if data is not None:
                            exchange_data[symbol] = data
                            self._check_high_volatility(exchange, symbol, data)

                    if not exchange_data:
                        continue

                    optimized_weights = self.portfolio_optimizer.optimize(exchange_data)

                    for symbol, weight in optimized_weights.items():
                        signals = self._generate_signals(exchange, symbol, exchange_data[symbol])
                        overall_signal = self.combine_signals(signals)
                        current_position = self.portfolio_managers[exchange].get_position(symbol)

                        target_position = weight * self.portfolio_managers[exchange].get_total_value()
                        position_difference = target_position - current_position

                        if position_difference > 0:
                            self.execute_order(exchange, symbol, 'buy', position_difference)
                        elif position_difference < 0:
                            self.execute_order(exchange, symbol, 'sell', abs(position_difference))

                    self.portfolio_managers[exchange].update_portfolio()
                    logger.info(f"Current portfolio on {exchange}: {self.portfolio_managers[exchange].get_portfolio()}")
                
                self._check_performance()
                time.sleep(60)  # Wait for 1 minute before next iteration

            except Exception as e:
                logger.error(f"An error occurred in the main loop: {e}")
                notifier.send_alert("Error", f"An error occurred in the main loop: {e}")
                time.sleep(60)

    def _generate_signals(self, exchange, symbol, data):
        signals = {}
        for name, strategy in self.strategies.items():
            try:
                signals[name] = strategy.generate_signal(data)
            except Exception as e:
                logger.error(f"Error generating signal for {name} strategy on {exchange} for {symbol}: {e}")

        if symbol in self.ml_models and exchange in self.ml_models[symbol]:
            try:
                ml_prediction = get_price_prediction(self.ml_models[symbol][exchange], data['close'])
                current_price = data['close'].iloc[-1]
                ml_signal = 1 if ml_prediction > current_price else -1
                signals['ML'] = ml_signal
            except Exception as e:
                logger.error(f"Error generating ML signal on {exchange} for {symbol}: {e}")

        return signals

    def combine_signals(self, signals):
        weights = {'MA': 0.3, 'RSI': 0.3, 'ML': 0.4}
        return sum(weights.get(name, 0) * signal for name, signal in signals.items() if name in weights)

    def run_backtest(self, start_date, end_date):
        results = self.backtester.run(self.strategies, start_date, end_date)
        logger.info(f"Backtest results: {results}")
        return results

if __name__ == "__main__":
    bot = AdvancedTradingBot()
    bot.run()