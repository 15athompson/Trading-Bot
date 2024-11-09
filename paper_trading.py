import time
from datetime import datetime
from collections import defaultdict
import logging
from advanced_bot import AdvancedTradingBot
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperTradingBot(AdvancedTradingBot):
    def __init__(self):
        super().__init__()
        self.paper_balance = config.PAPER_TRADING_BALANCE
        self.paper_positions = defaultdict(lambda: defaultdict(float))
        self.paper_trades = []

    def execute_order(self, exchange, symbol, side, amount):
        try:
            current_price = self.data_streams[exchange].get_latest_data(symbol)['close'].iloc[-1]
            
            if side == 'buy':
                cost = amount * current_price
                if cost > self.paper_balance:
                    logger.warning(f"Insufficient paper balance for {side} order on {exchange} for {symbol}")
                    return None
                self.paper_balance -= cost
                self.paper_positions[exchange][symbol] += amount
            elif side == 'sell':
                if amount > self.paper_positions[exchange][symbol]:
                    logger.warning(f"Insufficient paper position for {side} order on {exchange} for {symbol}")
                    return None
                revenue = amount * current_price
                self.paper_balance += revenue
                self.paper_positions[exchange][symbol] -= amount

            order = {
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': current_price,
                'timestamp': datetime.now().isoformat(),
                'exchange': exchange
            }
            self.paper_trades.append(order)
            logger.info(f"Paper trade executed on {exchange}: {order}")
            return order
        except Exception as e:
            logger.error(f"Error executing paper trade on {exchange}: {e}")
        return None

    def get_paper_portfolio(self):
        portfolio = {
            'balance': self.paper_balance,
            'positions': dict(self.paper_positions)
        }
        return portfolio

    def get_paper_trade_history(self):
        return self.paper_trades

    def run_paper_trading(self, duration_seconds):
        start_time = time.time()
        end_time = start_time + duration_seconds

        while time.time() < end_time:
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
                        current_position = self.paper_positions[exchange][symbol]

                        target_position = weight * self.paper_balance
                        position_difference = target_position - (current_position * exchange_data[symbol]['close'].iloc[-1])

                        if position_difference > 0:
                            self.execute_order(exchange, symbol, 'buy', position_difference / exchange_data[symbol]['close'].iloc[-1])
                        elif position_difference < 0:
                            self.execute_order(exchange, symbol, 'sell', abs(position_difference) / exchange_data[symbol]['close'].iloc[-1])

                logger.info(f"Current paper portfolio: {self.get_paper_portfolio()}")
                time.sleep(60)  # Wait for 1 minute before next iteration

            except Exception as e:
                logger.error(f"An error occurred in the paper trading loop: {e}")
                time.sleep(60)

        logger.info("Paper trading session completed")
        logger.info(f"Final paper portfolio: {self.get_paper_portfolio()}")
        logger.info(f"Trade history: {self.get_paper_trade_history()}")

if __name__ == "__main__":
    paper_bot = PaperTradingBot()
    paper_bot.run_paper_trading(3600)  # Run paper trading for 1 hour