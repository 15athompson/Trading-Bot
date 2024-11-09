import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from crypto_trading_bot.main import main as original_main
from crypto_trading_bot.paper_trading import PaperTrading
from crypto_trading_bot.config import PAPER_TRADING_INITIAL_BALANCE, PAPER_TRADING_FEE_RATE
import crypto_trading_bot.config as config

def paper_trading_main():
    # Initialize paper trading
    paper_trader = PaperTrading(PAPER_TRADING_INITIAL_BALANCE, PAPER_TRADING_FEE_RATE)
    
    # Store the original exchange methods
    original_create_order = config.exchange.create_order
    original_fetch_balance = config.exchange.fetch_balance
    
    # Override exchange methods with paper trading methods
    def paper_create_order(symbol, type, side, amount, price=None, params={}):
        if side == 'buy':
            success = paper_trader.buy(symbol, amount, price)
        elif side == 'sell':
            success = paper_trader.sell(symbol, amount, price)
        else:
            raise ValueError(f"Invalid order side: {side}")
        
        if success:
            return {
                'symbol': symbol,
                'type': type,
                'side': side,
                'amount': amount,
                'price': price,
                'status': 'closed'
            }
        else:
            raise Exception("Paper trading order failed")

    def paper_fetch_balance():
        balance = paper_trader.get_balance()
        positions = paper_trader.get_positions()
        return {
            'total': balance,
            'free': balance,
            'used': 0,
            'info': positions
        }

    # Replace exchange methods with paper trading methods
    config.exchange.create_order = paper_create_order
    config.exchange.fetch_balance = paper_fetch_balance

    # Run the original main function with paper trading
    original_main()

    # Restore original exchange methods
    config.exchange.create_order = original_create_order
    config.exchange.fetch_balance = original_fetch_balance

    # Print paper trading results
    print("\nPaper Trading Results:")
    print(f"Final Balance: {paper_trader.get_balance()}")
    print("Final Positions:")
    for symbol, amount in paper_trader.get_positions().items():
        print(f"  {symbol}: {amount}")
    print("\nTrade History:")
    for trade in paper_trader.get_trade_history():
        print(f"  {trade}")

if __name__ == "__main__":
    paper_trading_main()