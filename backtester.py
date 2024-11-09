import pandas as pd

class Backtester:
    def __init__(self, exchange, symbols, timeframe):
        self.exchange = exchange
        self.symbols = symbols
        self.timeframe = timeframe

    def fetch_historical_data(self, symbol, start_date, end_date):
        ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, since=int(start_date.timestamp()) * 1000, limit=1000)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df[(df.index >= start_date) & (df.index <= end_date)]

    def run(self, strategies, start_date, end_date):
        results = {}
        for symbol in self.symbols:
            data = self.fetch_historical_data(symbol, start_date, end_date)
            symbol_results = {}
            for name, strategy in strategies.items():
                initial_balance = 10000  # Starting with $10,000 for each strategy
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

                symbol_results[name] = {
                    'final_balance': final_balance,
                    'returns': returns,
                    'num_trades': len(trades)
                }

            results[symbol] = symbol_results

        return results