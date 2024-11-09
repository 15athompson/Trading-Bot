import pandas as pd

class DataStream:
    def __init__(self, exchange, symbols, timeframe):
        self.exchange = exchange
        self.symbols = symbols
        self.timeframe = timeframe
        self.data_cache = {symbol: None for symbol in symbols}

    def get_latest_data(self, symbol, limit=100):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            self.data_cache[symbol] = df
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def get_cached_data(self, symbol):
        return self.data_cache.get(symbol)

    def update_data(self):
        for symbol in self.symbols:
            self.get_latest_data(symbol)

    def get_current_price(self, symbol):
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            print(f"Error fetching current price for {symbol}: {e}")
            return None