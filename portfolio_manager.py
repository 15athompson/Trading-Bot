class PortfolioManager:
    def __init__(self, exchange, symbols):
        self.exchange = exchange
        self.symbols = symbols
        self.portfolio = {symbol: 0 for symbol in symbols}

    def update_portfolio(self):
        for symbol in self.symbols:
            try:
                balance = self.exchange.fetch_balance()
                self.portfolio[symbol] = balance[symbol.split('/')[0]]['total']
            except Exception as e:
                print(f"Error updating portfolio for {symbol}: {e}")

    def get_portfolio(self):
        return self.portfolio

    def get_position(self, symbol):
        return self.portfolio.get(symbol, 0)

    def calculate_portfolio_value(self):
        total_value = 0
        for symbol, amount in self.portfolio.items():
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                price = ticker['last']
                value = amount * price
                total_value += value
            except Exception as e:
                print(f"Error calculating value for {symbol}: {e}")
        return total_value