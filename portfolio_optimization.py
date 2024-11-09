import numpy as np
import pandas as pd
from scipy.optimize import minimize

class ModernPortfolioTheory:
    def __init__(self, risk_free_rate=0.02):
        self.risk_free_rate = risk_free_rate

    def calculate_returns(self, data):
        returns = {}
        for symbol, prices in data.items():
            returns[symbol] = prices['close'].pct_change().dropna()
        return pd.DataFrame(returns)

    def calculate_expected_returns(self, returns):
        return returns.mean() * 252  # Annualized returns

    def calculate_covariance_matrix(self, returns):
        return returns.cov() * 252  # Annualized covariance

    def portfolio_return(self, weights, expected_returns):
        return np.sum(expected_returns * weights)

    def portfolio_volatility(self, weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def sharpe_ratio(self, weights, expected_returns, cov_matrix):
        p_return = self.portfolio_return(weights, expected_returns)
        p_volatility = self.portfolio_volatility(weights, cov_matrix)
        return (p_return - self.risk_free_rate) / p_volatility

    def negative_sharpe_ratio(self, weights, expected_returns, cov_matrix):
        return -self.sharpe_ratio(weights, expected_returns, cov_matrix)

    def optimize(self, data):
        returns = self.calculate_returns(data)
        expected_returns = self.calculate_expected_returns(returns)
        cov_matrix = self.calculate_covariance_matrix(returns)

        num_assets = len(data)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = np.array([1/num_assets] * num_assets)

        result = minimize(
            self.negative_sharpe_ratio,
            initial_weights,
            args=(expected_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        return {symbol: weight for symbol, weight in zip(data.keys(), optimal_weights)}