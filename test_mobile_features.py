import unittest
import requests
import json
from main import calculate_performance_metrics, send_notification
import config

class TestMobileFeatures(unittest.TestCase):
    def setUp(self):
        self.api_url = config.MOBILE_APP_API_URL

    def test_performance_calculation(self):
        trades = [
            {'type': 'buy', 'price': 100, 'timestamp': '2023-01-01', 'profit': 0},
            {'type': 'sell', 'price': 110, 'timestamp': '2023-01-02', 'profit': 10},
            {'type': 'buy', 'price': 105, 'timestamp': '2023-01-03', 'profit': 0},
            {'type': 'sell', 'price': 115, 'timestamp': '2023-01-04', 'profit': 10},
        ]
        performance = calculate_performance_metrics(trades)
        self.assertEqual(performance['totalProfit'], 20)
        self.assertEqual(performance['winRate'], 1.0)
        self.assertEqual(performance['numberOfTrades'], 4)

    def test_notification_sending(self):
        message = "Test notification"
        send_notification(message)
        # Since we can't easily verify the notification was sent, we'll just check that no exception was raised

    def test_api_endpoints(self):
        # Test login
        login_data = {'username': 'testuser', 'password': 'testpassword'}
        response = requests.post(f"{self.api_url}/login", json=login_data)
        self.assertEqual(response.status_code, 200)
        token = response.json().get('token')
        self.assertIsNotNone(token)

        headers = {'Authorization': f'Bearer {token}'}

        # Test status endpoint
        response = requests.get(f"{self.api_url}/status", headers=headers)
        self.assertEqual(response.status_code, 200)
        status_data = response.json()
        self.assertIn('status', status_data)
        self.assertIn('currentPrice', status_data)

        # Test performance endpoint
        response = requests.get(f"{self.api_url}/performance", headers=headers)
        self.assertEqual(response.status_code, 200)
        performance_data = response.json()
        self.assertIn('totalProfit', performance_data)
        self.assertIn('winRate', performance_data)

        # Test settings endpoint
        response = requests.get(f"{self.api_url}/settings", headers=headers)
        self.assertEqual(response.status_code, 200)
        settings_data = response.json()
        self.assertIn('riskPerTrade', settings_data)

        # Test updating settings
        new_settings = {'riskPerTrade': 0.03}
        response = requests.post(f"{self.api_url}/settings", headers=headers, json=new_settings)
        self.assertEqual(response.status_code, 200)
        updated_settings = response.json()
        self.assertEqual(updated_settings['riskPerTrade'], 0.03)

if __name__ == '__main__':
    unittest.main()