import yfinance as yf
import datetime
import numpy as np
import requests

class Stock:

    def __init__(self, ticker, alpaca):
        self.ticker = ticker
        self.alpaca = alpaca
        self._training_dataset = None
        self._evaluation_dataset = None
        self.refresh_quantity()


    def copy(self):
        return Stock(self.ticker)

    def quantity(self):
        return self._quantity

    def refresh_quantity(self):
        try:
            self._quantity = float(self.alpaca.get_position(self.ticker).qty)
        except Exception as e:
            print("Caught Exception:", e)
            self._quantity = 0
        return self._quantity

    def price(self):
        if self._quantity <= 0:
            barset = self.alpaca.get_barset(self.ticker, 'minute', limit=1)
            return barset[self.ticker][0].c
        else:
            try:
                return float(self.alpaca.get_position(self.ticker).current_price)
            except Exception as e:
                print("Caught Exception:", e)
                barset = self.alpaca.get_barset(self.ticker, 'minute', limit=1)
                return barset[self.ticker][0].c

    def volume(self):
        barset = self.alpaca.get_barset(self.ticker, 'minute', limit=1)
        return barset[self.ticker][0].v

    def get_prices(self, length=200):
        barset = self.alpaca.get_barset(self.ticker, 'minute', limit=length)
        return np.array([bar.c for bar in barset[self.ticker]])

    def get_volumes(self, length=200):
        barset = self.alpaca.get_barset(self.ticker, 'minute', limit=length)
        return np.array([bar.v for bar in barset[self.ticker]])

    def get_training_dataset(self):
        if self._training_dataset is not None:
            return self._training_dataset

        # data = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={}&interval=1min&outputsize=full&apikey=KZCPT19V688NX62O".format(self.ticker))
        # data = data.json()
        # data = data.get("Time Series (1min)", [])


        # is_before_seven_days = lambda date_str: datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') < (datetime.datetime.now() - datetime.timedelta(7))
        # self._training_dataset = [float(data[time]["4. close"]) for time in data]
        self._training_dataset = self.get_evaluation_dataset()

        print("Data for {} is {} long.".format(self.ticker, len(self._training_dataset)))
        return self._training_dataset

    def get_evaluation_dataset(self):
        if self._evaluation_dataset is None:
            data = yf.download(tickers=self.ticker, period="7d", interval="1m")
            self._evaluation_dataset = [float(p) for p in data.Close]
        return self._evaluation_dataset

    def close_position(self):
        self.buy_shares(-self.refresh_quantity())

    def buy_shares(self, qty):
        if qty == 0:
            print("Quantity is 0, order of |", self.ticker, "| not completed.")
            return

        side = "buy" if qty > 0 else "sell"
        try:
            print("Attempting to place market order of |", str(abs(qty)), self.ticker, side, "|.")
            self.alpaca.submit_order(self.ticker, abs(qty), side, "market", "day")
            print("Market order of |", str(abs(qty)), self.ticker, side, "| completed.")
            self._quantity += qty
        except Exception as e:
            print("Caught Exception:", e)
            print("Order of |", str(abs(qty)), self.ticker, side, "| did not go through.")
