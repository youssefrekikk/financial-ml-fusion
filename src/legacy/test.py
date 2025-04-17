#to try and run move to src
from utils import load_data
ticker = "AAPL"  # Changed from SPY to AAPL
start_date = "2014-01-01"
end_date = "2024-01-01"

#
data = load_data(ticker, start_date, end_date)
print(data)