import yfinance as yf

# Download Apple stock data for 2025
aapl = yf.download(
    "AAPL",
    start="2025-01-01",
    end="2026-01-01",
    progress=False
)

# Extract open and close prices as Python lists
open_prices_2025 = aapl["Open"].tolist()
close_prices_2025 = aapl["Close"].tolist()

print("Open prices:", open_prices_2025[:5])
print("Close prices:", close_prices_2025[:5])
print("Number of trading days:", len(close_prices_2025))
