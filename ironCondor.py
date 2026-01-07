import math
import yfinance as yf
import numpy as np

def iron_condor(
    S,
    sigma,
    T,
    r=0.0,
    width=5.0
):
    """
    Constructs a simple iron condor payoff.

    Returns strikes and payoff function.
    """
    # 1-sigma move
    move = S * sigma * math.sqrt(T)

    short_put = S - move
    long_put = short_put - width

    short_call = S + move
    long_call = short_call + width

    return {
        "long_put": long_put,
        "short_put": short_put,
        "short_call": short_call,
        "long_call": long_call
    }

def iron_condor_payoff(S_T, strikes, credit):
    """
    Payoff at expiration
    """
    lp, sp, sc, lc = (
        strikes["long_put"],
        strikes["short_put"],
        strikes["short_call"],
        strikes["long_call"]
    )

    payoff = credit

    # Put spread
    payoff -= max(0, sp - S_T)
    payoff += max(0, lp - S_T)

    # Call spread
    payoff -= max(0, S_T - sc)
    payoff += max(0, S_T - lc)

    return payoff

# ------------------------
# Example Backtest (1 trade)
# ------------------------

# Pull AAPL prices
data = yf.download("AAPL", period="1y", progress=False)
S0 = data["Close"].iloc[-1]

# Estimate volatility
returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()
sigma = returns.std() * math.sqrt(252)

T = 30 / 365  # 30 days

# Build iron condor
strikes = iron_condor(S0, sigma, T)

# Assume we collect $2.00 credit
credit = 2.00

# Simulate price range at expiration
prices = np.linspace(S0 * 0.8, S0 * 1.2, 200)

payoffs = [iron_condor_payoff(p, strikes, credit) for p in prices]

print("Iron Condor Strikes:")
for k, v in strikes.items():
    print(f"{k}: {v:.2f}")

print("\nMax Profit:", max(payoffs))
print("Max Loss:", min(payoffs))
