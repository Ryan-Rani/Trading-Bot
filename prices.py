import math
from typing import Dict

def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x: float) -> float:
    """Standard normal probability density."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def black_scholes(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        q: float = 0.0,
) -> Dict[str, float]:
    """
    Compute Black‑Scholes price and Greeks for a European option.

    Parameters:
    - S: spot price
    - K: strike price
    - T: time to expiry in years (e.g. 30 days -> 30/365)
    - r: risk-free interest rate (annual, decimal)
    - sigma: volatility (annual, decimal)
    - option_type: 'call' or 'put'
    - q: continuous dividend yield (annual, decimal)

    Returns a dict with 'price', 'delta', 'gamma', 'vega', 'theta', 'rho'.
    """
    option_type = option_type.lower()
    if T <= 0:
        # At expiry: option value is intrinsic value (undiscounted)
        if option_type == "call":
            price = max(0.0, S - K)
        else:
            price = max(0.0, K - S)
        # Greeks collapse at expiry; return zeros except price
        return {"price": price, "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    if sigma <= 0:
        # Zero volatility -> forward price deterministic
        forward = S * math.exp(-q * T)
        discounted_payoff = math.exp(-r * T) * (max(0.0, forward - K) if option_type == "call" else max(0.0, K - forward))
        return {"price": discounted_payoff, "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if option_type == "call":
        price = S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        delta = math.exp(-q * T) * norm_cdf(d1)
        rho = K * T * math.exp(-r * T) * norm_cdf(d2)
        theta = (
                - (S * sigma * math.exp(-q * T) * norm_pdf(d1)) / (2 * sqrtT)
                - r * K * math.exp(-r * T) * norm_cdf(d2)
                + q * S * math.exp(-q * T) * norm_cdf(d1)
        )
    else:
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)
        delta = math.exp(-q * T) * (norm_cdf(d1) - 1.0)
        rho = -K * T * math.exp(-r * T) * norm_cdf(-d2)
        theta = (
                - (S * sigma * math.exp(-q * T) * norm_pdf(d1)) / (2 * sqrtT)
                + r * K * math.exp(-r * T) * norm_cdf(-d2)
                - q * S * math.exp(-q * T) * norm_cdf(-d1)
        )

    gamma = (math.exp(-q * T) * norm_pdf(d1)) / (S * sigma * sqrtT)
    vega = S * math.exp(-q * T) * norm_pdf(d1) * sqrtT  # per 1.0 vol (multiply by 0.01 for %)

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }

if __name__ == "__main__":
    # Example usage
    S = 150.0        # spot price
    K = 160.0        # strike
    days = 30        # days to expiry
    T = days / 365.0
    r = 0.04         # 4% risk-free
    sigma = 0.25     # 25% vol
    q = 0.0          # dividend yield

    result_call = black_scholes(S, K, T, r, sigma, "call", q)
    result_put = black_scholes(S, K, T, r, sigma, "put", q)

    print("Call:", result_call)
    print("Put: ", result_put)
