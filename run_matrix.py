import pandas as pd
import numpy as np
import stumpy
import matplotlib.pyplot as plt

# ---------------------------
# USER SETTINGS
# ---------------------------
SECURITIES_FILE = "securities_dump.csv"  # static historical dump
TARGET_FILE     = "target_stock.csv"     # updated price history of target ticker

WINDOW  = 50     # size of pattern window to match
FORWARD = 10     # forward bars to examine
TOP_K   = 5      # number of best matches to return


# ---------------------------
# LOAD TARGET STOCK FILE
# ---------------------------
target_df = pd.read_csv(TARGET_FILE)

# Format expected:
# Ticker,Date1,Date2,Date3....
# AAPL,191.2,192.3,193.1,...

target_ticker = target_df.iloc[0, 0]
target_dates  = target_df.columns[1:]                     # target’s dates only
target_prices = target_df.iloc[0, 1:].astype(float).values

if len(target_prices) < WINDOW:
    raise ValueError("Target stock has fewer data points than WINDOW size.")

query = target_prices[-WINDOW:]  # pattern to match


# ---------------------------
# LOAD UNIVERSE HISTORICAL FILE
# ---------------------------
hist_df = pd.read_csv(SECURITIES_FILE)

# Format:
# Ticker,Date1,Date2,Date3,...
hist_dates = hist_df.columns[1:]  # these belong ONLY to historical file

universe = {}
date_map = {}                     # store date labels for each ticker

for _, row in hist_df.iterrows():
    ticker = row.iloc[0]
    prices = row.iloc[1:].dropna().astype(float).values

    if len(prices) >= WINDOW + FORWARD:
        universe[ticker] = prices
        date_map[ticker] = hist_dates   # map ticker ➜ its own dates


# ---------------------------
# RUN ANALOGUE SEARCH
# ---------------------------
results = []

for ticker, series in universe.items():

    dp = stumpy.mass(query, series)
    best_idx = np.argmin(dp)
    best_dist = dp[best_idx]

    # forward window must fit
    if best_idx + WINDOW + FORWARD <= len(series):
        future = series[best_idx + WINDOW : best_idx + WINDOW + FORWARD]

        # extract dates using *THIS TICKER'S* date columns
        ticker_dates = date_map[ticker]
        match_start_date = ticker_dates[best_idx]
        match_end_date   = ticker_dates[best_idx + WINDOW - 1]

        results.append(
            (
                ticker,
                best_idx,
                best_dist,
                future,
                match_start_date,
                match_end_date
            )
        )


# sort by distance
results = sorted(results, key=lambda x: x[2])[:TOP_K]


# ---------------------------
# AGGREGATE FORWARD PROJECTIONS
# ---------------------------
forward_paths = np.array([r[3] for r in results])
mean_forward  = forward_paths.mean(axis=0)
median_forward = np.median(forward_paths, axis=0)


# ---------------------------
# PRINT OUTPUT
# ---------------------------
print(f"\nTarget Stock: {target_ticker}")
print(f"Using last {WINDOW} bars | Forecasting next {FORWARD} bars\n")

print("Top Matches:")
for (ticker, idx, dist, future, start_date, end_date) in results:
    print(f"  {ticker:<8} | {start_date} → {end_date} | dist={dist:.4f}")

print("\nMean Forward Move:", mean_forward)
print("Median Forward Move:", median_forward)


# ---------------------------
# PLOT RESULTS
# ---------------------------
plt.figure(figsize=(10, 5))

for path in forward_paths:
    plt.plot(path, alpha=0.3)

plt.plot(mean_forward, linewidth=3, linestyle="--", label="Mean")
plt.plot(median_forward, linewidth=3, linestyle="-.", label="Median")

plt.title(f"Analogue Projection for {target_ticker}")
plt.xlabel("Forward Bars")
plt.ylabel("Price")
plt.grid(True)
plt.legend()
plt.show()
