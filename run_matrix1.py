import pandas as pd
import numpy as np
import stumpy
import matplotlib.pyplot as plt

# ---------------------------
# USER SETTINGS
# ---------------------------
SECURITIES_FILE = "securities_dump.csv"
TARGET_FILE     = "target_stock.csv"

WINDOW  = 50     # size of pattern window to match
FORWARD = 10     # forward bars to examine
TOP_K   = 5      # number of best matches to return


# ---------------------------
# LOAD TARGET STOCK FILE
# ---------------------------
target_df = pd.read_csv(TARGET_FILE)

target_ticker = target_df.iloc[0, 0]
target_dates  = target_df.columns[1:]
target_prices = target_df.iloc[0, 1:].astype(float).values

if len(target_prices) < WINDOW:
    raise ValueError("Target stock has fewer data points than WINDOW size.")

query = target_prices[-WINDOW:]  # pattern to match


# ---------------------------
# LOAD HISTORICAL UNIVERSE
# ---------------------------
hist_df = pd.read_csv(SECURITIES_FILE)
hist_dates = hist_df.columns[1:]  # these dates only apply to the historical file

universe = {}
date_map = {}

for _, row in hist_df.iterrows():
    ticker = row.iloc[0]
    prices = row.iloc[1:].dropna().astype(float).values

    if len(prices) >= WINDOW + FORWARD:
        universe[ticker] = prices
        date_map[ticker] = hist_dates


# ---------------------------
# RUN ANALOGUE SEARCH
# ---------------------------
results = []

for ticker, series in universe.items():

    # Compute MASS distance profile
    dp = stumpy.mass(query, series)
    best_idx = np.argmin(dp)
    best_dist = dp[best_idx]

    # Ensure forward window fits
    if best_idx + WINDOW + FORWARD <= len(series):
        matched_window = series[best_idx:best_idx + WINDOW]
        future = series[best_idx + WINDOW : best_idx + WINDOW + FORWARD]

        ticker_dates = date_map[ticker]
        start_date = ticker_dates[best_idx]
        end_date   = ticker_dates[best_idx + WINDOW - 1]

        results.append(
            (ticker, matched_window, future, best_dist, start_date, end_date)
        )

# Keep top K lowest distances
results = sorted(results, key=lambda x: x[3])[:TOP_K]

matched_windows = np.array([r[1] for r in results])
future_paths = np.array([r[2] for r in results])

mean_hist = matched_windows.mean(axis=0)
median_hist = np.median(matched_windows, axis=0)

mean_forward = future_paths.mean(axis=0)
median_forward = np.median(future_paths, axis=0)


# ---------------------------
# PRINT RESULTS
# ---------------------------
print("\nTop Matches:")
for (ticker, _, _, dist, start_date, end_date) in results:
    print(f"  {ticker:<8} | {start_date} → {end_date} | dist={dist:.4f}")


# ---------------------------
# MATPLOTLIB GRAPH
# ---------------------------
plt.figure(figsize=(12, 6))

# Light gray: all historical matches
for w in matched_windows:
    plt.plot(range(WINDOW), w, color="lightgray", alpha=0.4)

# Black = mean of histories
plt.plot(range(WINDOW), mean_hist, color="black", linewidth=2, label="Mean (History)")

# Blue = median of histories
plt.plot(range(WINDOW), median_hist, color="blue", linewidth=2, linestyle="--",
         label="Median (History)")

# Light gray: all forward paths
for f in future_paths:
    plt.plot(range(WINDOW, WINDOW + FORWARD), f, color="lightgray", alpha=0.4)

# Red forward projections
plt.plot(
    range(WINDOW, WINDOW + FORWARD),
    mean_forward,
    color="red", linewidth=3, label="Mean Forward"
)
plt.plot(
    range(WINDOW, WINDOW + FORWARD),
    median_forward,
    color="red", linewidth=2, linestyle="--", label="Median Forward"
)

plt.title(f"Analogue Projection for {target_ticker}\nTop {TOP_K} Historical Matches")
plt.xlabel("Bars (Aligned Timeline)")
plt.ylabel("Price")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
