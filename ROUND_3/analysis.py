"""
IMC Prosperity 4 - Round 3 Comprehensive Data Analysis
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import skew, kurtosis, norm
import math

warnings.filterwarnings("ignore")

BASE = "/Users/gabrielkatri/Desktop/Projets/IMC Prosperity 4/ROUND_3"
OUT  = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

DAYS = [0, 1, 2]

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def load_prices():
    frames = []
    for d in DAYS:
        f = os.path.join(BASE, f"prices_round_3_day_{d}.csv")
        df = pd.read_csv(f, sep=";")
        df["day"] = d
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["global_ts"] = df["day"] * 1_000_000 + df["timestamp"]
    return df

def load_trades():
    frames = []
    for d in DAYS:
        f = os.path.join(BASE, f"trades_round_3_day_{d}.csv")
        df = pd.read_csv(f, sep=";")
        df["day"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def savefig(name):
    path = os.path.join(OUT, name)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")
    print(f"  saved → {path}")

def returns(series):
    r = series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return r

def log_returns(series):
    lr = np.log(series / series.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    return lr

def strike_from_name(name):
    """VEV_5400 → 5400"""
    try:
        return int(name.split("_")[1])
    except Exception:
        return None

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if sigma <= 0 or T <= 0 or S <= 0:
        return np.nan
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

def implied_vol(C_mkt, S, K, T, r=0.0, tol=1e-4, max_iter=200):
    """Newton-bisection IV solver."""
    if C_mkt <= 0 or S <= 0 or T <= 0:
        return np.nan
    intrinsic = max(S - K, 0.0)
    if C_mkt < intrinsic - tol:
        return np.nan
    lo, hi = 1e-6, 10.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        price = bs_call_price(S, K, T, r, mid)
        if abs(price - C_mkt) < tol:
            return mid
        if price < C_mkt:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

# ─────────────────────────────────────────────────────────────
# 1. INVENTORY
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("1. INVENTORY")
print("="*70)

prices_raw = load_prices()
trades_raw = load_trades()

inventory_lines = []
for fname, df in [
    ("prices_round_3_day_0.csv", pd.read_csv(os.path.join(BASE,"prices_round_3_day_0.csv"), sep=";")),
    ("prices_round_3_day_1.csv", pd.read_csv(os.path.join(BASE,"prices_round_3_day_1.csv"), sep=";")),
    ("prices_round_3_day_2.csv", pd.read_csv(os.path.join(BASE,"prices_round_3_day_2.csv"), sep=";")),
    ("trades_round_3_day_0.csv", pd.read_csv(os.path.join(BASE,"trades_round_3_day_0.csv"), sep=";")),
    ("trades_round_3_day_1.csv", pd.read_csv(os.path.join(BASE,"trades_round_3_day_1.csv"), sep=";")),
    ("trades_round_3_day_2.csv", pd.read_csv(os.path.join(BASE,"trades_round_3_day_2.csv"), sep=";")),
]:
    ts_col = "timestamp" if "timestamp" in df.columns else None
    ts_min = df[ts_col].min() if ts_col else "N/A"
    ts_max = df[ts_col].max() if ts_col else "N/A"
    missing = df.isnull().sum().sum()
    line = (f"  {fname:<40} rows={len(df):>7,}  "
            f"ts=[{ts_min}, {ts_max}]  missing={missing:,}  "
            f"cols={list(df.columns)}")
    print(line)
    inventory_lines.append(line)

# ─────────────────────────────────────────────────────────────
# 2. PRODUCTS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("2. PRODUCTS")
print("="*70)

all_products = sorted(prices_raw["product"].unique())
delta1  = [p for p in all_products if p in ("HYDROGEL_PACK", "VELVETFRUIT_EXTRACT")]
options = sorted([p for p in all_products if p.startswith("VEV_")])

print(f"  Delta-1 products : {delta1}")
print(f"  Options (VEV_*)  : {options}")
print(f"  All products     : {all_products}")

# ─────────────────────────────────────────────────────────────
# 3. PER-PRODUCT PRICE ANALYSIS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("3. PER-PRODUCT PRICE ANALYSIS")
print("="*70)

stats_records = []

for product in all_products:
    print(f"\n  [{product}]")
    sub = prices_raw[prices_raw["product"] == product].sort_values("global_ts")
    mid = sub["mid_price"].dropna()
    ts  = sub["global_ts"].loc[mid.index]

    if len(mid) < 10:
        print("    Skipping – too few rows.")
        continue

    r = log_returns(mid)

    rec = {
        "product"  : product,
        "n_obs"    : len(mid),
        "mean_mid" : mid.mean(),
        "std_mid"  : mid.std(),
        "min_mid"  : mid.min(),
        "max_mid"  : mid.max(),
        "mean_ret" : r.mean(),
        "std_ret"  : r.std(),
        "skew_ret" : skew(r.dropna()),
        "kurt_ret" : kurtosis(r.dropna()),
    }
    stats_records.append(rec)

    # ── Plot: price + rolling vol + autocorrelation
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(ts.values, mid.values, lw=0.7, color="steelblue")
    ax1.set_title(f"{product} – Mid Price", fontsize=12)
    ax1.set_xlabel("Global Timestamp")
    ax1.set_ylabel("Mid Price")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(r.index, r.values, lw=0.5, color="gray", alpha=0.7)
    ax2.set_title("Log Returns")
    ax2.set_ylabel("Log Return")

    ax3 = fig.add_subplot(gs[1, 1])
    rolling_vol = r.rolling(50).std() * np.sqrt(50)
    ax3.plot(rolling_vol.index, rolling_vol.values, lw=0.7, color="tomato")
    ax3.set_title("Rolling Volatility (window=50)")
    ax3.set_ylabel("Volatility")

    ax4 = fig.add_subplot(gs[2, 0])
    lags = range(1, 16)
    acf_vals = [r.autocorr(lag=l) for l in lags]
    ax4.bar(list(lags), acf_vals, color="steelblue", alpha=0.8)
    ax4.axhline(0, color="black", lw=0.8)
    ax4.axhline(1.96/np.sqrt(len(r)), color="red", lw=0.8, ls="--", label="±95% CI")
    ax4.axhline(-1.96/np.sqrt(len(r)), color="red", lw=0.8, ls="--")
    ax4.set_title("Return Autocorrelation (lags 1–15)")
    ax4.set_xlabel("Lag")
    ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.hist(r.dropna(), bins=80, color="steelblue", alpha=0.7, density=True)
    xr = np.linspace(r.min(), r.max(), 200)
    ax5.plot(xr, norm.pdf(xr, r.mean(), r.std()), color="red", lw=1.5, label="Normal")
    ax5.set_title("Return Distribution")
    ax5.set_xlabel("Log Return")
    ax5.legend(fontsize=8)

    fig.suptitle(f"{product} – Full Analysis", fontsize=14, fontweight="bold")
    savefig(f"3_price_{product}.png")

stats_df = pd.DataFrame(stats_records)
print("\n  Return Statistics Summary:")
print(stats_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 4. SPREAD & ORDER BOOK
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("4. SPREAD & ORDER BOOK")
print("="*70)

for product in all_products:
    sub = prices_raw[prices_raw["product"] == product].sort_values("global_ts").copy()
    if "bid_price_1" not in sub.columns or "ask_price_1" not in sub.columns:
        print(f"  {product}: no bid/ask columns – skipping.")
        continue

    sub = sub.dropna(subset=["bid_price_1", "ask_price_1"])
    if len(sub) < 10:
        continue

    sub["spread_abs"] = sub["ask_price_1"] - sub["bid_price_1"]
    sub["spread_pct"] = sub["spread_abs"] / sub["mid_price"] * 100
    sub["imbalance"] = (sub["bid_volume_1"].fillna(0) - sub["ask_volume_1"].fillna(0)) / \
                       (sub["bid_volume_1"].fillna(0) + sub["ask_volume_1"].fillna(1))

    print(f"\n  {product}: spread mean={sub['spread_abs'].mean():.4f}  "
          f"spread_pct mean={sub['spread_pct'].mean():.4f}%")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    axes[0].plot(sub["global_ts"].values, sub["spread_abs"].values, lw=0.6, color="darkorange")
    axes[0].set_title(f"{product} – Absolute Spread over Time")
    axes[0].set_ylabel("Spread (absolute)")

    axes[1].plot(sub["global_ts"].values, sub["imbalance"].values, lw=0.6, color="purple")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("Bid-Ask Imbalance ((bid_vol - ask_vol)/(bid_vol + ask_vol))")
    axes[1].set_ylabel("Imbalance")

    axes[2].hist(sub["spread_abs"].dropna(), bins=60, color="darkorange", alpha=0.8, density=True)
    axes[2].set_title("Spread Distribution")
    axes[2].set_xlabel("Spread")

    fig.suptitle(f"{product} – Order Book Analysis", fontsize=13, fontweight="bold")
    savefig(f"4_spread_{product}.png")

# ─────────────────────────────────────────────────────────────
# 5. CROSS-PRODUCT ANALYSIS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("5. CROSS-PRODUCT ANALYSIS")
print("="*70)

# Build a wide return matrix aligned by timestamp
mid_series = {}
for product in all_products:
    sub = prices_raw[prices_raw["product"] == product].sort_values("global_ts")
    s = sub.set_index("global_ts")["mid_price"]
    s = s[~s.index.duplicated(keep="first")]
    mid_series[product] = s

common_idx = None
for s in mid_series.values():
    if common_idx is None:
        common_idx = s.index
    else:
        common_idx = common_idx.union(s.index)

wide = pd.DataFrame({p: mid_series[p].reindex(common_idx) for p in all_products})
wide_ffill = wide.ffill()
ret_wide = wide_ffill.pct_change().replace([np.inf, -np.inf], np.nan)

corr = ret_wide.corr()
print("\n  Return Correlation Matrix:")
print(corr.to_string())

fig, ax = plt.subplots(figsize=(max(8, len(all_products)), max(6, len(all_products)-1)))
im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(corr)))
ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(corr.index, fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, ax=ax)
ax.set_title("Return Correlation Matrix – All Products", fontsize=12)
savefig("5a_correlation_matrix.png")

# Cross-correlation VEV vs VEF
if "VELVETFRUIT_EXTRACT" in all_products:
    vef_ret = ret_wide["VELVETFRUIT_EXTRACT"].dropna()
    lags_range = range(-10, 11)

    for opt in options:
        if opt not in ret_wide.columns:
            continue
        opt_ret = ret_wide[opt].dropna()
        shared = vef_ret.index.intersection(opt_ret.index)
        if len(shared) < 30:
            continue
        xcorr = [vef_ret.loc[shared].corr(opt_ret.loc[shared].shift(-lag)) for lag in lags_range]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(list(lags_range), xcorr, color="teal", alpha=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(1.96/np.sqrt(len(shared)), color="red", lw=0.8, ls="--", label="±95% CI")
        ax.axhline(-1.96/np.sqrt(len(shared)), color="red", lw=0.8, ls="--")
        ax.set_title(f"Cross-Correlation: VELVETFRUIT_EXTRACT vs {opt} (lag = VEF leads +)")
        ax.set_xlabel("Lag (positive = VEF leads)")
        ax.legend(fontsize=8)
        savefig(f"5b_xcorr_VEF_{opt}.png")

    # Scatter plots
    for opt in options:
        if opt not in wide_ffill.columns:
            continue
        shared = wide_ffill["VELVETFRUIT_EXTRACT"].dropna().index.intersection(
                 wide_ffill[opt].dropna().index)
        if len(shared) < 10:
            continue
        x = wide_ffill.loc[shared, "VELVETFRUIT_EXTRACT"]
        y = wide_ffill.loc[shared, opt]

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(x, y, s=2, alpha=0.3, color="navy")
        ax.set_xlabel("VELVETFRUIT_EXTRACT Price")
        ax.set_ylabel(f"{opt} Price")
        ax.set_title(f"Scatter: VEF Price vs {opt} Price")
        savefig(f"5c_scatter_VEF_{opt}.png")

# ─────────────────────────────────────────────────────────────
# 6. OPTIONS-SPECIFIC
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("6. OPTIONS-SPECIFIC ANALYSIS")
print("="*70)

# Assume T = remaining time fraction; competition is day-based
# Days: 0,1,2 → pretend total horizon = 3 days, T decreases
# We'll use a simple proxy: T = (3 - day) / 3 for each timestamp
# (just for qualitative IV shape – no exact expiry stated)

for opt in options:
    strike = strike_from_name(opt)
    if strike is None:
        continue

    sub_opt = prices_raw[prices_raw["product"] == opt].sort_values("global_ts").copy()
    sub_vef = prices_raw[prices_raw["product"] == "VELVETFRUIT_EXTRACT"].sort_values("global_ts").copy()

    if len(sub_opt) < 5 or len(sub_vef) < 5:
        continue

    sub_vef_idx = sub_vef.set_index("global_ts")["mid_price"]
    sub_vef_idx = sub_vef_idx[~sub_vef_idx.index.duplicated(keep="first")]

    # Align
    opt_ts = sub_opt["global_ts"].values
    vef_at_opt = sub_vef_idx.reindex(opt_ts, method="nearest").values
    opt_price  = sub_opt["mid_price"].values

    intrinsic  = np.maximum(vef_at_opt - strike, 0)
    time_value = opt_price - intrinsic

    print(f"\n  {opt} (K={strike}): opt_price mean={np.nanmean(opt_price):.3f}  "
          f"intrinsic mean={np.nanmean(intrinsic):.3f}  "
          f"time_value mean={np.nanmean(time_value):.3f}")

    # IV computation (sample every 100 rows for speed)
    ivs = []
    sample_idx = range(0, len(sub_opt), max(1, len(sub_opt)//300))
    for i in sample_idx:
        day_i = sub_opt.iloc[i]["day"]
        T = max((3 - float(day_i)) / 252, 1/252)  # rough annualised
        S = float(vef_at_opt[i]) if not np.isnan(vef_at_opt[i]) else np.nan
        C = float(opt_price[i])  if not np.isnan(opt_price[i])  else np.nan
        if S and C and S > 0 and C >= 0:
            iv = implied_vol(C, S, strike, T, r=0.0)
        else:
            iv = np.nan
        ivs.append(iv)

    iv_arr = np.array(ivs)
    iv_ts  = sub_opt.iloc[list(sample_idx)]["global_ts"].values

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))

    axes[0].plot(opt_ts, opt_price, lw=0.7, color="steelblue", label="Option Price")
    axes[0].plot(opt_ts, intrinsic,  lw=0.7, color="red",       label="Intrinsic Value")
    axes[0].set_title(f"{opt} (K={strike}) – Price vs Intrinsic Value")
    axes[0].legend(fontsize=8)

    axes[1].plot(opt_ts, time_value, lw=0.7, color="purple")
    axes[1].axhline(0, color="black", lw=0.8, ls="--")
    axes[1].set_title("Time Value = Option Price − Intrinsic Value")
    axes[1].set_ylabel("Time Value")

    valid = ~np.isnan(iv_arr)
    if valid.sum() > 3:
        axes[2].scatter(iv_ts[valid], iv_arr[valid], s=5, color="green", alpha=0.6)
        axes[2].set_title("Implied Volatility (Black-Scholes, call)")
        axes[2].set_ylabel("IV")
    else:
        axes[2].set_title("Implied Volatility – insufficient data")

    fig.suptitle(f"{opt} – Options Analysis", fontsize=13, fontweight="bold")
    savefig(f"6_options_{opt}.png")

# ─────────────────────────────────────────────────────────────
# 7. BOT BEHAVIOR
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("7. BOT BEHAVIOR")
print("="*70)

# Trades columns: timestamp;buyer;seller;symbol;currency;price;quantity
trades = trades_raw.copy()
print(f"  Trade columns: {list(trades.columns)}")

# Check for counterparty info
has_buyer  = "buyer"  in trades.columns and trades["buyer"].notna().any()
has_seller = "seller" in trades.columns and trades["seller"].notna().any()

if not has_buyer and not has_seller:
    print("  No counterparty info in trades – skipping bot behavior.")
else:
    # Melt buyer/seller into single 'counterparty' + 'side' perspective
    rows = []
    for _, row in trades.iterrows():
        if pd.notna(row.get("buyer")) and str(row.get("buyer","")).strip():
            rows.append({"cp": row["buyer"], "side": "buy", "symbol": row["symbol"],
                         "qty": row["quantity"], "price": row["price"], "day": row["day"]})
        if pd.notna(row.get("seller")) and str(row.get("seller","")).strip():
            rows.append({"cp": row["seller"], "side": "sell", "symbol": row["symbol"],
                         "qty": row["quantity"], "price": row["price"], "day": row["day"]})

    cp_df = pd.DataFrame(rows)

    if len(cp_df) == 0:
        print("  No named counterparties found.")
    else:
        vol_by_cp = cp_df.groupby("cp")["qty"].agg(["sum","mean","count"]).rename(
            columns={"sum":"total_vol","mean":"avg_size","count":"n_trades"}).sort_values("total_vol", ascending=False)
        print("\n  Volume by Counterparty:")
        print(vol_by_cp.to_string())

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        top = vol_by_cp.head(20)
        axes[0].barh(top.index, top["total_vol"], color="steelblue")
        axes[0].set_title("Total Volume by Counterparty (top 20)")
        axes[0].set_xlabel("Total Volume")
        axes[1].barh(top.index, top["avg_size"], color="salmon")
        axes[1].set_title("Average Trade Size by Counterparty (top 20)")
        axes[1].set_xlabel("Avg Size")
        savefig("7_bot_behavior.png")

        # PnL estimate per counterparty:
        # For each counterparty, sum(buy_value) - sum(sell_value)
        cp_df["signed_value"] = cp_df.apply(
            lambda r: -r["price"]*r["qty"] if r["side"]=="buy" else r["price"]*r["qty"], axis=1)
        pnl = cp_df.groupby("cp")["signed_value"].sum().sort_values()
        print("\n  PnL estimate per counterparty (cash flow proxy):")
        print(pnl.to_string())

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["green" if v >= 0 else "red" for v in pnl.values]
        ax.barh(pnl.index, pnl.values, color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title("PnL Estimate per Counterparty (cash flow proxy)")
        ax.set_xlabel("Net Cash Flow (positive = sold more than bought)")
        savefig("7_bot_pnl.png")

# ─────────────────────────────────────────────────────────────
# 8. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("8. FINAL SUMMARY")
print("="*70)

summary_lines = []
summary_lines.append("=" * 70)
summary_lines.append("IMC PROSPERITY 4 – ROUND 3 ANALYSIS SUMMARY")
summary_lines.append("=" * 70)
summary_lines.append("")

summary_lines.append("FILES ANALYSED:")
for l in inventory_lines:
    summary_lines.append(l)

summary_lines.append("")
summary_lines.append(f"PRODUCTS IDENTIFIED:")
summary_lines.append(f"  Delta-1 : {delta1}")
summary_lines.append(f"  Options : {options}")

summary_lines.append("")
summary_lines.append("RETURN STATISTICS (log-returns):")
summary_lines.append(stats_df.to_string(index=False))

summary_lines.append("")
summary_lines.append("CORRELATION MATRIX:")
summary_lines.append(corr.to_string())

summary_lines.append("")
summary_lines.append("KEY FINDINGS & ANOMALIES:")

# Auto-flag extreme kurtosis
for _, row in stats_df.iterrows():
    k = row.get("kurt_ret", 0)
    sk = row.get("skew_ret", 0)
    summary_lines.append(f"  {row['product']}: skew={sk:.3f}  kurt={k:.3f}", )
    if abs(k) > 5:
        summary_lines.append(f"    ⚠ Fat tails (|excess kurtosis|={k:.1f}) – non-normal returns")
    if abs(sk) > 1:
        summary_lines.append(f"    ⚠ Skewed returns ({sk:.2f}) – directional drift present")

# Flag high option time value
for opt in options:
    strike = strike_from_name(opt)
    sub_opt = prices_raw[prices_raw["product"] == opt]
    sub_vef = prices_raw[prices_raw["product"] == "VELVETFRUIT_EXTRACT"]
    if len(sub_opt) == 0 or len(sub_vef) == 0:
        continue
    avg_opt = sub_opt["mid_price"].mean()
    avg_vef = sub_vef["mid_price"].mean()
    intrinsic = max(avg_vef - strike, 0)
    tv = avg_opt - intrinsic
    summary_lines.append(f"  {opt}: avg_price={avg_opt:.3f}  intrinsic≈{intrinsic:.2f}  "
                          f"time_value≈{tv:.3f}")
    if tv < -0.5:
        summary_lines.append(f"    ⚠ NEGATIVE time value on average – possible mispricing / arb")
    if tv > avg_opt * 0.5 and tv > 1:
        summary_lines.append(f"    ★ High time value relative to option price – vol premium")

# Correlation flags
for c in corr.columns:
    if c == "VELVETFRUIT_EXTRACT":
        continue
    val = corr.loc["VELVETFRUIT_EXTRACT", c] if "VELVETFRUIT_EXTRACT" in corr.index and c in corr.columns else None
    if val is not None:
        if abs(val) > 0.5:
            summary_lines.append(f"  ★ Strong return correlation: VELVETFRUIT_EXTRACT ↔ {c} = {val:.3f}")
        elif abs(val) < 0.05:
            summary_lines.append(f"  ⚠ Near-zero return correlation: VELVETFRUIT_EXTRACT ↔ {c} = {val:.3f}")

summary_lines.append("")
summary_lines.append("PLOTS SAVED:")
for fn in sorted(os.listdir(OUT)):
    if fn.endswith(".png"):
        summary_lines.append(f"  {fn}")

summary_lines.append("")
summary_lines.append("END OF REPORT")

summary_text = "\n".join(summary_lines)
print(summary_text)

summary_path = os.path.join(OUT, "analysis_summary.txt")
with open(summary_path, "w") as f:
    f.write(summary_text)
print(f"\n  Summary saved → {summary_path}")
