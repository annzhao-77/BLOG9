"""Run from any directory after downloading the price files listed in data/tickers.txt."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
prices = {}
for symbol in (DATA / "tickers.txt").read_text().split():
    raw = json.loads((DATA / f"{symbol}.json").read_text())["chart"]["result"][0]
    dates = pd.to_datetime(raw["timestamp"], unit="s", utc=True).normalize().tz_localize(None)
    prices[symbol] = pd.Series(raw["indicators"]["adjclose"][0]["adjclose"], index=dates)
prices = pd.DataFrame(prices).sort_index()
market = prices["SPY"].dropna()
features = ["open", "high", "low", "close", "volume", "adjusted", "return", "spy_ret",
            "mom_1m", "mom_3m", "mom_6m", "vol_1m", "ret_5d", "excess_ret"]
frames = []
for symbol in prices.columns.drop("SPY"):
    raw = json.loads((DATA / f"{symbol}.json").read_text())["chart"]["result"][0]
    dates = pd.to_datetime(raw["timestamp"], unit="s", utc=True).normalize().tz_localize(None)
    f = pd.DataFrame(raw["indicators"]["quote"][0], index=dates)
    p = prices[symbol].reindex(dates)
    f["adjusted"] = p
    f["return"] = p.pct_change(fill_method=None)
    f["spy_ret"] = market.pct_change()
    for name, lag in [("mom_1m", 21), ("mom_3m", 63), ("mom_6m", 126), ("ret_5d", 5)]:
        f[name] = p.shift(lag) / p.shift(1) - 1
    f["vol_1m"] = f["return"].shift(1).rolling(22).std()
    f["excess_ret"] = f["return"] - f["spy_ret"]
    f["target"] = p.shift(-20) / p.shift(-1) - 1
    f["end"] = dates.to_series().shift(-20)
    f["symbol"] = symbol
    frames.append(f.dropna())
panel = pd.concat(frames).sort_index()
params = dict(objective="reg:squarederror", max_depth=5, eta=.05, min_child_weight=10,
              gamma=0, subsample=1, colsample_bynode=1, tree_method="hist", nthread=4, seed=123)

def fit(train):
    matrix = xgb.DMatrix(train[features], label=train["target"])
    return xgb.train(params, matrix, num_boost_round=300)

fixed = fit(panel[panel["end"] < "2024-01-01"])
signals = market.loc["2024":].groupby(market.loc["2024":].index.to_period("M")).head(1).index
errors, returns = [], []
for day in signals:
    past = panel[panel["end"] < day]
    today = panel.loc[[day]].set_index("symbol")
    matrix = xgb.DMatrix(today[features])
    prediction = pd.DataFrame({
        "Historical mean": past.groupby("symbol")["target"].mean(),
        "Classroom model": fixed.predict(matrix),
        "Monthly update": fit(past).predict(matrix),
    }, index=today.index)
    entry = market.index[market.index.get_loc(day) + 1]
    exit_day = market.index[market.index.get_loc(day) + 20]
    realized = prices.loc[exit_day] / prices.loc[entry] - 1
    row = {"date": exit_day, "entry_date": entry, "SPY": realized["SPY"]}
    for method in prediction:
        errors.extend((day, sym, method, abs(value - today.at[sym, "target"]))
                      for sym, value in prediction[method].items())
        best = prediction[method].nlargest(10).clip(lower=0)
        weights = best / best.sum() if best.sum() else pd.Series(.1, index=best.index)
        selected = realized.loc[weights.index]
        if selected.isna().any():
            raise ValueError(f"Missing holding-period price on {exit_day}")
        row[method] = weights.dot(selected)
    returns.append(row)
monthly = pd.DataFrame(returns).set_index("date")
detail = pd.DataFrame(errors, columns=["date", "symbol", "method", "error"])
mae = detail.groupby("method")["error"].mean() * 100
methods = ["Classroom model", "Monthly update", "Historical mean"]
period_pct = monthly[methods] * 100
summary = pd.concat([mae.rename("MAE_pp"), period_pct.mean().rename("mean_monthly_return_pct")], axis=1)
summary.to_csv(OUT / "comparison.csv")
table = ["| Forecast method | Mean absolute error |", "|---|---:|"]
for method in ["Historical mean", "Classroom model", "Monthly update"]:
    table.append(f"| {method} | {mae[method]:.2f} percentage points |")
(OUT / "_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
monthly.to_csv(OUT / "monthly_returns.csv")
chart = pd.DataFrame([period_pct.iloc[-2], period_pct.iloc[-1], period_pct.mean()],
                     index=["September 2026", "October 2026", f"{len(monthly)}-month average"])
chart.index.name = "Period"
chart.to_csv(OUT / "featured_returns.csv")
ax = chart.plot.bar(figsize=(8, 4.4), color=["#355c80", "#b35b30", "#54806c"], width=.76, rot=0)
for bars in ax.containers:
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
ax.axhline(0, color="#555555", linewidth=.8)
ax.set(xlabel="", ylabel="Return before costs (%)")
ax.margins(y=.25)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.17), ncol=3, frameon=False)
ax.figure.text(.1, .01, "Entry: trading day 1 after the signal. Exit: trading day 20.", fontsize=8)
ax.figure.tight_layout(rect=(0,.05,1,1))
ax.figure.savefig(OUT / "portfolio.png", dpi=160)
print(summary, len(monthly), detail.date.min(), monthly.index.max(), len(detail)//3, flush=True)


