# Blog 9 · Does updating a stock model every month improve its picks?

## Reproduce the results

From the BLOG9 directory, install the Python packages and run:

```sh
python -m pip install -r requirements.txt
python code/analysis.py
```

The script reads the saved Yahoo Finance snapshots in data/ and needs no network access once the packages are installed. The ticker list comes from the classroom R script. Each JSON file is the daily Yahoo chart API response for that ticker, with timestamps, volume, price fields and split/dividend-adjusted closing prices. The query requested January 1, 2022 through October 30, 2026 inclusive. Monthly investments run from January 2024 through October 2026.

## Files

- code/classroom.R preserves the supplied original. code/analysis.py reproduces the model specification in Python and implements the comparison.
- data/tickers.txt lists the classroom universe and SPY. The Yahoo chart endpoint is https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?period1=1640995200&period2=1793404800&interval=1d .
- results/comparison.csv contains pooled forecast MAE in percentage points and arithmetic mean monthly returns in percent. results/monthly_returns.csv gives entry and exit dates and returns in decimal units. results/portfolio.png shows gross monthly returns as percentages. results/_table.md is the forecast-error table included in the article.
- results/featured_returns.csv contains the selected results for the figure.
- index.qmd contains the article. Render with Quarto. The website copy belongs in website1/blog/posts/post9/ with both results/_table.md and results/portfolio.png. From the website root, run `quarto render blog/posts/post9/index.qmd`.

## Design choices

The fixed and expanding models use 300 trees, depth 5, learning rate 0.05, min child weight 10 and gamma 0. We preserve the classroom price-ratio directions, raw price and volume fields and 22-observation volatility window. We index outcomes by their actual end dates and exclude outcomes still unknown at training time. The historical-mean forecast uses identical training eligibility and targets. All forecasts are scored on the same stock-date observations on monthly signal dates.

Each strategy ranks stocks on the first trading day of the month, buys at the close one trading day after the signal and sells at the close 20 trading days after the signal. Holding-period portfolio return is the sum of initial weights times realized stock returns, before costs. All 34 months receive equal weight in the mean. The figure shows September and October 2026 alongside this full-period average. Forecast MAE uses the original classroom target from trading day 1 to trading day 20 after the signal, evaluated on the same 34 signal dates.
