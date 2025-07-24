# Investment Portfolio Tracker – User Instructions Manual

<!-- Instructions_Manual.md -->

Welcome! This manual walks you through installation, day‑to‑day usage and advanced features of the Portfolio Tracker.

## 1. Setup

1. **Install Python 3.10+** if it is not already available:
   *macOS/Linux* – use Homebrew, apt, etc.
   *Windows* – download from [python.org](https://python.org) and tick “Add to PATH”, or use your IDE of choice for python.
2. **Clone or download** the project repository.
3. (Optional) **Create a virtual environment** and activate it.
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
5. (Optional) **Alpha Vantage**: if you have a premium key, paste it into the `_AV_KEY` constant near the top of `model.py`. Otherwise the program will simply use Yahoo Finance.

## 2. Launching the CLI

```bash
python controller.py
```

You will see a welcome banner followed by the prompt:

```
Command (add, remove, portfolio, graph, simulate, quit):
```

Commands are case‑insensitive; `help` shows the list at any time.

## 3. Command Reference

The examples given here are done line by line in the CLI

### `add`

Adds a new holding.

```
Ticker symbol: AAPL
Quantity: 12
Purchase price: 98.17
```

The program fetches *sector* and *asset‑class* automatically.

### `remove`

Deletes an existing holding by ticker.

```
Enter ticker symbol to remove: TSLA
```

### `portfolio`

1. Refreshes live prices.
2. Re‑computes fundamentals and risk metrics.
3. Prints an aligned table plus weight breakdowns.

### `graph`

Plots historical prices for 1‑n tickers and saves an HTML file that opens in your browser.

```
Tickers (comma, or 'all'): AAPL,MSFT,GOOGL
Time‑frame (1d,5d,1mo,3mo,1y,ytd,max) [1y]: 5y
```

### `simulate`

Performs a 15‑year (default) projection of portfolio value using Monte‑Carlo.

```
Years to simulate [15]: 20
# Monte‑Carlo paths [100000]: 250000
Enter annual risk‑free rate in % (e.g. 3 for 3%): 2.7
Add inflation‑adjusted column? (y/N): y
  Enter expected annual inflation in %: 2
```

The results table appears in the console and an interactive histogram is written to `simulation_results.html`.

### `quit` / `exit`

Saves the current state to `portfolio.json` and exits.

## 4. Data File (`portfolio.json`)

The tracker keeps a *single* JSON file in the working directory with the following schema:

```json
{
  "assets": [
    {
      "ticker": "AAPL",
      "quantity": 12,
      "purchase_price": 98.17,
      "sector": "Technology",
      "asset_class": "STOCK"
    }
  ]
}
```

Feel free to edit it manually between sessions.

## 5. Tips & Troubleshooting

| Issue                      | Possible Cause / Fix                                   |
| -------------------------- | ------------------------------------------------------ |
| `ModuleNotFoundError`      | Run `pip install -r requirements.txt`.                 |
| No price data for a ticker | Check spelling; some tickers need “.X” country suffix. |
| Simulation very slow       | Reduce number of paths or ensure multi‑core CPU.       |
| Empty chart opens          | The selected period had no data or API rate limit hit. |

## 6. Extending the Application

* **New data sources** – implement alternative quote fetchers in `model.py`.
* **GUI** – replace the CLI loop in `controller.py` with a Flask or Streamlit front‑end.
* **Analytics** – add Sharpe ratio, VaR, ESG scores, etc., in the Model layer.

---

Happy investing!

<!-- End Instructions_Manual.md -->
