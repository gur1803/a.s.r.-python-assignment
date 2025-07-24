# ─────────────────────────  controller.py  ───────────────────────────────────
"""Controller layer: interface for user interaction."""
from model import Portfolio
import view


def main():
    pf = Portfolio()
    pf.load("portfolio.json")

    print("")
    print("Welcome to the Investment Portfolio Tracker!")
    print("Type 'help' for commands.")
    print("")

    while True:
        cmd = input("\nCommand (add, remove, portfolio, graph, simulate, quit): ").strip().lower()

        if cmd in ("quit", "exit"):
            pf.save("portfolio.json")
            print("Saved & exiting…")
            break

        # —— Add asset ————————————————————————————
        elif cmd == "add":
            tkr = input("  Ticker symbol: ").strip().upper()
            qty = float(input("  Quantity: "))
            price = float(input("  Purchase price: "))
            success = pf.add_asset(tkr, qty, price)
            if success:
                pf.save("portfolio.json")
                print(f"  Added {tkr}. Sector & asset-class auto-populated.")
            else:
                print("  Ticker already exists.")
        elif cmd== "remove":
            # Remove an asset by ticker
            ticker = input("  Enter ticker symbol to remove: ").strip()
            success = pf.remove_asset(ticker)
            if success:
                pf.save("portfolio.json")
                print(f"  Removed {ticker.upper()} from portfolio.")
            else:
                print(f"  Ticker '{ticker}' not found in portfolio.")

        # —— Portfolio / metrics display ——————————————
        elif cmd == "portfolio":
            print("")
            pf.update_prices()
            pf.refresh_fundamentals()
            pf.update_risk_metrics()
            view.display_portfolio(pf)

        # —— Graph ————————————————————————————————
        elif cmd == "graph":
            tickers_input = input("  Tickers (comma, or 'all'): ").strip()
            tickers = [a.ticker for a in pf.assets] if tickers_input.lower() == "all" else [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
            if not tickers:
                print("  No tickers provided.")
                continue
            period = input("  Time-frame (1d,5d,1mo,3mo,1y,ytd,max) [1y]: ").strip().lower() or "1y"
            data = pf.get_historical_data(tickers, period)
            view.plot_price_history(data, period)

        # —— Simulation ——————————————————————————————
        elif cmd == "simulate":
            if not pf.assets:
                print("Portfolio is empty. Add assets before running a simulation.")
                continue

            years = int(input("  Years to simulate [15]: ") or 15)
            paths = int(input("  # Monte-Carlo paths [100000]: ") or 100000)

            # ── Ask user for risk-free rate ─────────────────────────────────
            while True:
                rf_str = input("  Enter annual risk-free rate in % (e.g. 3 for 3%): ").strip()
                try:
                    risk_free_rate = float(rf_str) / 100
                    if risk_free_rate < -1:
                        raise ValueError
                    break
                except ValueError:
                    print("  Please enter a valid number (≥ -100).")

            # ── Ask whether to adjust for inflation ────────────────────────
            inflation_rate = None
            infl_choice = input("  Add inflation-adjusted column? (y/N): ").strip().lower()
            if infl_choice == "y":
                while True:
                    inf_str = input("    Enter expected annual inflation in %: ").strip()
                    try:
                        inflation_rate = float(inf_str) / 100
                        if inflation_rate < -1:
                            raise ValueError
                        break
                    except ValueError:
                        print("    Please enter a valid number (≥ -100).")

            pf.update_prices()
            res = pf.simulate(years=years, num_paths=paths)
            if res:
                view.display_simulation_results(res, years=years, risk_free_rate=risk_free_rate, inflation_rate=inflation_rate)
            else:
                print("  Simulation failed - check data.")

        elif cmd in ("help", "h", "?"):
            print("Commands: add, remove, portfolio, graph, simulate, quit")
        else:
            print("Unknown command. Type 'help'.")


if __name__ == "__main__":
    main()

# ─────────────────────────  end controller.py  ─────────────────────────