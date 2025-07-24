# _________________________  view.py  _________________________
"""View layer: CLI table & charts."""
from typing import Dict
import plotly.express as px
from plotly.offline import plot as plot_offline
import numpy as np


# _________ Portfolio table ______________________________________________

def _format_pct(x: float) -> str:
    return f"{x*100:>6.2f}%" if x is not None else "   -  "

def _fmt(x: float | None) -> str:
    return f"{x:,.2f}" if x is not None else "-"

def display_portfolio(pf):
    if not pf.assets:
        print("Portfolio is empty.")
        return

    # __ header ________________ 
    hdr = (f"{'Ticker':<8} {'Sector':<14} {'Class':<7} {'Qty':>8} "
           f"{'BuyPx':>10} {'TxValue':>12} {'CurPx':>10} {'Value':>11} "
           f"{'P/E':>8} {'Beta':>8} {'Alpha':>9}")
    print(hdr)
    print("-" * len(hdr))

    # __ rows __________________
    for a in pf.assets:
        tx_val = a.purchase_price * a.quantity
        print(f"{a.ticker:<8} {a.sector[:13]:<14} {a.asset_class:<7} "
              f"{a.quantity:>8.2f} {_fmt(a.purchase_price):>10} {_fmt(tx_val):>12} "
              f"{_fmt(a.current_price):>10} {_fmt(a.market_value()):>11} "
              f"{_fmt(a.pe_ratio):>8} {_fmt(a.beta):>8} {_fmt(a.alpha):>9}")

    print("-" * len(hdr))
    tot = pf.total_value()
    print(f"Total value: {_fmt(tot)}\n")

    # Portfolio‑level metrics
    port = pf.portfolio_metrics()
    if port:
        print(f"Portfolio beta  : {port['beta']:.3f}")
        print(f"Portfolio alpha : {port['alpha']:.5f}")
    # Sector metrics
    sectors = pf.sector_metrics()
    if sectors:
        print("\nSector metrics vs S&P 500:")
        for s, m in sectors.items():
            print(f"  {s:<14} β={m['beta']:.3f}  α={m['alpha']:.5f}")

    # Weights of portfolio based on asset, sector and class
    print("\nWeights by asset:")
    for t, w in pf.asset_weights().items():
        print(f"  {t:<6} {_format_pct(w)}")

    print("\nWeights by Sector:")
    for t, w in pf.sector_weights().items():
        print(f"  {t:<6} {_format_pct(w)}")   

    print("\nWeights by asset class:")
    for t, w in pf.class_weights().items():
        print(f"  {t:<6} {_format_pct(w)}") 

# _________ Price history chart _________________________________________

def plot_price_history(df, period: str):
    if df is None or df.empty:
        print("No data to plot.")
        return

    df_long = df.reset_index().melt(id_vars=df.index.name or "Date", var_name="Ticker", value_name="Price")

    # using qualitative palette - automatic diverse colours for lines in graph
    colours = px.colors.qualitative.Dark24
    fig = px.line(df_long, x=df_long.columns[0], y="Price", color="Ticker",
                  title=f"Price history - last {period}",
                  color_discrete_sequence=colours)

    fig.update_layout(plot_bgcolor="#000", paper_bgcolor="#000", font_color="#fff",
                      hovermode="x unified")
    fig.update_xaxes(gridcolor="#333")
    fig.update_yaxes(gridcolor="#333")

    plot_offline(fig, filename="price_history.html", auto_open=True)
    print("Chart saved to price_history.html")


def display_simulation_results(
    result: dict,
    *,
    years: int,
    risk_free_rate: float,
    inflation_rate: float | None = None,
):
    """
    Print nominal, risk-free-discounted and (optionally) real
    statistics, then plot the histogram.
    """
    if result is None:
        print("No simulation results to display.")
        return

    def pv(value: float, rate: float) -> float:
        return value / ((1 + rate) ** years)

    # __ Prepare number strings ___________________________________________
    fmt = lambda x: f"{x:,.2f}"

    col_nom  = {k: fmt(v) for k, v in {
        "Current": result["current_value"],
        "Mean":    result["mean"],
        "Median":  result["median"],
        "P5":      result["p5"],
        "P95":     result["p95"],
    }.items()}

    col_rf   = {k: fmt(pv(float(v.replace(',', '')), risk_free_rate))
                for k, v in col_nom.items()}

    if inflation_rate is not None:
        col_real = {k: fmt(pv(float(v.replace(',', '')), inflation_rate))
                    for k, v in col_nom.items()}

    # __ Work out column widths dynamically _______________________________
    hdr_nom = f"PV @ {risk_free_rate*100:.2f}%"
    widths  = {
        "Metric":  max(len("Metric"), 10),
        "Nom":     max(len("Nominal"), max(len(s) for s in col_nom.values())),
        "RF":      max(len(hdr_nom),   max(len(s) for s in col_rf.values())),
    }
    if inflation_rate is not None:
        hdr_real = f"Real @ {inflation_rate*100:.2f}%"
        widths["Real"] = max(len(hdr_real),
                             max(len(s) for s in col_real.values()))

    # __ Print table ______________________________________________________
    print(f"\nSimulation {years}-Year Projection (100,000 paths):")

    header_line = (
        f"{'Metric':<{widths['Metric']}} "
        f"{'Nominal':>{widths['Nom']}} "
        f"{hdr_nom:>{widths['RF']}}"
        + (f" {hdr_real:>{widths['Real']}}" if inflation_rate is not None else "")
    )
    print(header_line)
    print("-" * len(header_line))

    for key in ["Current", "Mean", "Median", "P5", "P95"]:
        line = (
            f"{key:<{widths['Metric']}} "
            f"{col_nom[key]:>{widths['Nom']}} "
            f"{col_rf[key]:>{widths['RF']}}"
            + (f" {col_real[key]:>{widths['Real']}}"
               if inflation_rate is not None else "")
        )
        print(line)

    print(f"\nProbability of Loss (final < current): "
          f"{result['prob_loss']*100:.2f}%")

    # __ Histogram __________________________________________________________
    dist = result["distribution"]

    # Ask user how to scale the x-axis
    scale_choice = input(
        "  Histogram x-axis: (t)runcate extreme tail, or (r)egular? [t/r]: "
    ).strip().lower()

    # Determine dataset, scale, title
    if scale_choice == "t":
        xmax = np.percentile(dist, 95)  # keep lower 95 %
        dist_plot = dist[dist <= xmax]
        title = f"Distribution (≤99th pct) - {years} yrs"
    else:
        dist_plot = dist
        title = f"Distribution - {years} yrs"

    # Build figure _____________________________________________________
    fig = px.histogram(
        x=dist_plot,
        nbins=200,
        log_x=False,
        labels={"x": "Nominal Portfolio Value", "y": "Frequency"},
        title=title,
        template="plotly_dark",
    )

    # Vertical dotted line for current value
    fig.add_vline(
        x=result["current_value"],
        line_dash="dash",
        line_color="red",
        annotation_text="Current value",
        annotation_position="top left",
    )

    # Fine‑tune the look
    fig.update_layout(
        plot_bgcolor="#000",
        paper_bgcolor="#000",
        font_color="#fff",
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#333")
    fig.update_yaxes(gridcolor="#333")

    plot_offline(fig, filename="simulation_results.html", auto_open=True)
    print("Simulation results chart saved to simulation_results.html")

# _________________________  end view.py  _________________________