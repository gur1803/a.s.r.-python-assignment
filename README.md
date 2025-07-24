# a.s.r.-python-project
<!-- README.md -->

# Investment Portfolio Tracker

This is a command‑line application for monitoring, analysing and stress‑testing an investment portfolio. The tool follows a clean *Model – View – Controller* architecture and was built to satisfy the requirements in the **a.s.r. Vermogensbeheer Portfolio‑Tracker assignment**.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Repository Structure](#repository-structure)
4. [Usage Overview](#usage-overview)
5. [Architecture](#architecture)
6. [Licence](#licence)
7. [Features](#features)

---

## Quick Start

```bash
# 1 — clone the repository
$ git clone https://github.com/gur1803/a.s.r.-python-project.git portfolio‑tracker
$ cd portfolio‑tracker

# 2 — create and activate a virtual environment (optional but recommended)
$ python -m venv .venv
$ source .venv/bin/activate   # PowerShell: .venv\Scripts\Activate.ps1

# 3 — install dependencies
$ pip install -r requirements.txt

# 4 — run the CLI
$ python controller.py
```

The program welcomes you with a prompt. Type `help` to see the built‑in commands.

---

## Installation

### Prerequisites

* Python ≥ 3.10
* Git (only if cloning the repo)

### Dependencies

All runtime dependencies are specified in `requirements.txt` and installed via `pip`:

```
numpy
pandas
scipy
yfinance
alpha‑vantage      # optional – only if you set your API key
arch
plotly
tqdm
```

> **Note:** `alpha‑vantage` is optional. If you leave the API key blank in `model.py`, the program will silently fall back to Yahoo Finance.

---

## Repository Structure

```
portfolio‑tracker/
├── controller.py   # CLI entry‑point (Controller layer)
├── model.py        # business logic & calculations (Model layer)
├── view.py         # formatting & charts (View layer)
├── requirements.txt
├── README.md
├── Instructions_Manual.md
└── LICENSE           
```
---

## Usage Overview

The application stores your portfolio in **`portfolio.json`** in the working directory and automatically reloads it on start‑up. The following commands are available from the prompt:

| Command         | Purpose                                                             |
| --------------- | ------------------------------------------------------------------- |
| `add`           | Add a new asset (ticker, qty, buy price). Sector & class auto‑fill. |
| `remove`        | Delete an existing asset by ticker.                                 |
| `portfolio`     | Refresh live quotes & metrics and print a detailed table.           |
| `graph`         | Plot historical price(s) for one or several tickers.                |
| `simulate`      | 15‑year Monte‑Carlo projection (GARCH) with 100 k paths.            |
| `help`          | Display a short command summary.                                    |
| `quit` / `exit` | Save and close the application.                                     |

For detailed, step‑by‑step instructions see the **User Instructions Manual** in `docs/Instructions_Manual.md`.

---

## Architecture

The code base is organised according to the MVC pattern:

* **Model (`model.py`)** – persistence, market‑data retrieval, risk/return calculations, Monte‑Carlo engine.
* **View (`view.py`)** – ASCII tables and interactive Plotly charts.
* **Controller (`controller.py`)** – CLI loop, user input parsing, orchestrating Model ↔ View.
  This separation makes the program easy to extend (e.g. replace the CLI with a web UI).

---

## Licence

This project is released under the **MIT Licence** – see `LICENCE` file for details.

---

## Features

* Interactive **command‑line interface** with contextual help.
* Automatic **JSON persistence** (`portfolio.json`) on start‑up / exit.
* **Add / remove assets** with *sector*, *asset‑class* and *P/E* auto‑lookup.
* Live **price refresh** via Alpha Vantage (with Yahoo Finance fallback).
* Comprehensive **portfolio table** including:

  * Quantity, buy price, transaction value, current price & market value.
  * Ticker‑level **α (alpha)** & **β (beta)** relative to the S\&P 500.
  * Derived **sector** & **asset‑class** for each holding.
* Calculation of **weights** by asset, sector and asset class.
* Aggregate **portfolio alpha / beta** and **sector‑level** risk metrics.
* **Historical price charts** with Plotly for any selection of tickers & time frames.
* **Monte‑Carlo simulation**:

  * Per‑asset *GARCH(1,1)* return model with drift capped at 30 % p.a.
  * 100 000 paths, multi‑processing for speed, optional chunk size tuning.
  * User‑defined horizon (default 15 years).
  * Outputs mean, median, 5th/95th percentiles, probability of loss.
  * Present‑value columns discounted by user‑supplied **risk‑free rate**.
  * Optional **inflation‑adjusted** (real) present‑value column.
  * Interactive histogram saved to `simulation_results.html`.
* Modular **MVC architecture** – easy to plug in a GUI or REST API later.
* Clean dependency management via `requirements.txt`.

<!-- End README.md -->
