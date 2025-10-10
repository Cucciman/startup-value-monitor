# 🧮 Startup Value Monitor

**Startup Value Monitor** is an open-source dashboard that tracks how **European startups are valued** in crowdfunding rounds compared to their **public-market peers**.

It introduces a simple, transparent metric — the **Valuation Gap Index (VGI)** — to highlight sector-level valuation trends and identify when startup optimism (or pessimism) diverges from fundamentals.

Built with **Streamlit**, **Python**, and **pandas**, and designed to evolve through community-maintained datasets and open financial APIs.

---

## 📊 Live Demo

🌐 [https://startup-value-monitor.streamlit.app](https://startup-value-monitor.streamlit.app)

The app dynamically visualizes:
- The **Valuation Gap Index (VGI)** by sector
- **Crowdfunding vs Public EV/Revenue ratios**
- Startups contributing to each sector’s median  
- Automatic data updates via **Google Sheets (Crowdfunding)** and **Yahoo Finance (Public Comps)**

---

## 📈 Valuation Gap Index (VGI)

The **Valuation Gap Index (VGI)** is the project’s core benchmark — a reproducible way to measure how startup valuations in crowdfunding rounds compare to similar listed companies.

### 🔹 Definition

$$
\text{VGI} = \frac{\mathrm{Median}\!\left(\text{Crowdfunding }\frac{\mathrm{EV}}{\mathrm{Revenue}}\right)}{\mathrm{Median}\!\left(\text{Public }\frac{\mathrm{EV}}{\mathrm{Revenue}}\right)}
$$

**Fallback (plain text):**  
`VGI = Median(Crowdfunding EV/Revenue) ÷ Median(Public EV/Revenue)`

---

### 🔹 Interpretation

- **VGI = 1.0** → crowdfunding valuations are aligned with public-market multiples.  
- **VGI > 1.0** → startups are priced higher (potential overvaluation).  
- **VGI < 1.0** → startups are priced lower (potential undervaluation).

---

### 🔹 Purpose

The VGI bridges the gap between private and public market valuations using **open data** and a **consistent methodology**.

It helps founders, investors, and policymakers identify where early-stage sentiment diverges from fundamentals — providing a cross-sector view of pricing behaviour in Europe’s innovation economy.

Because both crowdfunding and public data refresh regularly, the index becomes a **living pulse of market sentiment** across sectors such as:

- **Fintech**  
- **Climate Tech**  
- **Healthtech**  
- **AI-native**  
- **B2B SaaS**  
- **Consumer / Deeptech**

---

### 🔹 Methodology Overview

1. **Crowdfunding Data**
   - Aggregated via open community datasets or a shared Google Sheet (CSV format).  
   - Columns include:  
     `startup`, `country`, `sector`, `valuation_pre_money_eur`, `revenue_last_fy_eur`, `round_date`, `platform`, etc.  
   - Optional fields: `arr_eur`, `mrr_eur`, `gmv_eur`, `assumed_take_rate_pct`, `headcount` (used for estimating revenue if missing).

2. **Public Comparables**
   - Fetched via [`yfinance`](https://github.com/ranaroussi/yfinance) for listed peers defined in  
     `data/public_comps_watchlist.csv`.  
   - Calculates **EV/Revenue** using consistent fallbacks:
     - (1) `enterpriseValue / totalRevenue`  
     - (2) `marketCap / trailing 12-month revenue`  
     - (3) Marks as “insufficient” if neither is available.

3. **Normalization**
   - Sector names harmonized (e.g., “AI” → “AI-native”, “Climate” → “Climate Tech”).  
   - Currency and valuation fields standardized to EUR.  
   - Outliers and zeros removed before median computation.

4. **Computation**
   - Median EV/Revenue computed separately for crowdfunding and public datasets per sector.  
   - VGI is the ratio of those two medians.  
   - Visualized via interactive charts and downloadable tables.

---

## 🧠 Why It Matters

> “Valuation transparency should be a public good.”

Early-stage finance is notoriously opaque.  
The **VGI** makes valuation trends **visible, comparable, and verifiable** — empowering founders and investors with a shared, data-driven lens on pricing fairness.

It can help:
- **Founders** benchmark their funding rounds.  
- **Investors** spot sectors trading at unsustainable multiples.  
- **Analysts** monitor sentiment shifts across European innovation verticals.

---

## ⚙️ How Data Updates Work

The app automatically refreshes when new data is available in connected sources.

### Crowdfunding data
- Pulled dynamically from a **public Google Sheet (published as CSV)**  
- Example secret in `.streamlit/secrets.toml`:
  ```toml
  [sources]
  CF_SHEET_CSV = "https://docs.google.com/spreadsheets/d/e/.../pub?output=csv"
