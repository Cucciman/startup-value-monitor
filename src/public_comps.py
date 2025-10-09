import sys, time
import pandas as pd
import yfinance as yf
from datetime import datetime

WATCHLIST = "data/public_comps_watchlist.csv"
OUTFILE = "data/public_comps.csv"

def fetch_ev_revenue(ticker):
    t = yf.Ticker(ticker)
    info = t.info or {}
    # yfinance can be inconsistent; try to compute EV/Revenue if not provided
    ev = info.get("enterpriseValue")
    rev = info.get("totalRevenue") or info.get("revenueQuarterly")  # fallback
    pe = info.get("trailingPE")
    return ev, rev, pe

def main():
    wl = pd.read_csv(WATCHLIST)
    rows = []
    for i, r in wl.iterrows():
        ticker = r["ticker"]
        try:
            ev, rev, pe = fetch_ev_revenue(ticker)
            if ev and rev and rev != 0:
                ev_rev = ev / rev
            else:
                ev_rev = None
            rows.append({
                "ticker": ticker,
                "sector": r["sector"],
                "country": r["country"],
                "ev_eur_est": ev,     # yfinance often in native currency; treat as estimate
                "revenue_ttm_est": rev,
                "ev_to_revenue": ev_rev,
                "pe_ttm": pe,
                "data_date": datetime.utcnow().date().isoformat(),
                "notes": r.get("notes", "")
            })
        except Exception as e:
            rows.append({
                "ticker": ticker, "sector": r["sector"], "country": r["country"],
                "ev_eur_est": None, "revenue_ttm_est": None, "ev_to_revenue": None,
                "pe_ttm": None, "data_date": datetime.utcnow().date().isoformat(),
                "notes": f"error: {e}"
            })
        time.sleep(0.8)  # be gentle
    out = pd.DataFrame(rows)
    out.to_csv(OUTFILE, index=False)
    print(f"wrote {OUTFILE} with {len(out)} rows")

if __name__ == "__main__":
    main()
