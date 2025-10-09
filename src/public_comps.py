import sys, time
import pandas as pd
import yfinance as yf
from datetime import datetime, UTC

WATCHLIST = "data/public_comps_watchlist.csv"
OUTFILE = "data/public_comps.csv"

def _ttm_revenue_from_quarterlies(t: yf.Ticker):
    try:
        q = getattr(t, "quarterly_income_stmt", None)
        if isinstance(q, pd.DataFrame) and not q.empty:
            candidates = [r for r in q.index if isinstance(r, str) and "revenue" in r.lower()]
            if candidates:
                rev_row = candidates[0]
                vals = pd.to_numeric(q.loc[rev_row].dropna(), errors="coerce").sort_index(ascending=False)
                return vals.iloc[:4].sum() if len(vals) >= 1 else None
    except Exception:
        pass
    return None

def fetch_public_comps():
    rows = []
    wl = pd.read_csv(WATCHLIST, comment='#')

    for _, r in wl.iterrows():
        ticker = r["ticker"]
        sector = r.get("sector")
        country = r.get("country")
        notes = r.get("notes", "")
        method = None
        ev_to_rev = None
        enterprise_value = None
        market_cap = None
        revenue_total = None
        revenue_ttm = None

        try:
            t = yf.Ticker(ticker)

            # Try Method 1: EV / Total Revenue (from .info)
            info = {}
            try:
                info = t.info or {}
            except Exception:
                info = {}

            enterprise_value = info.get("enterpriseValue")
            revenue_total = info.get("totalRevenue")

            if enterprise_value and revenue_total and float(revenue_total) > 0:
                ev_to_rev = float(enterprise_value) / float(revenue_total)
                method = "EV/TotalRevenue"

            # If Method 1 unavailable, try Method 2: MarketCap / TTM revenue (from quarterlies)
            if ev_to_rev is None:
                market_cap = info.get("marketCap")
                if not market_cap:
                    fast = getattr(t, "fast_info", {}) or {}
                    market_cap = fast.get("market_cap")

                revenue_ttm = _ttm_revenue_from_quarterlies(t)
                if market_cap and revenue_ttm and float(revenue_ttm) > 0:
                    ev_to_rev = float(market_cap) / float(revenue_ttm)
                    method = "MktCap/TTMRevenue"

            # Light metadata
            fast = getattr(t, "fast_info", {}) or {}
            price = fast.get("last_price")
            currency = fast.get("currency")
            pe = fast.get("pe_ratio")

            rows.append({
                "ticker": ticker,
                "sector": sector,
                "country": country,
                "notes": notes,
                "enterprise_value": enterprise_value,
                "market_cap": market_cap,
                "revenue_total": revenue_total,
                "revenue_ttm": revenue_ttm,
                "ev_to_revenue": ev_to_rev,
                "ev_rev_method": method,
                "pe_ttm": pe,
                "price": price,
                "currency": currency,
                "data_date": datetime.now(UTC).date().isoformat(),
            })
        except Exception as e:
            rows.append({
                "ticker": ticker,
                "sector": sector,
                "country": country,
                "notes": f"error: {e}",
                "enterprise_value": None,
                "market_cap": None,
                "revenue_total": None,
                "revenue_ttm": None,
                "ev_to_revenue": None,
                "ev_rev_method": None,
                "pe_ttm": None,
                "price": None,
                "currency": None,
                "data_date": datetime.now(UTC).date().isoformat(),
            })

        time.sleep(0.5)  # be polite

    df = pd.DataFrame(rows)
    df.to_csv(OUTFILE, index=False)
    print(f"wrote {OUTFILE} with {len(df)} rows")
    return df

if __name__ == "__main__":
    try:
        fetch_public_comps()
    except KeyboardInterrupt:
        sys.exit(0)
