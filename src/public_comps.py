# src/public_comps.py
# Robust public-comps loader that prefers a BME Growth CSV from secrets
# and falls back to data/public_comps.csv. Computes EV/Revenue with
# multiple safe fallbacks and normalizes sectors to your Sifted taxonomy.

from pathlib import Path
import pandas as pd
import os

# Streamlit may not exist when running locally; import lazily
try:
    import streamlit as st  # type: ignore
except Exception:
    class _Dummy:
        secrets = {}
    st = _Dummy()  # type: ignore

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SIFTED_SECTORS = {
    "fintech": "Fintech",
    "b2b saas": "B2B SaaS",
    "saas": "B2B SaaS",
    "software": "B2B SaaS",
    "climate": "Climate",
    "climate tech": "Climate",
    "climatetech": "Climate",
    "consumer": "Consumer",
    "healthtech": "Healthtech",
    "health tech": "Healthtech",
    "medtech": "Healthtech",
    "deeptech": "Deeptech",
    "deep tech": "Deeptech",
    "ai-native": "AI-native",
    "ai native": "AI-native",
    "ai": "AI-native",
}

def _norm_sector(s: pd.Series) -> pd.Series:
    if s is None:
        return s
    key = s.astype(str).str.strip().str.lower()
    return key.map(SIFTED_SECTORS).fillna(s)

def _first_nonnull(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for c in cols:
        if c in df.columns:
            tmp = df[c]
            out = out.where(out.notna(), tmp)
    return out

def _to_num(df: pd.DataFrame, cols: list[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def _compute_ev_rev(df: pd.DataFrame) -> pd.DataFrame:
    # Try EV first, fall back to Market Cap if EV missing.
    ev = _first_nonnull(df, ["enterprise_value", "ev", "enterprisevalue"])
    mcap = _first_nonnull(df, ["market_cap", "marketcap", "mkt_cap"])
    revenue = _first_nonnull(df, ["revenue_ttm", "revenue_total", "revenue", "sales_ttm"])

    # numeric
    tmp = pd.DataFrame({"ev": ev, "mcap": mcap, "revenue": revenue})
    _to_num(tmp, ["ev", "mcap", "revenue"])

    method = pd.Series(pd.NA, index=df.index, dtype="string")
    ev_rev = pd.Series(pd.NA, index=df.index, dtype="Float64")

    # Method A: EV / Revenue
    mask_a = tmp["ev"].notna() & tmp["revenue"].notna() & (tmp["revenue"] > 0)
    ev_rev[mask_a] = tmp.loc[mask_a, "ev"] / tmp.loc[mask_a, "revenue"]
    method[mask_a] = "EV/Revenue"

    # Method B: MarketCap / Revenue (only where A unavailable)
    mask_b = ev_rev.isna() & tmp["mcap"].notna() & tmp["revenue"].notna() & (tmp["revenue"] > 0)
    ev_rev[mask_b] = tmp.loc[mask_b, "mcap"] / tmp.loc[mask_b, "revenue"]
    method[mask_b] = "MktCap/Revenue"

    out = df.copy()
    out["ev_to_revenue"] = ev_rev
    out["ev_rev_method"] = method
    return out

def _read_bme_growth_csv() -> pd.DataFrame | None:
    # Prefer Streamlit secrets; else environment variable
    url = ""
    try:
        url = (st.secrets.get("BME_GROWTH_CSV_URL") or "").strip()
    except Exception:
        url = os.getenv("BME_GROWTH_CSV_URL", "").strip()

    if not url:
        return None

    try:
        raw = pd.read_csv(url)
    except Exception:
        return None

    if raw is None or raw.empty:
        return None

    # Standardize column names to snake_case for resilience
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]

    # Harmonize minimum fields
    # Expected possibilities:
    # - ticker, name, sector, country
    # - enterprise_value / market_cap
    # - revenue_ttm / revenue_total
    # If your sheet uses different headers, add them here.
    needed_any = set(raw.columns)
    if "ticker" not in needed_any:
        # try symbol or isin as ticker
        guess = _first_nonnull(raw, ["symbol", "isin"])
        raw["ticker"] = guess

    df = raw.copy()
    df["sector"] = _norm_sector(_first_nonnull(df, ["sector", "industry", "vertical"]))
    df["country"] = _first_nonnull(df, ["country", "listing_country", "hq_country"])
    df = _compute_ev_rev(df)

    # Keep a clean subset plus useful metadata
    keep = [c for c in [
        "ticker", "name", "sector", "country",
        "enterprise_value", "market_cap",
        "revenue_ttm", "revenue_total",
        "ev_to_revenue", "ev_rev_method",
    ] if c in df.columns]
    df = df[keep].drop_duplicates()
    df["_source"] = "bme_growth_csv"
    return df

def _read_local_csv() -> pd.DataFrame:
    p = DATA_DIR / "public_comps.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "sector" in df.columns:
        df["sector"] = _norm_sector(df["sector"])
    df["_source"] = "local_public_csv"
    return df

def load_public_comps() -> pd.DataFrame:
    """Main entry point used by the Streamlit app."""
    # 1) Try BME Growth CSV from secrets
    bme = _read_bme_growth_csv()
    if isinstance(bme, pd.DataFrame) and not bme.empty:
        return bme

    # 2) Fall back to local CSV committed in repo
    local = _read_local_csv()
    if not local.empty:
        return local

    # 3) Last resort: empty frame
    return pd.DataFrame(columns=[
        "ticker","name","sector","country","ev_to_revenue","ev_rev_method","_source"
    ])
