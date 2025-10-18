# app/streamlit_app.py
# Clean MVP — Startup Value Monitor (VGI)
# - Loads crowdfunding from Google Sheet (secrets) -> local CSV -> sample
# - Loads public comps from BME Growth (secrets) -> local CSV -> sample
# - Normalizes sectors (Sifted-style)
# - Computes VGI = (CF EV/Rev) / (Public median EV/Rev per sector)
# - Charts + sector table; diagnostics moved to bottom

from pathlib import Path
from datetime import datetime
import math

import pandas as pd
import altair as alt
import streamlit as st

# ------------------------------- Paths & Constants -------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Sifted-style sector taxonomy normalization
SIFTED_SECTORS = {
    "fintech": "Fintech",
    "payments": "Fintech",
    "insurtech": "Fintech",
    "b2b saas": "B2B SaaS",
    "saas": "B2B SaaS",
    "software": "B2B SaaS",
    "climate": "Climate",
    "climate tech": "Climate",
    "climatetech": "Climate",
    "clean tech": "Climate",
    "cleantech": "Climate",
    "consumer": "Consumer",
    "health": "Healthtech",
    "healthtech": "Healthtech",
    "health tech": "Healthtech",
    "medtech": "Healthtech",
    "deeptech": "Deeptech",
    "deep tech": "Deeptech",
    "semiconductors": "Deeptech",
    "aerospace": "Deeptech",
    "ai-native": "AI-native",
    "ai native": "AI-native",
    "ai": "AI-native",
}

# Minimal CF columns to accept a Sheet/CSV (source_url is optional to avoid blocking)
CF_REQUIRED_CORE = {
    "startup", "country", "sector", "platform", "round_date", "valuation_pre_money_eur"
}

CF_NUMERIC = [
    "valuation_pre_money_eur", "amount_raised_eur", "revenue_last_fy_eur",
    "arr_eur", "mrr_eur", "gmv_eur", "assumed_take_rate_pct", "headcount"
]

PC_EXPECTED_ANY = {
    # at least one id
    "ticker", "symbol", "isin",
    # sector-ish
    "sector", "industry", "vertical",
    # value/revenue fields
    "enterprise_value", "market_cap",
    "revenue_ttm", "revenue_total",
    # optional method column
    "ev_rev_method"
}


# ------------------------------- Small Helpers -------------------------------

def _norm_sector(s: pd.Series) -> pd.Series:
    if s is None:
        return s
    key = s.astype(str).str.strip().str.lower()
    return key.map(SIFTED_SECTORS).fillna(s.astype(str).str.strip())


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _has_cols(df: pd.DataFrame, required: set[str]) -> tuple[bool, set[str]]:
    cols = {c.strip().lower() for c in df.columns}
    miss = {c for c in required if c not in cols}
    return (len(miss) == 0), miss


# ------------------------------- Crowdfunding Loader -------------------------------

def load_crowdfunding() -> tuple[pd.DataFrame, str, list[str]]:
    """Load CF with strict preference and expose clear rejection reasons.
       Order: Google Sheet (secrets) -> data/crowdfunding_live.csv -> data/crowdfunding_sample.csv
       Returns: (df, source_label, notes[])
    """
    notes: list[str] = []

    # 1) Google Sheet CSV from secrets
    sheet_url = ""
    try:
        sheet_url = (st.secrets.get("GOOGLE_SHEET_CSV") or "").strip()
    except Exception:
        sheet_url = ""

    if sheet_url:
        try:
            df = pd.read_csv(sheet_url)
            ok, miss = _has_cols(df, CF_REQUIRED_CORE)
            if not ok:
                notes.append(f"Sheet rejected: missing required columns: {sorted(miss)}")
            else:
                df["_source"] = "google_sheet_csv"
                # parse & normalize
                if "round_date" in df.columns:
                    df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")
                df["sector"] = _norm_sector(df.get("sector"))
                df = _to_numeric(df, CF_NUMERIC)
                return df, "google_sheet_csv", notes
        except Exception as e:
            notes.append(f"Sheet load error: {e!s}")

    # 2) Local live CSV
    live = DATA_DIR / "crowdfunding_live.csv"
    if live.exists():
        try:
            df = pd.read_csv(live)
            ok, miss = _has_cols(df, CF_REQUIRED_CORE)
            if not ok:
                notes.append(f"Local live CSV rejected: missing columns: {sorted(miss)}")
            else:
                df["_source"] = "local_live_csv"
                if "round_date" in df.columns:
                    df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")
                df["sector"] = _norm_sector(df.get("sector"))
                df = _to_numeric(df, CF_NUMERIC)
                return df, "local_live_csv", notes
        except Exception as e:
            notes.append(f"Local live CSV error: {e!s}")

    # 3) Sample
    sample = DATA_DIR / "crowdfunding_sample.csv"
    df = pd.read_csv(sample)
    df["_source"] = "sample_csv"
    if "round_date" in df.columns:
        df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")
    df["sector"] = _norm_sector(df.get("sector"))
    df = _to_numeric(df, CF_NUMERIC)
    return df, "sample_csv", notes


# ------------------------------- Public Comps Loader -------------------------------

def _coalesce_pc_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Map arbitrary column names to a common set: ticker, sector, country, ev_to_revenue, ev_rev_method."""
    cols = {c.lower(): c for c in df.columns}

    def pick(*cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    out = pd.DataFrame(index=df.index)
    # id
    out["ticker"] = df.get(pick("ticker", "symbol", "isin"))
    # sector
    sector_col = pick("sector", "industry", "vertical")
    if sector_col:
        out["sector"] = df[sector_col]
    else:
        out["sector"] = pd.NA
    # country
    out["country"] = df.get(pick("country", "listing_country", "hq_country"))
    # enterprise/revenue
    ev = df.get(pick("enterprise_value"))
    mcap = df.get(pick("market_cap"))
    rev_ttm = df.get(pick("revenue_ttm"))
    rev_total = df.get(pick("revenue_total"))

    ev_to_rev = pd.Series(pd.NA, index=df.index, dtype="Float64")
    method = pd.Series(pd.NA, index=df.index, dtype="string")

    # 1) EV / total revenue
    if ev is not None and rev_total is not None:
        num = pd.to_numeric(ev, errors="coerce")
        den = pd.to_numeric(rev_total, errors="coerce")
        ratio = num / den
        mask = den > 0
        ev_to_rev[mask] = ratio[mask]
        method[mask] = "EV/TotalRevenue"

    # 2) MarketCap / TTM revenue (fallback where EV/Total missing)
    if mcap is not None and rev_ttm is not None:
        num = pd.to_numeric(mcap, errors="coerce")
        den = pd.to_numeric(rev_ttm, errors="coerce")
        ratio = num / den
        # only fill where still NA
        fill_mask = ev_to_rev.isna() & (den > 0)
        ev_to_rev[fill_mask] = ratio[fill_mask]
        method[fill_mask] = "MktCap/TTMRevenue"

    out["ev_to_revenue"] = ev_to_rev
    # prefer explicit ev_rev_method column if present
    method_col = pick("ev_rev_method")
    out["ev_rev_method"] = df.get(method_col, method)
    return out


def load_public_comps() -> tuple[pd.DataFrame, str, list[str]]:
    """Load public comps with flexible column mapping.
       Order: BME Growth CSV (secrets) -> data/public_comps.csv -> data/public_comps_sample.csv
       Returns: (df, source_label, notes[])
    """
    notes: list[str] = []

    # 1) BME Growth CSV from secrets
    bme_url = ""
    try:
        bme_url = (st.secrets.get("BME_GROWTH_CSV_URL") or "").strip()
    except Exception:
        bme_url = ""

    if bme_url:
        try:
            raw = pd.read_csv(bme_url)
            if raw.empty:
                notes.append("BME CSV loaded but has 0 rows")
            else:
                df = _coalesce_pc_cols(raw)
                df["_pc_source"] = "bme_growth_csv"
                df["sector"] = _norm_sector(df.get("sector"))
                return df, "bme_growth_csv", notes
        except Exception as e:
            notes.append(f"BME CSV load error: {e!s}")

    # 2) Local public_comps.csv
    local = DATA_DIR / "public_comps.csv"
    if local.exists():
        try:
            raw = pd.read_csv(local)
            if raw.empty:
                notes.append("local public_comps.csv exists but is empty")
            else:
                df = _coalesce_pc_cols(raw)
                df["_pc_source"] = "local_csv"
                df["sector"] = _norm_sector(df.get("sector"))
                return df, "local_csv", notes
        except Exception as e:
            notes.append(f"local public_comps.csv error: {e!s}")

    # 3) Sample public comps
    sample = DATA_DIR / "public_comps_sample.csv"
    if sample.exists():
        raw = pd.read_csv(sample)
    else:
        # fallback to the original default if sample not available
        raw = pd.read_csv(DATA_DIR / "public_comps.csv")
    df = _coalesce_pc_cols(raw)
    df["_pc_source"] = "sample_csv"
    df["sector"] = _norm_sector(df.get("sector"))
    return df, "sample_csv", notes


# ------------------------------- VGI Computation -------------------------------

def estimate_revenue(cf: pd.DataFrame) -> pd.DataFrame:
    """Create estimated_revenue_eur, rev_source, confidence (deterministic + transparent)."""
    df = cf.copy()
    for c in ["arr_eur", "mrr_eur", "gmv_eur", "assumed_take_rate_pct", "revenue_last_fy_eur"]:
        if c not in df.columns:
            df[c] = pd.NA
    _to_numeric(df, ["arr_eur", "mrr_eur", "gmv_eur", "assumed_take_rate_pct", "revenue_last_fy_eur"])

    est = pd.Series(pd.NA, index=df.index, dtype="Float64")
    src = pd.Series(pd.NA, index=df.index, dtype="string")
    conf = pd.Series(pd.NA, index=df.index, dtype="string")

    # 1) Audited/declared revenue (high)
    m = df["revenue_last_fy_eur"].notna() & (df["revenue_last_fy_eur"] > 0)
    est[m] = df.loc[m, "revenue_last_fy_eur"]
    src[m] = "audited_revenue"
    conf[m] = "High"

    # 2) ARR (medium) — ARR ~ annual revenue proxy
    if "arr_eur" in df.columns:
        m = est.isna() & df["arr_eur"].notna() & (df["arr_eur"] > 0)
        est[m] = df.loc[m, "arr_eur"]
        src[m] = "arr"
        conf[m] = "Medium"

    # 3) MRR (medium) — MRR*12
    if "mrr_eur" in df.columns:
        m = est.isna() & df["mrr_eur"].notna() & (df["mrr_eur"] > 0)
        est[m] = df.loc[m, "mrr_eur"] * 12.0
        src[m] = "mrr_x12"
        conf[m] = "Medium"

    # 4) GMV * take-rate (low) — if both are available
    if "gmv_eur" in df.columns and "assumed_take_rate_pct" in df.columns:
        tr = pd.to_numeric(df["assumed_take_rate_pct"], errors="coerce") / 100.0
        m = est.isna() & df["gmv_eur"].notna() & (df["gmv_eur"] > 0) & tr.notna() & (tr > 0)
        est[m] = df.loc[m, "gmv_eur"] * tr[m]
        src[m] = "gmv_take_rate"
        conf[m] = "Low"

    df["estimated_revenue_eur"] = est
    df["rev_source"] = df.get("rev_source", src).fillna(src)
    df["confidence"] = df.get("confidence", conf).fillna(conf)
    return df


def sector_public_median(pc: pd.DataFrame, min_n: int = 1) -> pd.DataFrame:
    """Pick a single consistent method per sector, then compute median EV/Revenue."""
    df = pc.copy()
    # Determine dominant method per sector
    method_counts = (
        df.dropna(subset=["sector", "ev_to_revenue"])
          .groupby(["sector", "ev_rev_method"], dropna=False)
          .size()
          .reset_index(name="n")
    )
    # pick method with max n per sector
    winners = method_counts.sort_values(["sector", "n"], ascending=[True, False]) \
                           .drop_duplicates("sector")[["sector", "ev_rev_method"]]
    df = df.merge(winners, on="sector", how="left", suffixes=("", "_chosen"))
    # keep only chosen method rows
    df = df[df["ev_rev_method"] == df["ev_rev_method_chosen"]]
    # median per sector
    agg = (
        df.dropna(subset=["ev_to_revenue"])
          .groupby("sector")
          .agg(public_median_ev_rev=("ev_to_revenue", "median"),
               public_n=("ev_to_revenue", "count"),
               public_method_used=("ev_rev_method", "first"))
          .reset_index()
    )
    # filter coverage
    agg = agg[agg["public_n"] >= min_n]
    return agg


def compute_vgi(cf: pd.DataFrame, pc: pd.DataFrame) -> pd.DataFrame:
    """Compute startup_ev_rev and VGI per startup, joining sector public medians."""
    cf2 = estimate_revenue(cf)
    cf2["startup_ev_rev"] = cf2["valuation_pre_money_eur"] / cf2["estimated_revenue_eur"]
    pub = sector_public_median(pc, min_n=1)
    out = cf2.merge(pub, on="sector", how="left")
    out["VGI"] = out["startup_ev_rev"] / out["public_median_ev_rev"]
    return out


# ------------------------------- Unified Loader -------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, str, str, list[str], list[str]]:
    cf, cf_src_notes = None, []
    pc, pc_src_notes = None, []

    cf, cf_source, cf_src_notes = load_crowdfunding()
    pc, pc_source, pc_src_notes = load_public_comps()

    return cf, pc, cf_source, pc_source, cf_src_notes, pc_src_notes


# ------------------------------- UI -------------------------------

st.set_page_config(page_title="Startup Value Monitor — VGI (MVP)", layout="wide")

def main():
    st.title("Startup Value Monitor — VGI (MVP)")
    st.caption("Comparing crowdfunding valuations with public-market multiples.")

    # Data
    cf, pc, cf_source, pc_source, cf_notes, pc_notes = load_data()

    # Banners
    st.info(f"Crowdfunding source: {cf_source} • rows: {len(cf)}")
    st.info(f"Public comps source: {pc_source} • rows: {len(pc)}")

    # Charts only if we have some coverage
    if not cf.empty and not pc.empty:
        # Compute per-startup VGI
        per = compute_vgi(cf, pc)

        # Sector-level VGI chart
        sector_vgi = (
            per.dropna(subset=["VGI"])
               .groupby("sector")
               .agg(vgi=("VGI", "median"), n=("VGI", "count"))
               .reset_index()
               .sort_values("vgi", ascending=False)
        )
        if not sector_vgi.empty:
            st.subheader("Valuation Gap Index (by sector)")
            chart = (
                alt.Chart(sector_vgi)
                  .mark_bar()
                  .encode(
                      x=alt.X("sector:N", sort="-y", title="Sector"),
                      y=alt.Y("vgi:Q", title="VGI = CF EV/Rev ÷ Public EV/Rev"),
                      tooltip=["sector:N", alt.Tooltip("vgi:Q", format=".2f"), "n:Q"]
                  )
                  .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)

            overall = sector_vgi["vgi"].median() if not sector_vgi.empty else float("nan")
            st.metric("Median VGI (all sectors)", f"{overall:.2f}" if not math.isnan(overall) else "—")

        # Scatter: CF Pre-Money vs Public EV/Revenue (sector medians)
        pub = sector_public_median(pc, min_n=1)
        if not pub.empty:
            st.subheader("Crowdfunding Pre-Money vs Public EV/Revenue (by sector)")
            agg_cf = (
                per.dropna(subset=["valuation_pre_money_eur"])
                   .groupby("sector")
                   .agg(cf_median_premoney=("valuation_pre_money_eur", "median"))
                   .reset_index()
            )
            bubble = agg_cf.merge(pub, on="sector", how="inner")
            if not bubble.empty:
                scatter = (
                    alt.Chart(bubble)
                       .mark_circle()
                       .encode(
                           x=alt.X("public_median_ev_rev:Q", title="Public EV/Revenue (median)"),
                           y=alt.Y("cf_median_premoney:Q", title="Crowdfunding Pre-Money (median, €)"),
                           size=alt.Size("public_n:Q", title="# comps"),
                           color=alt.Color("sector:N", legend=None),
                           tooltip=[
                               "sector:N",
                               alt.Tooltip("public_median_ev_rev:Q", title="Public EV/Rev (median)", format=".2f"),
                               alt.Tooltip("cf_median_premoney:Q", title="CF Pre-Money (median €)", format=",.0f"),
                               alt.Tooltip("public_n:Q", title="Comps used")
                           ],
                       )
                       .properties(height=360)
                )
                st.altair_chart(scatter, use_container_width=True)

        # All startups (sector table)
        st.subheader("All startups included in VGI (by sector)")
        sectors = sorted(per["sector"].dropna().unique().tolist())
        if sectors:
            chosen = st.selectbox("Pick a sector", options=sectors, index=0)
            used = per[(per["sector"] == chosen) & per["VGI"].notna()].copy()

            desired = [
                "sector", "startup", "country",
                "valuation_pre_money_eur", "estimated_revenue_eur",
                "startup_ev_rev", "public_median_ev_rev", "VGI",
                "rev_source", "confidence", "platform", "round_date",
            ]
            cols = [c for c in desired if c in used.columns]

            st.caption(f"{len(used)} startups shown (valid VGI) out of { (per['sector'] == chosen).sum() } total rows in {chosen}.")
            if used.empty:
                st.info("No startups with a valid VGI in this sector yet (need non-zero revenue and a public median).")
            else:
                formatted = (
                    used[cols]
                    .sort_values("VGI", ascending=False)
                    .assign(
                        VGI=lambda d: d["VGI"].round(2),
                        startup_ev_rev=lambda d: d["startup_ev_rev"].round(2),
                        public_median_ev_rev=lambda d: d["public_median_ev_rev"].round(2),
                    )
                    .rename(columns={
                        "valuation_pre_money_eur": "Pre-money (€)",
                        "estimated_revenue_eur": "Revenue est. (€)",
                        "startup_ev_rev": "CF EV/Rev",
                        "public_median_ev_rev": "Public EV/Rev (median)",
                        "rev_source": "Revenue source",
                        "confidence": "Confidence",
                        "round_date": "Round date"
                    })
                )
                order = [
                    "sector", "startup", "country",
                    "Pre-money (€)", "Revenue est. (€)",
                    "CF EV/Rev", "Public EV/Rev (median)", "VGI",
                    "Revenue source", "Confidence", "platform", "Round date"
                ]
                display_cols = [c for c in order if c in formatted.columns]
                st.dataframe(formatted[display_cols], use_container_width=True)
        else:
            st.info("No sectors available — add data to your feed(s).")

    else:
        st.warning("Not enough data yet — ensure both CF and Public comps have rows.")

    # --------------------------------- Diagnostics (bottom) ---------------------------------
    st.divider()
    st.caption("Diagnostics")
    with st.expander("Crowdfunding diagnostics", expanded=False):
        st.write("Notes:", cf_notes if cf_notes else "none")
        st.write("Columns:", list(cf.columns))
        st.write(cf.head(10))
        st.write("Sectors:", cf["sector"].value_counts(dropna=False) if "sector" in cf.columns else "none")

    with st.expander("Public comps diagnostics", expanded=False):
        st.write("Notes:", pc_notes if pc_notes else "none")
        st.write("Columns:", list(pc.columns))
        st.write(pc.head(10))
        st.write("Sectors:", pc["sector"].value_counts(dropna=False) if "sector" in pc.columns else "none")


if __name__ == "__main__":
    main()
