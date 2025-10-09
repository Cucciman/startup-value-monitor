import pandas as pd
import altair as alt
import pathlib
import streamlit as st

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / 'data'

# --- Normalize sector taxonomy -------------------------------------------------
def _normalize_sector_names(df: pd.DataFrame, col: str = "sector") -> pd.DataFrame:
    """
    Map assorted sector labels into a canonical Sifted-style taxonomy:
    {Fintech, B2B SaaS, Climate, Deeptech, Healthtech, Consumer, AI-native}
    """
    if col not in df.columns:
        return df
    raw = (
        df[col].astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    mapping = {
        # SaaS
        "saas": "B2B SaaS", "SaaS": "B2B SaaS", "SaAS": "B2B SaaS",
        "software": "B2B SaaS", "enterprise software": "B2B SaaS",
        # Climate
        "climate": "Climate", "Climate Tech": "Climate",
        "climatetech": "Climate", "greentech": "Climate", "cleantech": "Climate",
        "energy": "Climate", "renewables": "Climate",
        # Fintech
        "fintech": "Fintech", "FinTech": "Fintech", "Fin tech": "Fintech", "payments": "Fintech",
        # Deeptech
        "deeptech": "Deeptech", "DeepTech": "Deeptech", "Deep tech": "Deeptech",
        "semiconductors": "Deeptech", "robotics": "Deeptech", "space": "Deeptech",
        # Healthtech
        "healthtech": "Healthtech", "HealthTech": "Healthtech", "digital health": "Healthtech",
        "medtech": "Healthtech", "biotech": "Healthtech",
        # Consumer
        "consumer": "Consumer", "ConsumerTech": "Consumer", "ecommerce": "Consumer",
        "marketplace": "Consumer", "D2C": "Consumer",
        # AI-native
        "ai": "AI-native", "AI": "AI-native", "AI native": "AI-native", "GenAI": "AI-native",
        "machine learning": "AI-native",
    }
    normalized = raw.replace(mapping)

    def infer(v: str) -> str:
        s = v.lower()
        if "fintech" in s or "payment" in s or "bank" in s: return "Fintech"
        if "saas" in s or "software" in s or "b2b" in s: return "B2B SaaS"
        if any(k in s for k in ["climate","green","clean","energy","renew"]): return "Climate"
        if any(k in s for k in ["deep","semicon","robot","space","quantum"]): return "Deeptech"
        if any(k in s for k in ["health","med","bio"]): return "Healthtech"
        if any(k in s for k in ["consumer","ecom","marketplace"]): return "Consumer"
        if any(k in s for k in ["ai","genai","ml","artificial intelligence"]): return "AI-native"
        return v

    normalized = normalized.apply(infer)
    allowed = {"Fintech","B2B SaaS","Climate","Deeptech","Healthtech","Consumer","AI-native"}
    df[col] = normalized.where(normalized.isin(allowed), normalized)
    return df
# -------------------------------------------------------------------------------

@st.cache_data
def load_data():
    # Crowdfunding (sample for now)
    cf = pd.read_csv(DATA_DIR / 'crowdfunding_sample.csv', parse_dates=['round_date'])
    cf = _normalize_sector_names(cf)

    # Prefer live public comps if present; fall back to sample
    public_primary = DATA_DIR / 'public_comps.csv'
    public_fallback = DATA_DIR / 'public_comps_sample.csv'
    if public_primary.exists():
        pc = pd.read_csv(public_primary)
        source = "public_comps.csv (live)"
    else:
        pc = pd.read_csv(public_fallback)
        source = "public_comps_sample.csv (sample)"
    pc = _normalize_sector_names(pc)

    return cf, pc, source

def sector_summary(cf: pd.DataFrame, pc: pd.DataFrame) -> pd.DataFrame:
    cf_sect = (
        cf.groupby('sector', as_index=False)
          .agg({'valuation_pre_money_eur':'median',
                'amount_raised_eur':'sum',
                'revenue_last_fy_eur':'median'})
          .rename(columns={
              'valuation_pre_money_eur':'cf_median_pre_money',
              'amount_raised_eur':'cf_total_raised',
              'revenue_last_fy_eur':'cf_median_revenue'
          })
    )
    pc_sect = (
        pc.groupby('sector', as_index=False)
          .agg({'ev_to_revenue':'median'})
          .rename(columns={'ev_to_revenue':'public_median_ev_rev'})
    )
    return cf_sect.merge(pc_sect, on='sector', how='left')

def compute_vgi(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['cf_ev_rev'] = out['cf_median_pre_money'] / out['cf_median_revenue']
    out['VGI'] = out['cf_ev_rev'] / out['public_median_ev_rev']
    return out

# --- Top 5 / Bottom 5 helpers (robust) ----------------------------------------
def compute_startup_vgi(cf: pd.DataFrame, pc: pd.DataFrame) -> pd.DataFrame:
    """Compute VGI for each startup (CF EV/Rev ÷ public median EV/Rev), safely."""
    per = cf.copy()
    if "startup" not in per.columns and "startup_name" in per.columns:
        per = per.rename(columns={"startup_name": "startup"})
    for c in ["valuation_pre_money_eur", "revenue_last_fy_eur"]:
        if c in per.columns:
            per[c] = pd.to_numeric(per[c], errors="coerce")
    if "revenue_last_fy_eur" in per.columns:
        per.loc[per["revenue_last_fy_eur"] <= 0, "revenue_last_fy_eur"] = pd.NA
    per["startup_ev_rev"] = per.get("valuation_pre_money_eur") / per.get("revenue_last_fy_eur")

    pub = pc.copy()
    if "ev_to_revenue" in pub.columns:
        pub["ev_to_revenue"] = pd.to_numeric(pub["ev_to_revenue"], errors="coerce")
    pub = (
        pub.groupby("sector", as_index=False)["ev_to_revenue"]
           .median()
           .rename(columns={"ev_to_revenue": "public_median_ev_rev"})
    )
    out = per.merge(pub, on="sector", how="left")
    out["VGI"] = out["startup_ev_rev"] / out["public_median_ev_rev"]
    out.replace([float("inf"), -float("inf")], pd.NA, inplace=True)
    return out

def top_bottom_by_sector(per_startup: pd.DataFrame, sector: str, n: int = 5):
    """Return top/bottom n startups by VGI for a given sector (handles missing cols)."""
    df = per_startup[per_startup["sector"] == sector].dropna(subset=["VGI"]).copy()
    if df.empty:
        return df, df
    if "startup" not in df.columns and "startup_name" in df.columns:
        df = df.rename(columns={"startup_name": "startup"})
    desired = [
        "platform","round_date","startup","country","sector",
        "valuation_pre_money_eur","revenue_last_fy_eur",
        "startup_ev_rev","public_median_ev_rev","VGI"
    ]
    present = [c for c in desired if c in df.columns]
    df = df[present].sort_values("VGI", ascending=False)
    topn = df.head(n)
    bottomn = df.tail(n).sort_values("VGI", ascending=True)
    return topn, bottomn
# -------------------------------------------------------------------------------

def main():
    st.set_page_config(page_title='Startup Value Monitor', layout='wide')
    st.title('Startup Value Monitor — VGI (MVP)')
    st.caption('Comparing crowdfunding valuations with public-market multiples.')

    cf, pc, source = load_data()
    st.caption(f"Public comps source: {source}")

    # ----- Sector-level VGI charts -----
    df = compute_vgi(sector_summary(cf, pc))

    c1, c2 = st.columns([2, 1], vertical_alignment="center")
    with c1:
        st.subheader('Valuation Gap Index (by sector)')
        bar = alt.Chart(df.dropna(subset=['VGI'])).mark_bar().encode(
            x=alt.X('sector', sort='-y', title='Sector'),
            y=alt.Y('VGI', title='VGI = CF EV/Rev ÷ Public EV/Rev'),
            tooltip=['sector',
                     alt.Tooltip('VGI', format='.2f'),
                     alt.Tooltip('cf_ev_rev', title='CF EV/Rev', format='.2f'),
                     alt.Tooltip('public_median_ev_rev', title='Public EV/Rev', format='.2f')]
        )
        st.altair_chart(bar, use_container_width=True)
    with c2:
        st.metric('Median VGI (all sectors)', f"{df['VGI'].median(skipna=True):.2f}")

    st.markdown('---')
    st.subheader('Crowdfunding Pre-Money vs Public EV/Revenue (by sector)')
    scatter = alt.Chart(df.dropna(subset=['public_median_ev_rev', 'cf_median_pre_money'])).mark_circle(size=140).encode(
        x=alt.X('public_median_ev_rev', title='Public EV/Revenue (median)'),
        y=alt.Y('cf_median_pre_money', title='Crowdfunding Pre-Money (median, €)'),
        tooltip=['sector',
                 alt.Tooltip('cf_median_pre_money', format=','),
                 alt.Tooltip('public_median_ev_rev', format='.2f'),
                 alt.Tooltip('cf_total_raised', title='CF Total Raised', format=',')]
    )
    st.altair_chart(scatter, use_container_width=True)

    # ----- Top 5 / Bottom 5 (by sector, VGI) -----
    st.markdown('---')
    st.subheader("Top 5 / Bottom 5 (by sector, VGI)")

    per_startup = compute_startup_vgi(cf, pc)
    sectors = sorted(per_startup["sector"].dropna().unique().tolist())
    st.caption(f"Debug: startups with VGI rows = {len(per_startup.dropna(subset=['VGI']))}")

    if not sectors:
        st.info("No sectors available yet. Add crowdfunding rows and public comps, then rerun.")
    else:
        view_type = st.radio("View as:", ["Tables", "Charts"], horizontal=True)
        chosen = st.selectbox("Pick a sector", options=sectors, index=0)

        top5, bottom5 = top_bottom_by_sector(per_startup, chosen, n=5)

        if top5.empty and bottom5.empty:
            st.info("Not enough data in this sector yet. Add more crowdfunding rows with revenue to compute VGI.")
        else:
            if view_type == "Tables":
                c1, c2 = st.columns(2)
                if not top5.empty:
                    with c1:
                        st.markdown(f"**Top 5 — {chosen}**")
                        st.dataframe(top5.assign(
                            VGI=lambda d: d["VGI"].round(2),
                            startup_ev_rev=lambda d: d["startup_ev_rev"].round(2),
                            public_median_ev_rev=lambda d: d["public_median_ev_rev"].round(2)
                        ))
                if not bottom5.empty:
                    with c2:
                        st.markdown(f"**Bottom 5 — {chosen}**")
                        st.dataframe(bottom5.assign(
                            VGI=lambda d: d["VGI"].round(2),
                            startup_ev_rev=lambda d: d["startup_ev_rev"].round(2),
                            public_median_ev_rev=lambda d: d["public_median_ev_rev"].round(2)
                        ))
            else:
                c1, c2 = st.columns(2)
                if not top5.empty:
                    with c1:
                        st.markdown(f"**Top 5 — {chosen}**")
                        chart = alt.Chart(top5.reset_index(drop=True)).mark_bar().encode(
                            x=alt.X('VGI:Q', title='VGI'),
                            y=alt.Y('startup:N', sort='-x', title='Startup'),
                            tooltip=['startup', alt.Tooltip('VGI', format='.2f'),
                                     alt.Tooltip('startup_ev_rev', title='CF EV/Rev', format='.2f'),
                                     alt.Tooltip('public_median_ev_rev', title='Public EV/Rev', format='.2f')]
                        )
                        st.altair_chart(chart, use_container_width=True)
                if not bottom5.empty:
                    with c2:
                        st.markdown(f"**Bottom 5 — {chosen}**")
                        chart = alt.Chart(bottom5.reset_index(drop=True)).mark_bar().encode(
                            x=alt.X('VGI:Q', title='VGI'),
                            y=alt.Y('startup:N', sort='x', title='Startup'),
                            tooltip=['startup', alt.Tooltip('VGI', format='.2f'),
                                     alt.Tooltip('startup_ev_rev', title='CF EV/Rev', format='.2f'),
                                     alt.Tooltip('public_median_ev_rev', title='Public EV/Rev', format='.2f')]
                        )
                        st.altair_chart(chart, use_container_width=True)

    st.markdown('---')
    st.write("**Notes**")
    st.write("- Public comps now read from `data/public_comps.csv` when available (yfinance).")
    st.write("- This demo uses sample crowdfunding CSVs; we’ll replace them with real data next.")
    st.write("- In production, compute EV/Revenue per round and aggregate by sector/country.")

if __name__ == '__main__':
    main()
