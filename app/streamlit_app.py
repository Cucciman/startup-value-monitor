import pathlib
import pandas as pd
import altair as alt
import streamlit as st

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"

# ---------------------------- Sector normalization ----------------------------

def _normalize_sector_names(df: pd.DataFrame, col: str = "sector") -> pd.DataFrame:
    """
    Map assorted sector labels into a canonical taxonomy aligned to:
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
        "climatetech": "Climate", "greentech": "Climate",
        "cleantech": "Climate", "energy": "Climate", "renewables": "Climate",
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
        "ai": "AI-native", "AI": "AI-native", "AI native": "AI-native",
        "GenAI": "AI-native", "machine learning": "AI-native",
    }
    normalized = raw.replace(mapping)

    def infer(v: str) -> str:
        s = v.lower()
        if "fintech" in s or "payment" in s or "bank" in s: return "Fintech"
        if "saas" in s or "software" in s or "b2b" in s: return "B2B SaaS"
        if any(k in s for k in ["climate", "green", "clean", "energy", "renew"]): return "Climate"
        if any(k in s for k in ["deep", "semicon", "robot", "space", "quantum"]): return "Deeptech"
        if any(k in s for k in ["health", "med", "bio"]): return "Healthtech"
        if any(k in s for k in ["consumer", "ecom", "marketplace"]): return "Consumer"
        if any(k in s for k in ["ai", "genai", "ml", "artificial intelligence"]): return "AI-native"
        return v

    normalized = normalized.apply(infer)
    allowed = {"Fintech", "B2B SaaS", "Climate", "Deeptech", "Healthtech", "Consumer", "AI-native"}
    df[col] = normalized.where(normalized.isin(allowed), normalized)
    return df

# ---------------------------- Data loading ------------------------------------

@st.cache_data
def load_data():
    # Crowdfunding (sample/mutable)
    cf = pd.read_csv(DATA_DIR / "crowdfunding_sample.csv", parse_dates=["round_date"])
    cf = _normalize_sector_names(cf)

    # Public comps (prefer live, fallback to sample)
    primary = DATA_DIR / "public_comps.csv"
    fallback = DATA_DIR / "public_comps_sample.csv"
    if primary.exists():
        pc = pd.read_csv(primary)
        src = "public_comps.csv (live)"
    else:
        pc = pd.read_csv(fallback)
        src = "public_comps_sample.csv (sample)"
    pc = _normalize_sector_names(pc)
    # Ensure expected columns exist
    for col in ["ev_to_revenue", "ev_rev_method", "sector", "ticker"]:
        if col not in pc.columns:
            pc[col] = pd.NA
    return cf, pc, src

# -------- Consistent method per sector (EV/TotalRevenue vs MktCap/TTMRevenue) --

def sector_public_median_by_consistent_method(pc: pd.DataFrame, min_n: int = 1) -> pd.DataFrame:
    """
    For each sector, choose ONE method (EV/TotalRevenue or MktCap/TTMRevenue)
    based on coverage (>= min_n). Return median + method used + n.
    MVP uses min_n=1 so sectors render even with limited comps.
    """
    df = pc.copy()
    df["ev_to_revenue"] = pd.to_numeric(df["ev_to_revenue"], errors="coerce")
    # Count valid comps per sector/method
    counts = (
        df.dropna(subset=["ev_to_revenue", "ev_rev_method"])
          .groupby(["sector", "ev_rev_method"])["ticker"]
          .count()
          .reset_index(name="n")
    )

    rows = []
    for sector in sorted(df["sector"].dropna().unique()):
        method_used = None
        n_used = 0
        # Prefer EV/TotalRevenue; else MktCap/TTMRevenue
        c1 = counts[(counts["sector"] == sector) & (counts["ev_rev_method"] == "EV/TotalRevenue")]
        if not c1.empty and int(c1["n"].iloc[0]) >= min_n:
            method_used = "EV/TotalRevenue"; n_used = int(c1["n"].iloc[0])
        else:
            c2 = counts[(counts["sector"] == sector) & (counts["ev_rev_method"] == "MktCap/TTMRevenue")]
            if not c2.empty and int(c2["n"].iloc[0]) >= min_n:
                method_used = "MktCap/TTMRevenue"; n_used = int(c2["n"].iloc[0])

        if method_used is not None:
            med = (
                df[(df["sector"] == sector) & (df["ev_rev_method"] == method_used)]
                  .dropna(subset=["ev_to_revenue"])["ev_to_revenue"]
                  .median()
            )
            rows.append({
                "sector": sector,
                "public_median_ev_rev": med,
                "public_method_used": method_used,
                "public_n_used": n_used,
            })
        else:
            rows.append({
                "sector": sector,
                "public_median_ev_rev": None,
                "public_method_used": "insufficient",
                "public_n_used": 0,
            })
    return pd.DataFrame(rows)

# ---------------------------- Aggregations ------------------------------------

def sector_summary(cf: pd.DataFrame, pc: pd.DataFrame) -> pd.DataFrame:
    cf_sect = (
        cf.groupby("sector", as_index=False)
          .agg({
              "valuation_pre_money_eur": "median",
              "amount_raised_eur": "sum",
              "revenue_last_fy_eur": "median",
          })
          .rename(columns={
              "valuation_pre_money_eur": "cf_median_pre_money",
              "amount_raised_eur": "cf_total_raised",
              "revenue_last_fy_eur": "cf_median_revenue",
          })
    )
    pc_med = sector_public_median_by_consistent_method(pc, min_n=1)  # MVP: 1
    return cf_sect.merge(pc_med, on="sector", how="left")

def compute_vgi(df: pd.DataFrame) -> pd.DataFrame:
    """Sector-level VGI = (CF EV/Rev) / (Public EV/Rev)."""
    out = df.copy()
    out["cf_ev_rev"] = pd.to_numeric(out.get("cf_median_pre_money"), errors="coerce") / pd.to_numeric(
        out.get("cf_median_revenue"), errors="coerce"
    )
    out["VGI"] = out["cf_ev_rev"] / pd.to_numeric(out.get("public_median_ev_rev"), errors="coerce")
    out.replace([float("inf"), -float("inf")], pd.NA, inplace=True)
    return out

def estimate_revenue(cf: pd.DataFrame) -> pd.DataFrame:
    """
    Create estimated_revenue_eur, rev_source, confidence
    from available columns: revenue_last_fy_eur, arr_eur, mrr_eur, gmv_eur,
    assumed_take_rate_pct, headcount.
    """
    df = cf.copy()

    # Ensure optional cols exist
    for c in ["arr_eur", "mrr_eur", "gmv_eur", "assumed_take_rate_pct", "headcount"]:
        if c not in df.columns:
            df[c] = pd.NA

    # Normalize numerics
    num_cols = ["revenue_last_fy_eur", "arr_eur", "mrr_eur", "gmv_eur", "assumed_take_rate_pct", "headcount"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    est = pd.Series(pd.NA, index=df.index, dtype="float")
    src = pd.Series(pd.NA, index=df.index, dtype="string")
    conf = pd.Series(pd.NA, index=df.index, dtype="string")

    # 1) Audited/last FY revenue
    mask = df["revenue_last_fy_eur"].notna() & (df["revenue_last_fy_eur"] > 0)
    est[mask] = df.loc[mask, "revenue_last_fy_eur"]
    src[mask] = "audited_revenue"
    conf[mask] = "High"

    # 2) ARR
    mask = est.isna() & df["arr_eur"].notna() & (df["arr_eur"] > 0)
    est[mask] = df.loc[mask, "arr_eur"]
    src[mask] = "ARR"
    conf[mask] = "Medium-High"

    # 3) MRR x 12
    mask = est.isna() & df["mrr_eur"].notna() & (df["mrr_eur"] > 0)
    est[mask] = df.loc[mask, "mrr_eur"] * 12.0
    src[mask] = "MRRx12"
    conf[mask] = "Medium"

    # 4) GMV x take-rate (use defaults if missing)
    default_take = {
        "Consumer": 8.0,   # marketplaces
        "Fintech": 2.0,    # payments
        "B2B SaaS": 100.0, # GMV rarely applies; if present assume it's revenue
        "AI-native": 100.0,
        "Healthtech": 30.0,
        "Climate": 20.0,
        "Deeptech": 100.0
    }
    take = df["assumed_take_rate_pct"].copy()
    take = take.where(take.notna(), df["sector"].map(default_take))
    mask = est.isna() & df["gmv_eur"].notna() & (df["gmv_eur"] > 0) & take.notna()
    est[mask] = df.loc[mask, "gmv_eur"] * (take[mask] / 100.0)
    src[mask] = "GMVxTakeRate"
    conf[mask] = "Low-Med"

    # 5) Headcount model (rough sector benchmarks for revenue/employee)
    rev_per_emp = {
        "B2B SaaS": 160000.0,
        "AI-native": 150000.0,
        "Fintech": 180000.0,
        "Consumer": 120000.0,
        "Healthtech": 170000.0,
        "Climate": 200000.0,
        "Deeptech": 220000.0,
    }
    bench = df["sector"].map(rev_per_emp)
    mask = est.isna() & df["headcount"].notna() & (df["headcount"] > 0) & bench.notna()
    est[mask] = df.loc[mask, "headcount"] * bench[mask]
    src[mask] = "HeadcountModel"
    conf[mask] = "Low"

    df["estimated_revenue_eur"] = est
    df["rev_source"] = src
    df["confidence"] = conf
    return df

def compute_startup_vgi(cf: pd.DataFrame, pc: pd.DataFrame) -> pd.DataFrame:
    """Per-startup VGI = (startup EV/Rev) / (sector public EV/Rev)."""
    per = cf.copy()
    if "startup" not in per.columns and "startup_name" in per.columns:
        per = per.rename(columns={"startup_name": "startup"})
    for c in ["valuation_pre_money_eur", "revenue_last_fy_eur"]:
        if c in per.columns:
            per[c] = pd.to_numeric(per[c], errors="coerce")
    if "revenue_last_fy_eur" in per.columns:
        per.loc[per["revenue_last_fy_eur"] <= 0, "revenue_last_fy_eur"] = pd.NA
    per["startup_ev_rev"] = per.get("valuation_pre_money_eur") / per.get("revenue_last_fy_eur")

    pub = sector_public_median_by_consistent_method(pc, min_n=1)  # MVP: 1
    out = per.merge(pub[["sector", "public_median_ev_rev"]], on="sector", how="left")
    out["VGI"] = out["startup_ev_rev"] / out["public_median_ev_rev"]
    out.replace([float("inf"), -float("inf")], pd.NA, inplace=True)
    return out

# ---------------------------- UI ---------------------------------------------

def main():
    st.set_page_config(page_title="Startup Value Monitor", layout="wide")
    st.title("Startup Value Monitor — VGI (MVP)")
    st.caption("Comparing crowdfunding valuations with public-market multiples.")

    cf, pc, source = load_data()
    st.caption(f"Public comps source: {source}")

    # --- Sector-level summary & charts
    df = compute_vgi(sector_summary(cf, pc))

    c1, c2 = st.columns([2, 1], vertical_alignment="center")
    with c1:
        st.subheader("Valuation Gap Index (by sector)")
        bar = alt.Chart(df.dropna(subset=["VGI"])).mark_bar().encode(
            x=alt.X("sector:N", sort="-y", title="Sector"),
            y=alt.Y("VGI:Q", title="VGI = CF EV/Rev ÷ Public EV/Rev"),
            tooltip=[
                "sector",
                alt.Tooltip("VGI:Q", format=".2f"),
                alt.Tooltip("cf_ev_rev:Q", title="CF EV/Rev", format=".2f"),
                alt.Tooltip("public_median_ev_rev:Q", title="Public EV/Rev", format=".2f"),
                "public_method_used:N",
                "public_n_used:Q",
            ],
        )
        st.altair_chart(bar, use_container_width=True)
    with c2:
        st.metric("Median VGI (all sectors)", f"{df['VGI'].median(skipna=True):.2f}")

    with st.expander("Public comps: method and coverage per sector"):
        st.dataframe(
            df[["sector", "public_median_ev_rev", "public_method_used", "public_n_used"]]
              .sort_values("sector")
              .assign(public_median_ev_rev=lambda d: d["public_median_ev_rev"].round(2))
        )

    # --- Layered scatter: sector medians + per-startup points
    st.markdown("---")
    st.subheader("Crowdfunding Pre-Money vs Public EV/Revenue (by sector)")

    # Background: sector medians (one dot per sector)
    sector_scatter = alt.Chart(
        df.dropna(subset=["public_median_ev_rev", "cf_median_pre_money"])
    ).mark_circle(size=200, opacity=0.35).encode(
        x=alt.X("public_median_ev_rev:Q", title="Public EV/Revenue (median)"),
        y=alt.Y("cf_median_pre_money:Q", title="Crowdfunding Pre-Money (median, €)"),
        tooltip=[
            "sector:N",
            alt.Tooltip("cf_median_pre_money:Q", title="CF Pre-Money (median, €)", format=","),
            alt.Tooltip("public_median_ev_rev:Q", title="Public EV/Rev (median)", format=".2f"),
            alt.Tooltip("VGI:Q", title="Sector VGI", format=".2f"),
        ],
        color=alt.value("lightgray"),
    )

    # Foreground: per-startup points (each uses sector public median on X)
    per_startup = compute_startup_vgi(cf, pc)
    pts_source = per_startup.dropna(
        subset=["public_median_ev_rev", "valuation_pre_money_eur", "startup"]
    ).copy()

    hover = alt.selection_point(fields=["startup"], on="mouseover", nearest=True, empty="none")

    startup_points = alt.Chart(pts_source).mark_circle(size=45).encode(
        x=alt.X("public_median_ev_rev:Q", title="Public EV/Revenue (median)"),
        y=alt.Y("valuation_pre_money_eur:Q", title="Crowdfunding Pre-Money (€)"),
        tooltip=[
            "startup:N",
            "sector:N",
            alt.Tooltip("valuation_pre_money_eur:Q", title="Pre-Money (€)", format=","),
            alt.Tooltip("public_median_ev_rev:Q", title="Public EV/Rev (median)", format=".2f"),
            alt.Tooltip("VGI:Q", title="Startup VGI", format=".2f"),
        ],
        color=alt.condition(hover, "sector:N", alt.value("lightgray")),
        opacity=alt.condition(hover, alt.value(1.0), alt.value(0.35)),
    ).add_params(hover)

    layered = (sector_scatter + startup_points).properties(height=420)
    st.altair_chart(layered, use_container_width=True)

    # --- All startups included in VGI (by sector)
    st.markdown("---")
    st.subheader("All startups included in VGI (by sector)")

    sectors = sorted(per_startup["sector"].dropna().unique().tolist())
    if not sectors:
        st.info("No sectors available yet. Add crowdfunding rows and public comps, then rerun.")
    else:
        chosen = st.selectbox("Pick a sector", options=sectors, index=0)

        used = per_startup[
            (per_startup["sector"] == chosen) & (per_startup["VGI"].notna())
        ].copy()

        desired = [
            "sector", "startup", "country",
            "valuation_pre_money_eur", "revenue_last_fy_eur",
            "startup_ev_rev", "public_median_ev_rev", "VGI",
            "platform", "round_date",
        ]
        cols = [c for c in desired if c in used.columns]

        if used.empty:
            st.info("No startups with a valid VGI in this sector yet (need non-zero revenue and a public median).")
        else:
            table = (
                used[cols]
                .sort_values("VGI", ascending=False)
                .assign(
                    VGI=lambda d: d["VGI"].round(2),
                    startup_ev_rev=lambda d: d["startup_ev_rev"].round(2),
                    public_median_ev_rev=lambda d: d["public_median_ev_rev"].round(2),
                )
            )
            st.dataframe(table, use_container_width=True)

            csv = table.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Download startups used for {chosen} (CSV)",
                data=csv,
                file_name=f"vgi_startups_{chosen}.csv",
                mime="text/csv",
            )

        # Transparency: which public comps were used for this sector’s median?
        st.markdown("**Public comps used for this sector’s median (method-consistent)**")
        pc_med = sector_public_median_by_consistent_method(pc, min_n=1)
        meta = pc_med.set_index("sector").to_dict(orient="index")
        method = meta.get(chosen, {}).get("public_method_used", None)

        if method and method != "insufficient" and "ev_rev_method" in pc.columns:
            comps_used = pc[
                (pc["sector"] == chosen)
                & (pc["ev_rev_method"] == method)
                & (pc["ev_to_revenue"].notna())
            ].copy()
            comp_cols = [c for c in ["ticker", "country", "ev_rev_method", "ev_to_revenue", "data_date"] if c in comps_used.columns]
            if not comps_used.empty:
                st.dataframe(
                    comps_used[comp_cols].assign(
                        ev_to_revenue=lambda d: pd.to_numeric(d["ev_to_revenue"], errors="coerce").round(2)
                    ).sort_values("ev_to_revenue"),
                    use_container_width=True,
                )
            else:
                st.info("No valid comps found yet for the chosen method; add/refresh the watchlist.")
        else:
            st.info("Public comps coverage is insufficient for a consistent median in this sector.")

    # --- Notes
    st.markdown("---")
    st.write("**Notes**")
    st.write("- Public comps median uses a single consistent method per sector (shown above).")
    st.write("- Replace sample crowdfunding CSVs with real data next.")
    st.write("- In production, compute EV/Revenue per round and aggregate by sector/country.")


if __name__ == "__main__":
    main()
