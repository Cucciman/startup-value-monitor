import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime


# -------------------------------------------------------------------
# Data loading and helper utilities
# -------------------------------------------------------------------

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()


def sector_public_median_by_consistent_method(pc: pd.DataFrame, min_n=1) -> pd.DataFrame:
    """
    Aggregate median EV/Revenue per sector using consistent method.
    """
    if "ev_rev" not in pc.columns:
        return pd.DataFrame(columns=["sector", "public_median_ev_rev"])

    out = (
        pc.groupby("sector", as_index=False)
        .agg(public_median_ev_rev=("ev_rev", "median"), n_comps=("ev_rev", "count"))
        .query("n_comps >= @min_n")
    )
    return out


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
    # Use consistent method per sector, but allow min_n=1 for MVP coverage
    pc_med = sector_public_median_by_consistent_method(pc, min_n=1)
    return cf_sect.merge(pc_med, on='sector', how='left')


def compute_vgi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute sector-level VGI = (CF EV/Rev) / (Public EV/Rev)."""
    out = df.copy()
    out["cf_ev_rev"] = pd.to_numeric(out.get("cf_median_pre_money"), errors="coerce") / pd.to_numeric(
        out.get("cf_median_revenue"), errors="coerce"
    )
    out["VGI"] = out["cf_ev_rev"] / pd.to_numeric(out.get("public_median_ev_rev"), errors="coerce")
    out.replace([float("inf"), -float("inf")], pd.NA, inplace=True)
    return out


def compute_startup_vgi(cf: pd.DataFrame, pc: pd.DataFrame) -> pd.DataFrame:
    """Compute VGI for each startup (CF EV/Rev ÷ public median EV/Rev), safely."""
    per = cf.copy()

    # Normalize startup column name
    if "startup" not in per.columns and "startup_name" in per.columns:
        per = per.rename(columns={"startup_name": "startup"})

    # Ensure numeric types
    for c in ["valuation_pre_money_eur", "revenue_last_fy_eur"]:
        if c in per.columns:
            per[c] = pd.to_numeric(per[c], errors="coerce")

    # Avoid division by zero / negatives
    if "revenue_last_fy_eur" in per.columns:
        per.loc[per["revenue_last_fy_eur"] <= 0, "revenue_last_fy_eur"] = pd.NA

    # CF proxy EV/Rev per startup
    per["startup_ev_rev"] = per.get("valuation_pre_money_eur") / per.get("revenue_last_fy_eur")

    # Use the SAME consistent public medians per sector (MVP: min_n=1)
    pub = sector_public_median_by_consistent_method(pc, min_n=1)

    out = per.merge(pub[["sector", "public_median_ev_rev"]], on="sector", how="left")
    out["VGI"] = out["startup_ev_rev"] / out["public_median_ev_rev"]
    out.replace([float("inf"), -float("inf")], pd.NA, inplace=True)
    return out


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------

def main():
    st.title("Startup Value Monitor — VGI (MVP)")
    st.caption("Comparing crowdfunding valuations with public-market multiples.")

    cf = load_csv("data/crowdfunding_sample.csv")
    pc = load_csv("data/public_comps.csv")

    st.write("Public comps source:", f"**{len(pc)} rows** loaded from `public_comps.csv`")

    df = compute_vgi(sector_summary(cf, pc))
    per_startup = compute_startup_vgi(cf, pc)
    sectors = sorted(per_startup["sector"].dropna().unique().tolist())
    st.caption(f"Startups with valid VGI: {per_startup['VGI'].notna().sum()}")

    # --- Diagnostics: why only 1 row shows per sector -------------------------
    with st.expander("Diagnostics: VGI availability by sector"):
        diag = per_startup.copy()

        # Reasons for missing VGI
        reasons = pd.Series(index=diag.index, dtype="object")
        if "revenue_last_fy_eur" in diag.columns:
            reasons[diag["revenue_last_fy_eur"].isna() | (diag["revenue_last_fy_eur"] <= 0)] = "no/zero revenue"
        if "public_median_ev_rev" in diag.columns:
            mask = diag["public_median_ev_rev"].isna()
            reasons[mask] = (reasons[mask].fillna("") + "; no public median").str.strip("; ").replace("", "no public median")
        if "startup_ev_rev" in diag.columns:
            mask = diag["startup_ev_rev"].isna()
            reasons[mask] = (reasons[mask].fillna("") + "; CF EV/Rev NaN").str.strip("; ")
        mask = diag["VGI"].isna()
        reasons[mask] = (reasons[mask].fillna("") + "; VGI NaN").str.strip("; ")

        # Per-sector counts
        summary = (diag.assign(valid=diag["VGI"].notna())
                      .groupby("sector", as_index=False)
                      .agg(valid_rows=("valid","sum"),
                           total_rows=("startup","count")))
        st.write("Per-sector counts (valid VGI vs total CF rows):")
        st.dataframe(summary.sort_values(["valid_rows","total_rows"], ascending=[False, False]))

        # Sample problematic rows
        bad = diag[diag["VGI"].isna()].copy()
        bad["missing_reason"] = reasons[bad.index]
        cols = [c for c in ["sector","startup","revenue_last_fy_eur",
                            "startup_ev_rev","public_median_ev_rev","VGI","missing_reason"]
                if c in bad.columns]
        st.write("Sample rows with missing VGI:")
        st.dataframe(bad[cols].head(20))
    # --------------------------------------------------------------------------

    st.subheader("Valuation Gap Index (by sector)")
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("sector:N", sort="-y", title="Sector"),
        y=alt.Y("VGI:Q", title="VGI = CF EV/Rev ÷ Public EV/Rev"),
        tooltip=["sector", "cf_ev_rev", "public_median_ev_rev", "VGI"]
    ).properties(height=400)
    st.altair_chart(chart, use_container_width=True)

    median_vgi = df["VGI"].median(skipna=True)
    st.metric(label="Median VGI (all sectors)", value=f"{median_vgi:.2f}" if pd.notna(median_vgi) else "n/a")

    st.subheader("Crowdfunding Pre-Money vs Public EV/Revenue (by sector)")
    scatter = alt.Chart(df).mark_point(size=120).encode(
        x=alt.X("public_median_ev_rev", title="Public EV/Revenue (median)"),
        y=alt.Y("cf_median_pre_money", title="Crowdfunding Pre-Money (median, €)"),
        color="sector",
        tooltip=["sector", "cf_median_pre_money", "cf_median_revenue", "public_median_ev_rev", "VGI"]
    ).properties(height=400)
    st.altair_chart(scatter, use_container_width=True)

    st.caption("Notes:")
    st.markdown(
        "- Uses sample CSVs in `data/`. Replace with real scrapes/APIs.\n"
        "- VGI here is a proxy for demo; production should compute EV/Revenue per round, then aggregate.\n"
        "- All values normalized to EUR in the pipeline (coming next)."
    )


if __name__ == "__main__":
    main()
