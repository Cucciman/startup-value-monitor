import pandas as pd
import altair as alt
import pathlib
import streamlit as st

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / 'data'

@st.cache_data
def load_data():
    cf = pd.read_csv(DATA_DIR / 'crowdfunding_sample.csv', parse_dates=['round_date'])
    pc = pd.read_csv(DATA_DIR / 'public_comps_sample.csv')
    return cf, pc

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

def main():
    st.set_page_config(page_title='Startup Value Monitor', layout='wide')
    st.title('Startup Value Monitor — VGI (MVP)')
    st.caption('Comparing crowdfunding valuations with public-market multiples (sample data).')

    cf, pc = load_data()
    df = compute_vgi(sector_summary(cf, pc))

    c1, c2 = st.columns([2, 1], vertical_alignment="center")
    with c1:
        st.subheader('Valuation Gap Index (by sector)')
        bar = alt.Chart(df).mark_bar().encode(
            x=alt.X('sector', sort='-y', title='Sector'),
            y=alt.Y('VGI', title='VGI = CF EV/Rev ÷ Public EV/Rev'),
            tooltip=['sector',
                     alt.Tooltip('VGI', format='.2f'),
                     alt.Tooltip('cf_ev_rev', title='CF EV/Rev', format='.2f'),
                     alt.Tooltip('public_median_ev_rev', title='Public EV/Rev', format='.2f')]
        )
        st.altair_chart(bar, use_container_width=True)
    with c2:
        st.metric('Median VGI (all sectors)', f"{df['VGI'].median():.2f}")

    st.markdown('---')
    st.subheader('Crowdfunding Pre-Money vs Public EV/Revenue (by sector)')
    scatter = alt.Chart(df).mark_circle(size=140).encode(
        x=alt.X('public_median_ev_rev', title='Public EV/Revenue (median)'),
        y=alt.Y('cf_median_pre_money', title='Crowdfunding Pre-Money (median, €)'),
        tooltip=['sector',
                 alt.Tooltip('cf_median_pre_money', format=','),
                 alt.Tooltip('public_median_ev_rev', format='.2f'),
                 alt.Tooltip('cf_total_raised', title='CF Total Raised', format=',')]
    )
    st.altair_chart(scatter, use_container_width=True)

    st.markdown('---')
    st.write("**Notes**")
    st.write("- This uses sample CSVs in `data/`. Replace with real scrapes/APIs.")
    st.write("- VGI here is a proxy for demo; production should compute EV/Revenue per round, then aggregate.")
    st.write("- All values normalized to EUR in the pipeline (coming next).")

if __name__ == '__main__':
    main()
