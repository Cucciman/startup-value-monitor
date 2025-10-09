# Startup Value Monitor (MVP)

An open-source tool to compare the real value of European startups across crowdfunding platforms and public markets.

## Methodology
We normalize to EUR, align sectors/stages, compute EV/Revenue for crowdfunding and compare to public comps. The headline metric is the Valuation Gap Index (VGI) = median CF multiple / median public multiple.

---

## ⚖️ Legal & Data Ethics

This project collects and analyzes only **publicly available** data (e.g., crowdfunding campaign pages, exchange listings, official issuer filings). All collection routines will:
- Respect each site’s **robots.txt** and **Terms of Service** (no aggressive scraping).
- Prefer **official/public APIs** where available.
- **Attribute sources** clearly in datasets and notebooks (e.g., “Source: Crowdcube public campaign page, accessed YYYY-MM-DD”).
- **Exclude personal data** (no emails, investor names, or other PII).

The goal is transparency and reproducibility — not commercial data extraction.

---

## 📈 Valuation Gap Index (VGI)

The **Valuation Gap Index (VGI)** measures how startup valuations on crowdfunding platforms compare to public-market benchmarks.

It is defined as:

\[
\text{VGI} = \frac{\text{Median (Crowdfunding EV/Revenue)}}{\text{Median (Public EV/Revenue)}}
\]

- **VGI = 1.0** → crowdfunding valuations are in line with comparable listed companies.  
- **VGI > 1.0** → startups are priced higher (potential overvaluation).  
- **VGI < 1.0** → startups are priced lower (potential undervaluation).

This metric provides a simple, reproducible way to compare sectors and track how market sentiment toward early-stage ventures evolves over time.
