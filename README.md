🧮 Startup Value Monitor

An open-source tool to compare the real value of European startups across crowdfunding platforms and public markets.

Startup Value Monitor brings transparency to early-stage finance by tracking how private-market valuations evolve against public-market fundamentals.
The project is fully open source — combining public APIs, community-maintained data, and reproducible methods to make startup valuations measurable, comparable, and explainable.

⸻

📈 Valuation Gap Index (VGI)

The Valuation Gap Index (VGI) is the core benchmark of the project — a simple, reproducible way to compare how startups are priced in crowdfunding rounds relative to their listed peers.

🔹 Definition

VGI = \frac{\text{Median (Crowdfunding EV / Revenue)}}{\text{Median (Public EV / Revenue)}}
	•	VGI = 1.0 → crowdfunding valuations are aligned with public-market multiples.
	•	VGI > 1.0 → startups are priced higher (potential overvaluation).
	•	VGI < 1.0 → startups are priced lower (potential undervaluation).

🔹 Purpose

VGI bridges the gap between private and public market valuations by using open data and a consistent methodology.
It helps investors, founders, and analysts identify where market sentiment diverges from fundamentals.

The metric evolves dynamically as new crowdfunding campaigns or market data arrive — offering a near-real-time indicator of how optimism or caution shifts across sectors such as Fintech, Climate Tech, Healthtech, AI-native, and B2B SaaS.

⸻

🔧 Implementation
	•	Crowdfunding data: community-maintained Google Sheet published as CSV (CROWD_CF_CSV_URL), editable by contributors.
	•	Public comparables: live data fetched via yfinance and stored under data/public_comps.csv.
	•	Dashboard: Streamlit app (app/streamlit_app.py) automatically deployed via Streamlit Cloud.
	•	Data processing: Python scripts under src/ generate sector medians, valuation ratios, and consistency diagnostics.
	•	Open license: all datasets and code under permissive open-source licensing for transparency and reuse.

⸻

🌍 Roadmap
	1.	Expand coverage
	•	Add more European crowdfunding platforms (e.g. Seedrs, CapitalCell, WiSeed, Crowdcube ES).
	2.	Refine metrics
	•	Introduce additional comparables: EV/GMV, EV/ARR, EV/Headcount.
	3.	Automation
	•	Enable scheduled weekly refresh through GitHub Actions + Streamlit Cloud.
	4.	Global expansion
	•	Extend model to North America, LATAM, and APAC with localized comps.
	5.	Community collaboration
	•	Invite analysts, developers, and academics to contribute data sources, code, and visualizations.

⸻

🧠 Credits & Philosophy

This project is inspired by the belief that valuation transparency should be a public good.
By combining open finance data, reproducible Python code, and community-driven methods, we aim to help investors and founders alike see the real signals behind startup hype.

The **Valuation Gap Index (VGI)** measures how startup valuations on crowdfunding platforms compare to public-market benchmarks.

It is defined as:

\[
\text{VGI} = \frac{\text{Median (Crowdfunding EV/Revenue)}}{\text{Median (Public EV/Revenue)}}
\]

- **VGI = 1.0** → crowdfunding valuations are in line with comparable listed companies.  
- **VGI > 1.0** → startups are priced higher (potential overvaluation).  
- **VGI < 1.0** → startups are priced lower (potential undervaluation).

This metric provides a simple, reproducible way to compare sectors and track how market sentiment toward early-stage ventures evolves over time.
