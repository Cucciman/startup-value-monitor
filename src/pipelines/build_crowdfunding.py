import os
import pandas as pd
from pathlib import Path

from ingest import scrape_crowdcube

DATA_DIR = Path("data")
OUTFILE = DATA_DIR / "crowdfunding_live.csv"

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def unify_frames(frames):
    cols = set()
    for f in frames:
        cols |= set(f.columns)
    cols = list(cols)
    merged = pd.concat([f.reindex(columns=cols) for f in frames], ignore_index=True)
    return merged

def main():
    # You can pass SVM_CROWDCUBE_LIST_URL env to enable real scraping
    cc_url = os.getenv("SVM_CROWDCUBE_LIST_URL")
    cc = scrape_crowdcube(list_url=cc_url, max_pages=1, sleep_s=1.0)

    all_ = unify_frames([cc])
    ensure_dir(OUTFILE)
    all_.to_csv(OUTFILE, index=False)
    print(f"Wrote {len(all_)} rows to {OUTFILE}")

if __name__ == "__main__":
    main()
