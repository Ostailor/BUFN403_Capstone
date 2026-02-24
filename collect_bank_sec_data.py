"""
Collect financial data for the Top 50 US Banks by Assets.

Downloads:
  1. SEC filings (10-K, 10-Q, DEF 14A, 8-K) via sec-edgar-downloader
  2. Structured XBRL financials via edgartools

Usage:
  1. pip install -r requirements.txt
  2. Set SEC_EDGAR_EMAIL below (or as env var)
  3. python collect_bank_data.py
"""

import os, re, sys, logging, zipfile, time
from pathlib import Path
import pandas as pd

from sec_edgar_downloader import Downloader

# Configuration
email = 'abc@email.com'
SEC_EDGAR_EMAIL = os.environ.get("SEC_EDGAR_EMAIL", email)

# Paths
BASE_DIR = Path(__file__).resolve().parent
CSV = BASE_DIR / "banks.csv"
DATA_DIR = BASE_DIR / "data"
SEC_DIR = DATA_DIR / "sec-edgar-filings"

df = pd.read_csv(CSV)
tickers = df['Ticker']

# Tickers that were recently delisted (acquired) — map to their CIK numbers
# so sec-edgar-downloader can still find their historical filings.
DELISTED_TICKER_TO_CIK = {
    "DFS": "0001393612",   # Discover Financial — acquired by Capital One (COF), May 2025
    "CMA": "0000028412",   # Comerica — acquired by Fifth Third Bancorp (FITB), Feb 2026
    "SNV": "0000018349",   # Synovus — acquired by Pinnacle Financial (PNFP), Jan 2026
    "PNFP": "0001115055",  # original TN entity; ticker now resolves to post-merger GA shell
}

filings_config = [
    ("10-K", 2),
    ("10-Q", 8),
    ("DEF 14A", 2),
    ("8-K", 8),
]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "collect_bank_data.log"),
    ],
)
log = logging.getLogger(__name__)

max_attempts = 5
download_failures = []
incomplete_tickers = []

# Step 1: Download SEC filings
def download_sec_filings(tickers, attempt_num=1, retry=False):
    """Download 10-K, 10-Q, DEF 14A, and 8-K filings using sec-edgar-downloader."""
    dl = Downloader("BUFN403", SEC_EDGAR_EMAIL, str(DATA_DIR))

    for ticker in tickers:
        # Use CIK for delisted tickers that can't be resolved by ticker symbol
        lookup = DELISTED_TICKER_TO_CIK.get(ticker, ticker)
        for form_type, limit in filings_config:
            try:
                dl.get(form_type, lookup, limit=limit, download_details=True)
            except Exception as e:
                download_failures.append(f"{ticker}/{form_type}: {e}")
                
        # checking download integrity
        incomplete_ticker = _ticker_filings_ok(SEC_DIR / lookup)

        # passes if None returned i.e. correct install
        if incomplete_ticker:
            if attempt_num <= max_attempts:
                log.error(f"At least one filing not installed: {incomplete_ticker}, Attempt {attempt_num} of {max_attempts}...")
                time.sleep(0.5)
                download_sec_filings([ticker], attempt_num=attempt_num + 1, retry=True)
            else:
                log.error(f"All filings could not be installed for: {incomplete_ticker}")
                incomplete_tickers.append(incomplete_ticker)
                continue

    if not retry:
        if download_failures:
            log.error(f"SEC ticker filing failures ({len(download_failures)}):")
            for f in download_failures:
                log.error(f"  {f}")
        
        if incomplete_tickers:
            log.error(f"Incomplete Ticker Downloads: {incomplete_tickers}")
        
        log.info(f"SEC filings: downloaded for {len(tickers)} tickers, {len(download_failures)} failures")

def _ticker_filings_ok(ticker_dir):
    """check correct download of expected number of files
    returns ticker where error happened or None if everything was correctly downloaded
    """
    ticker = ticker_dir.name

    for form_type, expected_count in filings_config:
        form_dir = ticker_dir / form_type
        if not form_dir.exists():
            return ticker

        filing_dirs = [d for d in form_dir.iterdir() if d.is_dir()]
        if len(filing_dirs) != expected_count:
            return ticker

        for filing in filing_dirs:
            for f in filing.rglob("*"):
                if f.is_file() and f.stat().st_size == 0:
                    return ticker

    return None

def _parse_filing_header(accession_dir):
    """Read the first 30 lines of full-submission.txt and return
    (period_of_report, fiscal_year_end) as raw strings, e.g. ('20250930', '1231').
    Either value is None if not found.
    """
    submission_file = accession_dir / "full-submission.txt"
    try:
        with open(submission_file, "r", errors="ignore") as f:
            period, fy_end = None, None
            for i, line in enumerate(f):
                if i >= 30:
                    break

                if period is None:
                    m1 = re.search(r"CONFORMED PERIOD OF REPORT:\s+(\d{8})", line)
                    if m1:
                        period = m1.group(1)

                if fy_end is None:
                    m2 = re.search(r"FISCAL YEAR END:\s+(\d{4})", line)
                    if m2:
                        fy_end = m2.group(1)
                
                if period and fy_end:
                    break

    except Exception as e:
        log.error(f"Fiscal Year End could not be read from file: {accession_dir}")
        pass

    return period, fy_end

def _fiscal_quarter(period_str, fy_end_str):
    """Return the fiscal quarter (1-4) for a given period end date.
    Uses FISCAL YEAR END to compute true fiscal quarters so non-calendar
    fiscal years are handled correctly (e.g. FY ending Jan 31 means their
    Q4 ends in January, not December).
    """
    period_month = int(period_str[4:6])
    fy_end_month = int(fy_end_str[:2])

    return 4 - (abs(period_month - fy_end_month) % 12) // 3

rename_failures=[]
# Step 2: Rename SEC filing folders to friendly names
def rename_sec_folders():
    """Rename accession-number folders to readable names like JPM_10-Q_Q3_2025."""
    renamed_count = 0
    for ticker_dir in sorted(SEC_DIR.iterdir()):
        ticker = ticker_dir.name
                
        # rename delisted companies' CIK-named dirs back to their ticker symbols
        if ticker in DELISTED_TICKER_TO_CIK.values():
            ticker = next(k for k, v in DELISTED_TICKER_TO_CIK.items() if v == ticker)
            ticker_dir = ticker_dir.rename(SEC_DIR / ticker)
            log.info(f"Renamed CIK directory to ticker: {ticker}")

        for form_dir in sorted(ticker_dir.iterdir()):
            form_type = form_dir.name

            for accession_dir in sorted(form_dir.iterdir()):
                try:
                    period_str, fy_end_str = _parse_filing_header(accession_dir)
                    year, quarter = period_str[:4], _fiscal_quarter(period_str, fy_end_str)

                    form_clean = form_type.replace(" ", "").replace("/", "-")
                    friendly = f"{ticker}_{form_clean}_{year}_Q{quarter}"

                    new_path = form_dir / friendly

                    # handle collisions (e.g. multiple 8-Ks in the same quarter)
                    if new_path.exists():
                        suffix = 2
                        while (form_dir / f"{friendly}_{suffix}").exists():
                            suffix += 1
                        new_path = form_dir / f"{friendly}_{suffix}"

                    accession_dir.rename(new_path)
                    renamed_count += 1
                except Exception as e:
                    rename_failures.append(f"{accession_dir.name}: {e}")

    if rename_failures:
        log.error(f"Rename failures ({len(rename_failures)}):")
        for f in rename_failures:
            log.error(f"  {f}")
    log.info(f"Renamed {renamed_count} filing folders, {len(rename_failures)} failed")

# Step 3: Zip the data folder
def zip_data_folder():
    """Compress the entire data/ directory into data.zip"""
    zip_path = BASE_DIR / "data.zip"

    file_count = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(DATA_DIR.rglob("*")):
            arcname = file.relative_to(BASE_DIR)
            zf.write(file, arcname)
            file_count += 1

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    log.info(f"Created {zip_path.name}: {file_count} files, {size_mb:.1f} MB")

# Main
def main():
    log.info(f"Processing {len(tickers)} tickers")

    log.info("STEP 1: Downloading SEC filings...")
    download_sec_filings(tickers)

    log.info("STEP 2: Renaming SEC filing folders...")
    rename_sec_folders()

    if len(rename_failures) or len(download_failures) or len(incomplete_tickers):
        log.info("Manual download or naming changes may be required before zipping")
    else:
        log.info("STEP 3: Zipping data folder...")
        zip_data_folder()

    log.info("Collection complete! Check ./data/ and data.zip for results.")

if __name__ == "__main__":
    main()
