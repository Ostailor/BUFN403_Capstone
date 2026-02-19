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

import os
import re
import sys
import time
import json
import logging
import traceback
import zipfile
from pathlib import Path

import openpyxl
import pandas as pd

# Configuration
SEC_EDGAR_EMAIL = os.environ.get("SEC_EDGAR_EMAIL", "enteryouremail@gmail.com")

# Paths
BASE_DIR = Path(__file__).resolve().parent
SPREADSHEET = BASE_DIR / "Copy of Top 50 US Banks by Assets.xlsx"
DATA_DIR = BASE_DIR / "data"
SEC_DIR = DATA_DIR / "sec_filings"
FIN_DIR = DATA_DIR / "financials"

# Tickers that were recently delisted (acquired) — map to their CIK numbers
# so sec-edgar-downloader can still find their historical filings.
DELISTED_TICKER_TO_CIK = {
    "DFS": "0001393612",   # Discover Financial — acquired by Capital One (COF), May 2025
    "CMA": "0000028412",   # Comerica — acquired by Fifth Third Bancorp (FITB), Feb 2026
    "SNV": "0000018349",   # Synovus — acquired by Pinnacle Financial (PNFP), Jan 2026
}

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


# Step 1: Read tickers from spreadsheet
def read_tickers() -> list[str]:
    """Read ticker symbols from the Top 50 US Banks spreadsheet."""
    wb = openpyxl.load_workbook(SPREADSHEET, read_only=True)
    ws = wb.active

    tickers = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        ticker = row[2]  # Column C = Ticker
        if ticker and isinstance(ticker, str) and ticker.strip():
            tickers.append(ticker.strip())

    wb.close()
    log.info(f"Read {len(tickers)} tickers from spreadsheet")
    return tickers


# Step 2: Download SEC filings
def download_sec_filings(tickers: list[str]):
    """Download 10-K, 10-Q, DEF 14A, and 8-K filings using sec-edgar-downloader."""
    from sec_edgar_downloader import Downloader

    dl = Downloader("BUFN403", SEC_EDGAR_EMAIL, str(SEC_DIR))

    filings_config = [
        ("10-K", 2),
        ("10-Q", 8),
        ("DEF 14A", 2),
        ("8-K", 8),
    ]

    failures = []
    for ticker in tickers:
        # Use CIK for delisted tickers that can't be resolved by ticker symbol
        lookup = DELISTED_TICKER_TO_CIK.get(ticker, ticker)
        for form_type, limit in filings_config:
            try:
                dl.get(form_type, lookup, limit=limit)
            except Exception as e:
                failures.append(f"{ticker}/{form_type}: {e}")
            time.sleep(0.15)

    if failures:
        log.error(f"SEC filing failures ({len(failures)}):")
        for f in failures:
            log.error(f"  {f}")
    log.info(f"SEC filings: downloaded for {len(tickers)} tickers, {len(failures)} failures")


# Step 3: Extract structured financials via edgartools
def extract_financials(tickers: list[str]):
    """Use edgartools to parse financial statements and save as CSV."""
    from edgar import Company, set_identity

    set_identity(SEC_EDGAR_EMAIL)

    success_count = 0
    skip_count = 0
    failures = []

    for ticker in tickers:
        out_dir = FIN_DIR / ticker
        out_dir.mkdir(parents=True, exist_ok=True)

        expected = ["balance_sheet.csv", "income_statement.csv", "cash_flow.csv"]
        if all((out_dir / f).exists() for f in expected):
            skip_count += 1
            continue

        try:
            company = Company(ticker)
            filings = company.get_filings(form=["10-K", "10-Q"]).latest(10)

            all_bs = []
            all_is = []
            all_cf = []

            for filing in filings:
                try:
                    obj = filing.obj()
                    if obj is None:
                        continue

                    if hasattr(obj, "balance_sheet") and obj.balance_sheet is not None:
                        bs = obj.balance_sheet
                        if hasattr(bs, "to_dataframe"):
                            all_bs.append(bs.to_dataframe())
                        elif isinstance(bs, pd.DataFrame):
                            all_bs.append(bs)

                    if hasattr(obj, "income_statement") and obj.income_statement is not None:
                        ist = obj.income_statement
                        if hasattr(ist, "to_dataframe"):
                            all_is.append(ist.to_dataframe())
                        elif isinstance(ist, pd.DataFrame):
                            all_is.append(ist)

                    if hasattr(obj, "cash_flow_statement") and obj.cash_flow_statement is not None:
                        cf = obj.cash_flow_statement
                        if hasattr(cf, "to_dataframe"):
                            all_cf.append(cf.to_dataframe())
                        elif isinstance(cf, pd.DataFrame):
                            all_cf.append(cf)

                except Exception:
                    pass

                time.sleep(0.15)

            saved_any = False
            if all_bs:
                pd.concat(all_bs, ignore_index=True).to_csv(out_dir / "balance_sheet.csv", index=False)
                saved_any = True
            if all_is:
                pd.concat(all_is, ignore_index=True).to_csv(out_dir / "income_statement.csv", index=False)
                saved_any = True
            if all_cf:
                pd.concat(all_cf, ignore_index=True).to_csv(out_dir / "cash_flow.csv", index=False)
                saved_any = True

            if saved_any:
                success_count += 1
            else:
                failures.append(f"{ticker}: no financial data extracted from filings")

        except Exception as e:
            failures.append(f"{ticker}: {e}")

    if failures:
        log.error(f"Financials failures ({len(failures)}):")
        for f in failures:
            log.error(f"  {f}")
    log.info(f"Financials: {success_count} extracted, {skip_count} skipped, {len(failures)} failed")


# Step 4: Rename SEC filing folders to friendly names
def rename_sec_folders():
    """Rename accession-number folders to human-readable names like JPM_10-K_2025-02-21."""
    edgar_root = SEC_DIR / "sec-edgar-filings"
    if not edgar_root.exists():
        log.warning("No sec-edgar-filings directory found, skipping rename step")
        return

    renamed_count = 0
    failures = []
    for ticker_dir in sorted(edgar_root.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name

        for form_dir in sorted(ticker_dir.iterdir()):
            if not form_dir.is_dir():
                continue
            form_type = form_dir.name

            for accession_dir in sorted(form_dir.iterdir()):
                if not accession_dir.is_dir():
                    continue

                if accession_dir.name.startswith(ticker):
                    continue

                filing_date = None
                metadata_file = accession_dir / "filing-details.json"
                if metadata_file.exists():
                    try:
                        meta = json.loads(metadata_file.read_text())
                        filing_date = meta.get("filingDate") or meta.get("filing_date")
                    except Exception:
                        pass

                if not filing_date:
                    for jf in accession_dir.glob("*.json"):
                        try:
                            meta = json.loads(jf.read_text())
                            if isinstance(meta, dict):
                                filing_date = meta.get("filingDate") or meta.get("filing_date")
                                if filing_date:
                                    break
                        except Exception:
                            continue

                if not filing_date:
                    match = re.search(r"-(\d{2})-\d+$", accession_dir.name)
                    if match:
                        yy = int(match.group(1))
                        year = 2000 + yy
                        filing_date = str(year)

                form_clean = form_type.replace(" ", "").replace("/", "-")
                if filing_date:
                    friendly = f"{ticker}_{form_clean}_{filing_date}"
                else:
                    friendly = f"{ticker}_{form_clean}_{accession_dir.name}"

                new_path = form_dir / friendly
                suffix = 1
                while new_path.exists():
                    suffix += 1
                    new_path = form_dir / f"{friendly}_{suffix}"

                try:
                    accession_dir.rename(new_path)
                    renamed_count += 1
                except Exception as e:
                    failures.append(f"{accession_dir.name}: {e}")

    if failures:
        log.error(f"Rename failures ({len(failures)}):")
        for f in failures:
            log.error(f"  {f}")
    log.info(f"Renamed {renamed_count} filing folders, {len(failures)} failed")


# Step 5: Zip the data folder
def zip_data_folder():
    """Compress the entire data/ directory into data.zip using LZMA (best compression)."""
    zip_path = BASE_DIR / "data.zip"

    if zip_path.exists():
        zip_path.unlink()

    file_count = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_LZMA) as zf:
        for file in sorted(DATA_DIR.rglob("*")):
            if file.is_file():
                arcname = file.relative_to(BASE_DIR)
                zf.write(file, arcname)
                file_count += 1

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    log.info(f"Created {zip_path.name}: {file_count} files, {size_mb:.1f} MB")


# Main
def main():
    log.info("Starting bank data collection")

    for d in [SEC_DIR, FIN_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    tickers = read_tickers()
    if not tickers:
        log.error("No tickers found in spreadsheet!")
        sys.exit(1)

    log.info(f"Processing {len(tickers)} tickers")

    log.info("STEP 1: Downloading SEC filings...")
    download_sec_filings(tickers)

    log.info("STEP 2: Extracting structured financials...")
    extract_financials(tickers)

    log.info("STEP 3: Renaming SEC filing folders...")
    rename_sec_folders()

    log.info("STEP 4: Zipping data folder...")
    zip_data_folder()

    log.info("Collection complete! Check ./data/ and data.zip for results.")


if __name__ == "__main__":
    main()
