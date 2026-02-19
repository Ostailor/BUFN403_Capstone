"""
Scrape earnings call transcripts from Motley Fool.

Discovers transcript URLs via Google site-search, then fetches and
parses the full transcript text from each page.

Usage:
  1. pip install -r requirements.txt
  2. python scrape_transcripts.py          # scrape all tickers from spreadsheet
  3. python scrape_transcripts.py JPM BAC  # scrape specific tickers only
"""

import re
import sys
import time
import random
import logging
from pathlib import Path
from urllib.parse import quote

import openpyxl
import requests
from bs4 import BeautifulSoup

# Paths
BASE_DIR = Path(__file__).resolve().parent
SPREADSHEET = BASE_DIR / "Copy of Top 50 US Banks by Assets.xlsx"
DATA_DIR = BASE_DIR / "data"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"

# Target quarters
TARGET_QUARTERS = set()
for year in [2024, 2025]:
    for q in [1, 2, 3, 4]:
        TARGET_QUARTERS.add((year, q))

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "scrape_transcripts.log"),
    ],
)
log = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
]

# Regex to extract ticker + quarter from URL slugs
# Handles both "-jpm-q4-2025-earnings-call-transcript" and "-jpm-q4-2025-earnings-transcript"
SLUG_RE = re.compile(r"-([a-z]+)-q(\d)-(\d{4})-earnings(?:-call)?-transcript")


def _headers():
    return {"User-Agent": random.choice(USER_AGENTS)}


def _polite_sleep(lo=2, hi=5):
    time.sleep(random.uniform(lo, hi))


def read_tickers() -> list[str]:
    """Read ticker symbols from the Top 50 US Banks spreadsheet."""
    wb = openpyxl.load_workbook(SPREADSHEET, read_only=True)
    ws = wb.active
    tickers = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        ticker = row[2]
        if ticker and isinstance(ticker, str) and ticker.strip():
            tickers.append(ticker.strip().upper())
    wb.close()
    return tickers


def discover_urls_for_ticker(ticker: str, needed: set[tuple[int, int]]) -> dict[str, str]:
    """
    Use Google site-search to find Motley Fool transcript URLs for a ticker.

    Returns dict mapping "TICKER_YYYY_QN" -> URL for quarters in `needed`.
    """
    found = {}
    query = quote(f"site:fool.com/earnings/call-transcripts {ticker} earnings call transcript")
    search_url = f"https://www.google.com/search?q={query}&num=20"

    try:
        resp = requests.get(search_url, headers=_headers(), timeout=30)
        if resp.status_code != 200:
            return found

        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract all fool.com transcript URLs from search results
        for a in soup.find_all("a", href=True):
            href = a["href"]

            # Google wraps URLs in /url?q=...&sa=... redirects
            if "/url?q=" in href:
                href = href.split("/url?q=")[1].split("&")[0]

            if "fool.com/earnings/call-transcripts/" not in href:
                continue

            # Try to match ticker + quarter from the slug
            m = SLUG_RE.search(href.lower())
            if not m:
                continue

            slug_ticker = m.group(1).upper()
            quarter = int(m.group(2))
            year = int(m.group(3))

            if slug_ticker != ticker:
                continue
            if (year, quarter) not in needed:
                continue

            key = f"{ticker}_{year}_Q{quarter}"
            if key not in found:
                found[key] = href

    except Exception as e:
        log.error(f"Google search failed for {ticker}: {e}")

    return found


def discover_urls_from_listing(tickers: set[str], needed: dict[str, set], already_found: dict[str, str]) -> dict[str, str]:
    """
    Fallback: crawl the Motley Fool listing pages to find any transcripts
    not discovered via Google search.
    """
    total_still_needed = 0
    for ticker in tickers:
        ticker_needed = needed.get(ticker, set())
        for year, quarter in ticker_needed:
            key = f"{ticker}_{year}_Q{quarter}"
            if key not in already_found:
                total_still_needed += 1

    if total_still_needed == 0:
        return {}

    log.info(f"Crawling listing pages for {total_still_needed} remaining transcripts...")
    found = {}
    tickers_lower = {t.lower() for t in tickers}
    page = 1
    max_pages = 300
    empty_streak = 0

    while page <= max_pages and len(found) < total_still_needed:
        url = f"https://www.fool.com/earnings-call-transcripts/?page={page}"
        try:
            resp = requests.get(url, headers=_headers(), timeout=30)
            if resp.status_code != 200:
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            links = soup.find_all("a", href=re.compile(r"/earnings/call-transcripts/20"))

            if not links:
                empty_streak += 1
                if empty_streak >= 3:
                    break
                page += 1
                _polite_sleep()
                continue

            empty_streak = 0
            seen = set()

            for link in links:
                href = link.get("href", "")
                if href in seen:
                    continue
                seen.add(href)

                m = SLUG_RE.search(href.lower())
                if not m:
                    continue

                ticker = m.group(1).upper()
                quarter = int(m.group(2))
                year = int(m.group(3))

                if ticker.lower() not in tickers_lower:
                    continue
                if (year, quarter) not in needed.get(ticker, set()):
                    continue

                key = f"{ticker}_{year}_Q{quarter}"
                if key in already_found or key in found:
                    continue

                full_url = href if href.startswith("http") else f"https://www.fool.com{href}"
                found[key] = full_url

            # Check if we've gone past our date range (before 2024)
            dates = []
            for link in links:
                dm = re.search(r"/call-transcripts/(\d{4})/", link.get("href", ""))
                if dm:
                    dates.append(int(dm.group(1)))
            if dates and max(dates) < 2024:
                log.info(f"Reached transcripts from before 2024 at page {page}, stopping")
                break

        except Exception as e:
            log.error(f"Error crawling page {page}: {e}")

        if page % 20 == 0:
            log.info(f"  Listing page {page}, found {len(found)} additional transcripts")

        page += 1
        _polite_sleep(1.5, 3)

    log.info(f"Listing crawl: found {len(found)} additional transcripts across {page - 1} pages")
    return found


def parse_transcript(url: str) -> str | None:
    """Fetch a transcript page and extract the full transcript text."""
    try:
        resp = requests.get(url, headers=_headers(), timeout=30)
        resp.raise_for_status()
    except Exception as e:
        log.error(f"Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find the transcript body container
    body = soup.find("div", id="article-body-transcript")
    if not body:
        body = soup.find("div", class_="article-body")
    if not body:
        return None

    # Look for "Full Conference Call Transcript" heading
    transcript_heading = None
    for h2 in body.find_all("h2"):
        if "full conference call transcript" in h2.get_text(strip=True).lower():
            transcript_heading = h2
            break

    if transcript_heading:
        parts = []
        sibling = transcript_heading.find_next_sibling()
        while sibling:
            if sibling.name == "p":
                text = sibling.get_text(strip=True)
                if text:
                    parts.append(text)
            sibling = sibling.find_next_sibling()
        if parts:
            return "\n\n".join(parts)

    # Fallback: extract all paragraph text from the body
    parts = []
    for p in body.find_all("p"):
        text = p.get_text(strip=True)
        if text:
            parts.append(text)

    return "\n\n".join(parts) if parts else None


def scrape_transcripts(tickers: list[str]):
    """Main scraping workflow: discover URLs via search, then download transcripts."""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    # Figure out what we still need
    needed = {}
    for ticker in tickers:
        out_dir = TRANSCRIPT_DIR / ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        missing = set()
        for year, quarter in TARGET_QUARTERS:
            filename = out_dir / f"{year}-Q{quarter}.txt"
            if not filename.exists():
                missing.add((year, quarter))
        if missing:
            needed[ticker] = missing

    total_needed = sum(len(v) for v in needed.values())
    if total_needed == 0:
        log.info("All transcripts already downloaded!")
        return

    log.info(f"Need {total_needed} transcripts across {len(needed)} tickers")

    # Step 1: Discover URLs via Google search (fast, targeted)
    url_map = {}
    for i, ticker in enumerate(needed.keys(), 1):
        results = discover_urls_for_ticker(ticker, needed[ticker])
        url_map.update(results)
        if i % 5 == 0:
            log.info(f"  Searched {i}/{len(needed)} tickers, found {len(url_map)} URLs so far")
        _polite_sleep(3, 7)  # Be polite to Google

    log.info(f"Google search found {len(url_map)}/{total_needed} transcript URLs")

    # Step 2: Fallback listing crawl for anything Google missed
    if len(url_map) < total_needed:
        extra = discover_urls_from_listing(set(tickers), needed, url_map)
        url_map.update(extra)

    if not url_map:
        log.error("No transcript URLs found.")
        return

    # Step 3: Download and save each transcript
    saved_count = 0
    failures = []

    for key, url in sorted(url_map.items()):
        parts = key.split("_")
        ticker = parts[0]
        year = parts[1]
        quarter = parts[2]

        out_file = TRANSCRIPT_DIR / ticker / f"{year}-{quarter}.txt"
        if out_file.exists():
            continue

        transcript = parse_transcript(url)
        _polite_sleep()

        if transcript:
            out_file.write_text(transcript, encoding="utf-8")
            saved_count += 1
        else:
            failures.append(key)

    not_found = total_needed - len(url_map)
    if failures:
        log.error(f"Failed to parse ({len(failures)}):")
        for f in failures:
            log.error(f"  {f}")
    if not_found > 0:
        log.warning(f"{not_found} transcripts not found on Motley Fool (may not be published yet)")
    log.info(f"Transcripts: {saved_count} saved, {len(failures)} parse failures, {not_found} not found")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        tickers = [t.upper() for t in sys.argv[1:]]
        log.info(f"Scraping transcripts for: {', '.join(tickers)}")
    else:
        tickers = read_tickers()
        log.info(f"Read {len(tickers)} tickers from spreadsheet")

    scrape_transcripts(tickers)
