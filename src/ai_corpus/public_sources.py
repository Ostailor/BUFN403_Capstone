from __future__ import annotations

import difflib
from dataclasses import dataclass
from datetime import date
from typing import Any

import requests

from .config import CALL_REPORT_FIELDS, DEFAULT_AS_OF_DATE
from .utils import normalize_key, quarter_end_date

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_no_zero}/{accession_compact}/{primary_document}"
FDIC_INSTITUTIONS_URL = "https://api.fdic.gov/banks/institutions"
FDIC_FINANCIALS_URL = "https://api.fdic.gov/banks/financials"


@dataclass(slots=True)
class SecFilingRecord:
    ticker: str
    form_type: str
    filing_date: str
    accession_number: str
    primary_document: str
    cik: str
    period_year: int
    period_quarter: int
    source_url: str


@dataclass(slots=True)
class FdicInstitutionMatch:
    ticker: str
    bank_name: str
    cert: int
    institution_name: str
    holding_company: str
    city: str
    state: str
    confidence: float


class SecClient:
    def __init__(self, user_agent: str = "BUFN403 Capstone omtailor@example.com") -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self._ticker_map: dict[str, str] | None = None

    def ticker_map(self) -> dict[str, str]:
        if self._ticker_map is None:
            response = self.session.get(SEC_TICKERS_URL, timeout=30)
            response.raise_for_status()
            payload = response.json()
            self._ticker_map = {
                row["ticker"].upper(): f"{int(row['cik_str']):010d}"
                for row in payload.values()
                if isinstance(row, dict) and row.get("ticker")
            }
        return self._ticker_map

    def cik_for_ticker(self, ticker: str) -> str | None:
        return self.ticker_map().get(ticker.upper())

    def list_filings(
        self,
        ticker: str,
        *,
        as_of: date = DEFAULT_AS_OF_DATE,
        forms: set[str] | None = None,
    ) -> list[SecFilingRecord]:
        cik = self.cik_for_ticker(ticker)
        if not cik:
            return []
        response = self.session.get(SEC_SUBMISSIONS_URL.format(cik=cik), timeout=30)
        response.raise_for_status()
        payload = response.json()
        recent = payload.get("filings", {}).get("recent", {})
        result: list[SecFilingRecord] = []
        total = len(recent.get("form", []))
        for index in range(total):
            form = recent["form"][index]
            if forms and form not in forms:
                continue
            filing_date = recent["filingDate"][index]
            filing_dt = date.fromisoformat(filing_date)
            if filing_dt > as_of:
                continue
            accession = recent["accessionNumber"][index]
            primary_document = recent["primaryDocument"][index]
            accession_compact = accession.replace("-", "")
            quarter = ((filing_dt.month - 1) // 3) + 1
            result.append(
                SecFilingRecord(
                    ticker=ticker.upper(),
                    form_type=form,
                    filing_date=filing_date,
                    accession_number=accession,
                    primary_document=primary_document,
                    cik=cik,
                    period_year=filing_dt.year,
                    period_quarter=quarter,
                    source_url=SEC_ARCHIVE_URL.format(
                        cik_no_zero=str(int(cik)),
                        accession_compact=accession_compact,
                        primary_document=primary_document,
                    ),
                )
            )
        return result


class FdicClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self._institutions: list[dict[str, Any]] | None = None

    def list_institutions(self) -> list[dict[str, Any]]:
        if self._institutions is not None:
            return self._institutions
        rows: list[dict[str, Any]] = []
        offset = 0
        limit = 10000
        while True:
            response = self.session.get(
                FDIC_INSTITUTIONS_URL,
                params={
                    "format": "json",
                    "limit": limit,
                    "offset": offset,
                    "fields": "NAME,CERT,NAMEHCR,CITY,STALP",
                },
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data", [])
            if not data:
                break
            rows.extend(item.get("data", {}) for item in data if isinstance(item, dict))
            offset += limit
            if offset >= int(payload.get("meta", {}).get("total", 0)):
                break
        self._institutions = rows
        return rows

    def match_bank(self, ticker: str, bank_name: str) -> FdicInstitutionMatch | None:
        wanted = normalize_key(bank_name.replace("corp", "").replace("corporation", ""))
        best_row: dict[str, Any] | None = None
        best_score = 0.0
        for row in self.list_institutions():
            name = normalize_key(str(row.get("NAME", "")))
            holding = normalize_key(str(row.get("NAMEHCR", "")))
            combined = max(
                difflib.SequenceMatcher(a=wanted, b=name).ratio(),
                difflib.SequenceMatcher(a=wanted, b=holding).ratio(),
            )
            if ticker.upper() in holding.replace(" ", ""):
                combined += 0.05
            if combined > best_score:
                best_score = combined
                best_row = row
        if not best_row or best_score < 0.55:
            return None
        return FdicInstitutionMatch(
            ticker=ticker.upper(),
            bank_name=bank_name,
            cert=int(best_row["CERT"]),
            institution_name=str(best_row.get("NAME", "")),
            holding_company=str(best_row.get("NAMEHCR", "")),
            city=str(best_row.get("CITY", "")),
            state=str(best_row.get("STALP", "")),
            confidence=round(best_score, 2),
        )

    def list_call_reports(self, cert: int) -> list[dict[str, Any]]:
        response = self.session.get(
            FDIC_FINANCIALS_URL,
            params={
                "format": "json",
                "limit": 200,
                "sort_by": "REPDTE",
                "sort_order": "DESC",
                "fields": ",".join(CALL_REPORT_FIELDS),
                "filters": f"CERT:{cert}",
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        return [item.get("data", {}) for item in payload.get("data", []) if isinstance(item, dict)]

    def find_call_report_rows(self, cert: int, periods: list[tuple[int, int]]) -> dict[tuple[int, int], dict[str, Any]]:
        period_lookup = {quarter_end_date(year, quarter): (year, quarter) for year, quarter in periods}
        rows = {}
        for row in self.list_call_reports(cert):
            repdte = str(row.get("REPDTE", ""))
            if repdte in period_lookup:
                rows[period_lookup[repdte]] = row
        return rows
