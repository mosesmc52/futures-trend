from __future__ import annotations

import csv
import io
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import requests
from fake_useragent import UserAgent
from lxml import html


# -----------------------------
# Data model
# -----------------------------
@dataclass(frozen=True)
class StooqBar:
    date: str
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    volume: Optional[float]
    open_interest: Optional[float]


# -----------------------------
# Downloader + incremental updater
# -----------------------------
class StooqDownloader:
    """
    Downloader for historical OHLCV data from stooq.com.

    Update workflow:
      1) Read existing CSV (if present) and find last date
      2) Crawl Stooq and collect only rows newer than last date
      3) Append and rewrite CSV sorted by date

    Notes:
      - Your sample CSV date format is "23 Apr 1993" (HTML style).
      - We support BOTH "23 Apr 1993" and "1993-04-23" when reading.
      - We WRITE dates in HTML style by default ("%d %b %Y") to match your sample.
    """

    BASE_URL = "https://stooq.com"
    HTML_PATH = "/q/d/"
    CSV_PATH = "/q/d/l/"

    def __init__(
        self,
        timeout: int = 30,
        max_pages: int = 2000,
        page_delay: float = 1.0,  # seconds between pages
        page_jitter: float = 0.0,  # optional random jitter
    ):
        self.timeout = timeout
        self.max_pages = max_pages
        self.page_delay = page_delay
        self.page_jitter = page_jitter

        self.session = requests.Session()
        ua = UserAgent(platforms="desktop")
        self.session.headers.update(
            {
                "User-Agent": ua.chrome,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )

    # ============================================================
    # Public API
    # ============================================================

    def update_prices(
        self, symbol: str, path: str, interval: str = "d"
    ) -> List[StooqBar]:
        """
        Incrementally update `path` with any bars newer than the last date in the file.

        - If `path` doesn't exist: downloads full history (HTML crawl) and writes it.
        - If `path` exists: reads last date, crawls until reaching that date, appends newer bars.

        Returns: full series (existing + new), sorted ascending by date.
        """
        symbol = symbol.lower().strip()
        interval = interval.lower().strip()

        existing, last_dt, date_fmt = self._read_existing_csv(path)

        # No file (or no valid date) => pull all history
        if last_dt is None:
            bars = self._crawl_html(symbol, interval, stop_at_dt=None)
            self.write_csv(bars, path, date_format=date_fmt or "%d %b %Y")
            return bars

        # Pull only newer bars (HTML paging is efficient because we stop once we hit old data)
        new_bars = self._crawl_html(symbol, interval, stop_at_dt=last_dt)

        if not new_bars:
            # no updates
            return existing

        # Merge, dedupe by dt, sort
        merged = self._merge_bars(
            existing, new_bars, out_date_format=date_fmt or "%d %b %Y"
        )
        self.write_csv(merged, path, date_format=date_fmt or "%d %b %Y")
        return merged

    def fetch(self, symbol: str, interval: str = "d") -> List[StooqBar]:
        """
        Fetch full historical series for symbol (HTML crawl).
        interval: d / w / m / q / y
        """
        symbol = symbol.lower().strip()
        interval = interval.lower().strip()
        return self._crawl_html(symbol, interval, stop_at_dt=None)

    def write_csv(
        self, bars: List[StooqBar], path: str, *, date_format: str = "%d %b %Y"
    ) -> None:
        """
        Writes CSV with header:
          Date,Open,High,Low,Close,Volume,OpenInterest
        Date is formatted with `date_format`.
        """
        # normalize to requested output format if possible
        normalized: List[StooqBar] = []
        for b in bars:
            dt = self._parse_any_date(b.date)
            date_out = dt.strftime(date_format) if dt else b.date
            normalized.append(
                StooqBar(
                    date=date_out,
                    open=b.open,
                    high=b.high,
                    low=b.low,
                    close=b.close,
                    volume=b.volume,
                    open_interest=b.open_interest,
                )
            )

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                ["Date", "Open", "High", "Low", "Close", "Volume", "OpenInterest"]
            )
            for b in normalized:
                w.writerow(
                    [b.date, b.open, b.high, b.low, b.close, b.volume, b.open_interest]
                )

    # ============================================================
    # Existing CSV reading
    # ============================================================

    def _read_existing_csv(
        self, path: str
    ) -> Tuple[List[StooqBar], Optional[datetime], Optional[str]]:
        """
        Returns:
          (bars_sorted, last_dt, detected_date_format)

        detected_date_format is either:
          - "%d %b %Y" if file looks like "23 Apr 1993"
          - "%Y-%m-%d" if file looks like "1993-04-23"
          - None if unknown
        """
        if not path or not os.path.exists(path):
            return [], None, None

        bars: List[StooqBar] = []
        detected_fmt: Optional[str] = None

        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                ds = (r.get("Date") or "").strip()
                if not ds:
                    continue

                # detect date format from first good date
                if detected_fmt is None:
                    if self._try_parse_date(ds, "%d %b %Y"):
                        detected_fmt = "%d %b %Y"
                    elif self._try_parse_date(ds, "%Y-%m-%d"):
                        detected_fmt = "%Y-%m-%d"

                bars.append(
                    StooqBar(
                        date=ds,
                        open=self._safe_float(r.get("Open")),
                        high=self._safe_float(r.get("High")),
                        low=self._safe_float(r.get("Low")),
                        close=self._safe_float(r.get("Close")),
                        volume=self._safe_float(r.get("Volume")),
                        # tolerate either "OpenInterest" (your sample) or "Open Interest" (stooq csv sometimes)
                        open_interest=self._safe_float(
                            r.get("OpenInterest") or r.get("Open Interest")
                        ),
                    )
                )

        # sort + find last valid dt
        bars_sorted = sorted(bars, key=self._bar_dt_sort_key)
        last_dt = None
        for b in reversed(bars_sorted):
            dt = self._parse_any_date(b.date)
            if dt:
                last_dt = dt
                break

        return bars_sorted, last_dt, detected_fmt

    # ============================================================
    # HTML crawl path (incremental stop)
    # ============================================================

    def _crawl_html(
        self, symbol: str, interval: str, stop_at_dt: Optional[datetime]
    ) -> List[StooqBar]:
        """
        Crawl HTML pages and return:
          - full history if stop_at_dt is None
          - only bars strictly NEWER than stop_at_dt if provided

        We stop early when we see dates <= stop_at_dt because stooq pages are typically newest-first.
        """
        bars: List[StooqBar] = []
        seen = set()

        for page in range(1, self.max_pages + 1):
            if page > 1:
                delay = self.page_delay + (
                    random.uniform(0, self.page_jitter) if self.page_jitter > 0 else 0.0
                )
                time.sleep(delay)

            doc = self._fetch_html_page(symbol, interval, page)
            rows = self._extract_table_rows(doc)
            if not rows:
                break

            added_this_page = 0
            should_stop = False

            for cells in rows:
                # Your existing code uses cells[1] for date. Keep that but guard.
                if len(cells) < 6:
                    continue

                date_raw = (cells[1] or "").strip()
                if not self._is_valid_html_date(date_raw):
                    continue

                dt = self._try_parse_date(date_raw, "%d %b %Y")
                if dt is None:
                    continue

                # incremental stop logic
                if stop_at_dt is not None and dt <= stop_at_dt:
                    should_stop = True
                    continue  # don't add old/equal bars

                if date_raw in seen:
                    continue
                seen.add(date_raw)
                added_this_page += 1

                bars.append(
                    StooqBar(
                        date=date_raw,
                        open=self._safe_float(cells[2]),
                        high=self._safe_float(cells[3]),
                        low=self._safe_float(cells[4]),
                        close=self._safe_float(cells[5]),
                        volume=self._safe_float(cells[7]) if len(cells) > 7 else None,
                        open_interest=(
                            self._safe_float(cells[8]) if len(cells) > 8 else None
                        ),
                    )
                )

            # If we didn’t add anything and we are in incremental mode, we’re done.
            if stop_at_dt is not None:
                if should_stop:
                    break
                if added_this_page == 0:
                    break

            # In full mode, if the page added nothing new, stop.
            if stop_at_dt is None and added_this_page == 0:
                break

        # sort ascending by date
        bars.sort(key=self._bar_dt_sort_key)
        return bars

    def _fetch_html_page(self, symbol: str, interval: str, page: int):
        url = f"{self.BASE_URL}{self.HTML_PATH}"
        params = {"s": symbol, "i": interval, "l": str(page)}
        r = self.session.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return html.fromstring(r.content)

    def _extract_table_rows(self, doc) -> List[List[str]]:
        tables = doc.xpath("//table[@class='fth1']")
        target = None

        for t in tables:
            hdr = " ".join(
                " ".join(th.itertext()) for th in t.xpath(".//thead")
            ).lower()
            if "date" in hdr and "open" in hdr and "close" in hdr:
                target = t
                break

        if target is None:
            return []

        rows: List[List[str]] = []
        for tr in target.xpath(".//tr"):
            cells = [
                " ".join("".join(td.itertext()).split()).strip()
                for td in tr.xpath("./td")
            ]
            if len(cells) < 6:
                continue
            rows.append(cells)

        return rows

    # ============================================================
    # Utilities
    # ============================================================

    @staticmethod
    def _safe_float(v: Optional[str]) -> Optional[float]:
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        s = s.replace(",", "").strip()
        try:
            return float(s)
        except ValueError:
            return None

    @staticmethod
    def _try_parse_date(s: str, fmt: str) -> Optional[datetime]:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            return None

    @classmethod
    def _parse_any_date(cls, s: str) -> Optional[datetime]:
        s = (s or "").strip()
        if not s:
            return None
        # try both common stooq formats
        return cls._try_parse_date(s, "%d %b %Y") or cls._try_parse_date(s, "%Y-%m-%d")

    @classmethod
    def _bar_dt_sort_key(cls, b: StooqBar):
        dt = cls._parse_any_date(b.date)
        return dt if dt is not None else datetime.max

    @staticmethod
    def _is_valid_html_date(s: Optional[str]) -> bool:
        """
        Expected stooq HTML table format: "23 Jan 2026"
        """
        if not s:
            return False
        s = " ".join(str(s).split()).strip()
        if not s:
            return False

        low = s.lower()
        if low in {"date", "data", "n/a", "-", "—"}:
            return False
        if "http" in low or "/" in s:
            return False

        try:
            dt = datetime.strptime(s, "%d %b %Y")
        except Exception:
            return False

        if dt.year < 1800:
            return False
        if dt.date() > datetime.utcnow().date():
            return False

        return True

    @staticmethod
    def _merge_bars(
        existing: List[StooqBar], new: List[StooqBar], *, out_date_format: str
    ) -> List[StooqBar]:
        """
        Merge + dedupe by datetime (not string), then return ascending by date.
        """
        by_dt = {}

        for b in existing + new:
            dt = StooqDownloader._parse_any_date(b.date)
            if dt is None:
                continue
            by_dt[dt] = b  # new overwrites existing for same dt

        merged = []
        for dt in sorted(by_dt.keys()):
            b = by_dt[dt]
            merged.append(
                StooqBar(
                    date=dt.strftime(out_date_format) if out_date_format else b.date,
                    open=b.open,
                    high=b.high,
                    low=b.low,
                    close=b.close,
                    volume=b.volume,
                    open_interest=b.open_interest,
                )
            )
        return merged
