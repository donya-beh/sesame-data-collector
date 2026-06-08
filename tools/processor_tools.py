"""
processor_tools.py

Tools for reading article URLs from CSV files and fetching article text with
publication metadata. Provides Strands-compatible @tool-decorated functions
for the Article_Processor_Agent in the SESAME data collection pipeline.
"""

import csv
import json
import os
import re

from bs4 import BeautifulSoup
from strands import tool


# ---------------------------------------------------------------------------
# Helper: parse a date string into M/DD/YYYY format
# ---------------------------------------------------------------------------

def _parse_date_string(date_str: str) -> str:
    """Attempt to parse a date string and return it as M/DD/YYYY.

    Tries several common formats found in HTML metadata.  Returns None if
    parsing fails so the caller can fall through to the next source.
    """
    if not date_str:
        return None

    date_str = date_str.strip()

    # Try ISO-8601 / RFC-3339 variants first: 2024-05-03T... or 2024-05-03
    iso_match = re.match(r"(\d{4})-(\d{2})-(\d{2})", date_str)
    if iso_match:
        year, month, day = int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{month}/{day:02d}/{year}"

    # Try MM/DD/YYYY or M/D/YYYY
    slash_match = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", date_str)
    if slash_match:
        month, day, year = int(slash_match.group(1)), int(slash_match.group(2)), int(slash_match.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{month}/{day:02d}/{year}"

    # Try DD-MM-YYYY (European style) — less common in US news metadata
    dmy_match = re.match(r"(\d{2})-(\d{2})-(\d{4})", date_str)
    if dmy_match:
        day, month, year = int(dmy_match.group(1)), int(dmy_match.group(2)), int(dmy_match.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{month}/{day:02d}/{year}"

    # Try named-month formats: "May 3, 2024" or "3 May 2024"
    month_names = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
        "oct": 10, "nov": 11, "dec": 12,
    }

    # "Month DD, YYYY" or "Month D, YYYY"
    mdy_named = re.match(
        r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", date_str
    )
    if mdy_named:
        month_name = mdy_named.group(1).lower()
        if month_name in month_names:
            month = month_names[month_name]
            day = int(mdy_named.group(2))
            year = int(mdy_named.group(3))
            if 1 <= day <= 31:
                return f"{month}/{day:02d}/{year}"

    # "DD Month YYYY"
    dmy_named = re.match(
        r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", date_str
    )
    if dmy_named:
        month_name = dmy_named.group(2).lower()
        if month_name in month_names:
            month = month_names[month_name]
            day = int(dmy_named.group(1))
            year = int(dmy_named.group(3))
            if 1 <= day <= 31:
                return f"{month}/{day:02d}/{year}"

    return None


# ---------------------------------------------------------------------------
# Helper: extract publication date from raw HTML
# ---------------------------------------------------------------------------

def _extract_date_from_html(html: str) -> str:
    """Parse raw HTML and return the publication date as M/DD/YYYY or 'unknown'.

    Checks sources in priority order:
      1. <meta property="article:published_time" content="...">
      2. <meta property="og:article:published_time" content="...">
      3. <meta name="article:published_time" content="...">
      4. <meta name="date" content="...">
      5. <meta name="pubdate" content="...">
      6. <meta name="publishdate" content="...">
      7. JSON-LD script tags — looks for datePublished field
      8. <time datetime="..."> element
    """
    if not html:
        return "unknown"

    soup = BeautifulSoup(html, "lxml")

    # Priority 1-6: meta tags
    meta_selectors = [
        {"property": "article:published_time"},
        {"property": "og:article:published_time"},
        {"name": "article:published_time"},
        {"name": "date"},
        {"name": "pubdate"},
        {"name": "publishdate"},
    ]

    for attrs in meta_selectors:
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            parsed = _parse_date_string(tag["content"])
            if parsed:
                return parsed

    # Priority 7: JSON-LD script tags
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            # data may be a dict or a list of dicts
            if isinstance(data, list):
                candidates = data
            else:
                candidates = [data]
            for item in candidates:
                if isinstance(item, dict):
                    date_val = item.get("datePublished") or item.get("dateCreated")
                    if date_val:
                        parsed = _parse_date_string(str(date_val))
                        if parsed:
                            return parsed
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue

    # Priority 8: <time datetime="...">
    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag and time_tag.get("datetime"):
        parsed = _parse_date_string(time_tag["datetime"])
        if parsed:
            return parsed

    return "unknown"


# ---------------------------------------------------------------------------
# Tool 1: read_urls_from_csv
# ---------------------------------------------------------------------------

@tool
def read_urls_from_csv(csv_path: str) -> str:
    """Read article URLs from a CSV file with a 'url' column.

    Opens the CSV at csv_path, reads the 'url' column, and filters out blank
    rows.  Returns a JSON array of URL strings (may be empty if no valid URLs
    are present).

    Returns:
        JSON array of URL strings on success.
        'ERROR: File not found: <path>' if the file does not exist.
        'ERROR: CSV file has no url column: <path>' if the column is missing.
        'ERROR: Could not read CSV file <path>: <exc>' on other exceptions.
    """
    if not os.path.exists(csv_path):
        return f"ERROR: File not found: {csv_path}"

    try:
        urls = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "url" not in reader.fieldnames:
                return f"ERROR: CSV file has no 'url' column: {csv_path}"
            for row in reader:
                url = row.get("url", "").strip()
                if url:
                    urls.append(url)
        return json.dumps(urls)
    except Exception as exc:
        return f"ERROR: Could not read CSV file {csv_path}: {exc}"


# ---------------------------------------------------------------------------
# Tool 2: fetch_article_with_metadata
# ---------------------------------------------------------------------------

@tool
def fetch_article_with_metadata(url: str) -> str:
    """Fetch article text and publication date from a URL.

    Uses trafilatura to download the raw HTML and extract the main article body
    text.  Parses the raw HTML with BeautifulSoup to extract the publication
    date from meta tags, JSON-LD structured data, or <time> elements.

    Article text is truncated to 8000 characters; a '\\n[truncated]' marker is
    appended when truncation occurs.

    Returns:
        JSON string {"url": str, "text": str, "article_date": str} on success,
        where article_date is M/DD/YYYY or "unknown".
        'ERROR: trafilatura could not fetch URL: <url>' if fetch returns None.
        'ERROR: trafilatura extracted no text from URL: <url>' if extract
        returns None or empty string.
        'ERROR: <url> — <exc>' on any other exception.
    """
    try:
        import trafilatura  # imported here so the module is mockable in tests

        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return f"ERROR: trafilatura could not fetch URL: {url}"

        text = trafilatura.extract(downloaded)
        if not text:
            return f"ERROR: trafilatura extracted no text from URL: {url}"

        # Truncate article text to 8000 characters
        if len(text) > 8000:
            text = text[:8000] + "\n[truncated]"

        # Extract publication date from the raw HTML
        article_date = _extract_date_from_html(downloaded)

        result = {
            "url": url,
            "text": text,
            "article_date": article_date,
        }
        return json.dumps(result)

    except Exception as exc:
        return f"ERROR: {url} — {exc}"
