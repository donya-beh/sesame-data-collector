"""
nces_tools.py

Tools for enriching misconduct records with school location data using local
NCES CCD data files, with a DuckDuckGo web search as a last resort.

The NCES_Location_Agent (Claude) reads the article and extracts every school
and district name it can find, then passes them as a JSON array to
lookup_nces_location. The tool iterates through the list and fuzzy-matches
each name against the NCES data files deterministically.

Lookup order for each candidate name:
  1. Fuzzy match against ccd_school_districts.csv (LEA_NAME column)
  2. Fuzzy match against ccd_public_schools.csv (SCH_NAME column),
     then join to ccd_school_districts.csv via LEAID to get the district
     name and location (LCITY, LSTATE, LZIP)
  3. DuckDuckGo web search as a last resort (using the first candidate)

Data files are loaded once at module import and cached in memory.
"""

import json
import os
import re
import functools

import pandas as pd
from rapidfuzz import fuzz, process
from strands import tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FUZZY_THRESHOLD = 75        # token_sort_ratio score for district name matching
_FUZZY_THRESHOLD_SCHOOL = 80 # stricter threshold for school→district join

_UNKNOWN_RESULT = {
    "school_district": "unknown",
    "city": "unknown",
    "state": "unknown",
    "zip": "unknown",
}

# Paths to the local NCES data files (one level above the tools/ directory)
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
_DISTRICTS_CSV = os.path.join(_DATA_DIR, "ccd_school_districts.csv")
_SCHOOLS_CSV = os.path.join(_DATA_DIR, "ccd_public_schools.csv")


# ---------------------------------------------------------------------------
# Data loading — loaded once, cached in memory
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _load_districts() -> pd.DataFrame:
    """Load and cache the CCD school districts dataframe."""
    df = pd.read_csv(
        _DISTRICTS_CSV,
        usecols=["LEAID", "LEA_NAME", "LCITY", "LSTATE", "LZIP"],
        dtype=str,
        low_memory=False,
    )
    df = df.dropna(subset=["LEA_NAME"])
    df["LEA_NAME_LOWER"] = df["LEA_NAME"].str.lower().str.strip()
    df["LEAID"] = df["LEAID"].str.strip()
    df["LSTATE"] = df["LSTATE"].str.strip().str.upper()
    df["LZIP"] = df["LZIP"].str.strip()
    return df.reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def _load_schools() -> pd.DataFrame:
    """Load and cache the CCD public schools dataframe."""
    df = pd.read_csv(
        _SCHOOLS_CSV,
        usecols=["LEAID", "SCH_NAME", "LSTATE"],
        dtype=str,
        low_memory=False,
    )
    df = df.dropna(subset=["SCH_NAME"])
    df["SCH_NAME_LOWER"] = df["SCH_NAME"].str.lower().str.strip()
    df["LEAID"] = df["LEAID"].str.strip()
    df["LSTATE"] = df["LSTATE"].str.strip().str.upper()
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fix_acronyms(text: str) -> str:
    """Restore uppercase for common education acronyms after title-casing.

    .title() lowercases acronyms like ISD → Isd, USD → Usd, etc.
    This function corrects them back to their proper uppercase form.
    """
    # Map of title-cased form → correct form
    _ACRONYMS = {
        "Isd": "ISD",   # Independent School District
        "Usd": "USD",   # Unified School District
        "Cusd": "CUSD",
        "Cisd": "CISD",
        "Uisd": "UISD",
        "Nisd": "NISD",
        "Bisd": "BISD",
        "Gisd": "GISD",
        "Lisd": "LISD",
        "Misd": "MISD",
        "Risd": "RISD",
        "Wisd": "WISD",
        "Aisd": "AISD",
        "Fisd": "FISD",
        "Eisd": "EISD",
        "Hisd": "HISD",
        "Kisd": "KISD",
        "Pisd": "PISD",
        "Sisd": "SISD",
        "Tisd": "TISD",
        "Visd": "VISD",
        "Disd": "DISD",
        "Jisd": "JISD",
        "Oisd": "OISD",
        "Csd": "CSD",   # Central/Community School District
        "Lea": "LEA",
        "Ccd": "CCD",
        "Pss": "PSS",
        "Ii": "II",     # e.g. "District II"
        "Iii": "III",
        "Iv": "IV",
    }
    words = text.split()
    return " ".join(_ACRONYMS.get(w, w) for w in words)


def _district_row_to_result(row: pd.Series) -> dict:
    """Convert a districts dataframe row to the result dict.

    Applies title-case then restores uppercase acronyms (ISD, USD, etc.)
    since NCES stores names in all caps. State abbreviation stays uppercase.
    """
    return {
        "school_district": _fix_acronyms(str(row["LEA_NAME"]).strip().title()) or "unknown",
        "city": _fix_acronyms(str(row["LCITY"]).strip().title()) or "unknown",
        "state": str(row["LSTATE"]).strip().upper() or "unknown",
        "zip": str(row["LZIP"]).strip() or "unknown",
    }


def _fuzzy_match_district(
    query: str,
    state_hint: str = "",
    threshold: int = _FUZZY_THRESHOLD,
) -> dict | None:
    """Fuzzy-match query against district LEA_NAME, optionally filtered by state.

    Returns a result dict on match, or None.
    """
    try:
        districts = _load_districts()

        if state_hint and state_hint.upper() not in ("", "UNKNOWN"):
            subset = districts[districts["LSTATE"] == state_hint.upper()]
            if subset.empty:
                subset = districts  # fall back to full dataset
        else:
            subset = districts

        choices = subset["LEA_NAME_LOWER"].tolist()
        result = process.extractOne(
            query.lower().strip(),
            choices,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if result is None:
            return None

        _matched_name, _score, idx = result
        return _district_row_to_result(subset.iloc[idx])
    except Exception:
        return None


def _fuzzy_match_school_to_district(
    query: str,
    state_hint: str = "",
    threshold: int = _FUZZY_THRESHOLD_SCHOOL,
) -> dict | None:
    """Fuzzy-match query against public school SCH_NAME, then join to districts.

    Returns a result dict (with district name from the districts file) on match,
    or None.
    """
    try:
        schools = _load_schools()
        districts = _load_districts()

        if state_hint and state_hint.upper() not in ("", "UNKNOWN"):
            school_subset = schools[schools["LSTATE"] == state_hint.upper()]
            if school_subset.empty:
                school_subset = schools
        else:
            school_subset = schools

        choices = school_subset["SCH_NAME_LOWER"].tolist()
        result = process.extractOne(
            query.lower().strip(),
            choices,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if result is None:
            return None

        _matched_name, _score, idx = result
        school_row = school_subset.iloc[idx]
        leaid = str(school_row["LEAID"]).strip()

        district_rows = districts[districts["LEAID"] == leaid]
        if district_rows.empty:
            return None

        return _district_row_to_result(district_rows.iloc[0])
    except Exception:
        return None


def _web_search_fallback(school_name: str, state_hint: str = "") -> dict:
    """Last-resort DuckDuckGo search for school district name, city, state, zip."""
    try:
        from ddgs import DDGS

        state_clause = f" {state_hint}" if state_hint and state_hint.upper() not in ("", "UNKNOWN") else ""
        query = f'"{school_name}"{state_clause} school district address city state zip'

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        for result in results:
            snippet = (result.get("body") or result.get("snippet") or "").strip()
            title = (result.get("title") or "").strip()
            combined = title + " " + snippet

            zip_match = re.search(
                r"([A-Za-z\s\.\-]+),\s*([A-Z]{2})\s+(\d{5}(?:-\d{4})?)",
                combined,
            )
            if zip_match:
                return {
                    "school_district": school_name,
                    "city": zip_match.group(1).strip().title(),
                    "state": zip_match.group(2).strip().upper(),
                    "zip": zip_match.group(3).strip(),
                }

        return dict(_UNKNOWN_RESULT)

    except Exception:
        return dict(_UNKNOWN_RESULT)


# ---------------------------------------------------------------------------
# Tool: lookup_nces_location
# ---------------------------------------------------------------------------

@tool
def lookup_nces_location(school_names: str, state_hint: str = "") -> str:
    """Look up school_district, city, state, and zip using a list of candidate names.

    The NCES_Location_Agent (Claude) reads the article and extracts every school
    and district name it finds, then passes them as a JSON array string to this
    tool. The tool iterates through the list and fuzzy-matches each name against
    the local NCES CCD data files deterministically.

    Search order:
      Pass 1 — try every candidate against ccd_school_districts.csv (LEA_NAME),
               filtered by state_hint when provided. Uses token_sort_ratio ≥ 75.
               District names are tried first across all candidates.
      Pass 2 — try every candidate against ccd_public_schools.csv (SCH_NAME),
               filtered by state_hint, then join to ccd_school_districts.csv via
               LEAID to get the district name and location. Uses ratio ≥ 80.
      Pass 3 — DuckDuckGo web search using the first candidate as the query.

    All results return title-cased strings (e.g. "Birdville Isd", "Haltom City").
    State abbreviation is always uppercase (e.g. "TX").

    Args:
        school_names: JSON array string of school/district name candidates
                      extracted from the article by the agent, e.g.:
                      '["Birdville ISD", "Haltom High School"]'
                      Also accepts a plain string (treated as a single candidate).
        state_hint:   Optional 2-letter US state abbreviation (e.g. "TX", "FL")
                      used to pre-filter NCES candidates before fuzzy matching.
                      Pass "" or "unknown" to search all states.

    Returns:
        JSON string {"school_district": str, "city": str, "state": str, "zip": str}
        with "unknown" for any fields that could not be determined.
    """
    # --- Parse the candidate list ---
    candidates = []
    if school_names:
        raw = school_names.strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                candidates = [str(n).strip() for n in parsed if str(n).strip()]
            elif isinstance(parsed, str):
                candidates = [parsed.strip()]
        except (json.JSONDecodeError, TypeError):
            candidates = [raw]

    candidates = [c for c in candidates if c and c.lower() != "unknown"]

    if not candidates:
        return json.dumps(_UNKNOWN_RESULT)

    hint = (state_hint or "").strip()

    # --- Pass 1: district file ---
    for name in candidates:
        result = _fuzzy_match_district(name, hint)
        if result is not None:
            return json.dumps(result)

    # --- Pass 2: schools file → district join ---
    for name in candidates:
        result = _fuzzy_match_school_to_district(name, hint)
        if result is not None:
            return json.dumps(result)

    # --- Pass 3: web search ---
    result = _web_search_fallback(candidates[0], hint)
    return json.dumps(result)
