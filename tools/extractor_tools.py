"""
extractor_tools.py

Tools for extracting structured misconduct fields from article text and saving results to CSV.
Provides Strands-compatible @tool-decorated functions for the Data_Extraction_Agent.
"""

import csv
import json
import os
import re

from strands import tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _derive_victim_age_range(victim_age: int) -> str:
    """Deterministically derive victim_age_range from victim_age.

    Mapping:
        3–7   → "3-7"
        8–14  → "8-14"
        15–18 → "15-18"
        ≥19   → "19+"
        0, 1, 2 or any negative → "unknown"
    """
    if victim_age < 0:
        return "unknown"
    if victim_age <= 2:
        return "unknown"
    if victim_age <= 7:
        return "3-7"
    if victim_age <= 14:
        return "8-14"
    if victim_age <= 18:
        return "15-18"
    return "19+"


def _coerce_int(value, sentinel: int = -1) -> int:
    """Try to coerce *value* to int; return *sentinel* on failure."""
    if value is None:
        return sentinel
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return sentinel


def _extract_json_from_text(text: str) -> dict:
    """Try to extract a JSON object from *text*.

    Attempts in order:
    1. Direct JSON parse.
    2. Strip markdown code fences (```json ... ``` or ``` ... ```).
    3. Find the first ``{...}`` block via regex.

    Returns an empty dict if all attempts fail.
    """
    # 1. Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Strip markdown fences
    stripped = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        pass

    # 3. Find first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, TypeError):
            pass

    return {}


# ---------------------------------------------------------------------------
# Canonical column order for the final CSV (20 columns, school_name excluded)
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "article_date",
    "arrest_date",
    "convicted",
    "school_district",
    "city",
    "state",
    "zip",
    "offender_last_name",
    "offender_middle_name",
    "offender_first_name",
    "offender_age",
    "offender_gender",
    "offender_roles",
    "grades_taught",
    "victim_amount",
    "victim_gender",
    "victim_age",
    "victim_age_range",
    "teacher_coach_of_year",
    "notes",
]


# ---------------------------------------------------------------------------
# Tool: extract_misconduct_fields
# ---------------------------------------------------------------------------

@tool
def extract_misconduct_fields(record_json: str) -> str:
    """Validate and normalize the 15 extraction fields in a Misconduct_Record JSON.

    Fills in correct sentinel values for any missing or invalid fields.
    Derives victim_age_range from victim_age deterministically.
    Preserves article_date and notes from the input record.
    Initializes later-agent fields (school_district, city, state, zip,
    teacher_coach_of_year) to their defaults if absent.

    Returns the validated JSON string of the full record.
    """
    # --- Parse input ---
    record = _extract_json_from_text(record_json)

    # --- Preserve fields set by Article_Processor_Agent ---
    article_date = record.get("article_date", "unknown")
    if article_date is None:
        article_date = "unknown"

    notes = record.get("notes", "")
    if notes is None:
        notes = ""

    # --- Validate / fill the 15 extraction fields ---

    # String fields with sentinel "unknown"
    str_unknown_fields = [
        "arrest_date",
        "convicted",
        "school_name",
        "state_hint",
        "offender_last_name",
        "offender_first_name",
        "offender_gender",
        "offender_roles",
        "grades_taught",
        "victim_amount",
        "victim_gender",
    ]
    for field in str_unknown_fields:
        val = record.get(field)
        if val is None or str(val).strip() == "":
            record[field] = "unknown"
        else:
            record[field] = str(val)

    # offender_middle_name — sentinel is "" (empty string, NOT "unknown")
    omn = record.get("offender_middle_name")
    if omn is None:
        record["offender_middle_name"] = ""
    else:
        record["offender_middle_name"] = str(omn)

    # offender_age — int, sentinel -1
    record["offender_age"] = _coerce_int(record.get("offender_age"), sentinel=-1)

    # victim_age — int, sentinel -1
    record["victim_age"] = _coerce_int(record.get("victim_age"), sentinel=-1)

    # victim_age_range — always recomputed from victim_age
    record["victim_age_range"] = _derive_victim_age_range(record["victim_age"])

    # --- Restore preserved fields ---
    record["article_date"] = article_date
    record["notes"] = notes

    # --- Initialize later-agent fields if missing ---
    for field, default in [
        ("school_district", "unknown"),
        ("city", "unknown"),
        ("state", "unknown"),
        ("zip", "unknown"),
        ("teacher_coach_of_year", "no"),
    ]:
        if field not in record or record[field] is None:
            record[field] = default

    return json.dumps(record)


# ---------------------------------------------------------------------------
# Tool: save_results
# ---------------------------------------------------------------------------

@tool
def save_results(records_json: str, output_dir: str, timestamp: str) -> str:
    """Write the JSON array of EnrichedRecords to a fixed-name CSV file.

    Creates output_dir if it does not exist.
    Always writes to 'results.csv', overwriting any previous run.
    Returns the full output file path.
    """
    try:
        records = json.loads(records_json)
    except (json.JSONDecodeError, TypeError):
        records = []

    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, "results.csv")

    with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore",
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    return file_path
