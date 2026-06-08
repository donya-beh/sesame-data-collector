#!/usr/bin/env python3
"""SESAME Data Collector — sequential workflow.

Pipeline:
  Step 1 — Article Fetch        : reads URLs from CSV, downloads article text + date
                                  (Python tools: read_urls_from_csv, fetch_article_with_metadata)
  Step 2 — Data Extraction      : Claude Sonnet 4.5 reads each article and extracts
                                  structured fields; Python validates via extract_misconduct_fields
                                  (Strands Agent + Python tool)
  Step 3 — NCES Location Lookup : Claude extracts school/district names; Python fuzzy-matches
                                  against local NCES CSV files via lookup_nces_location
                                  (Strands Agent + Python tool)
  Step 4 — Award Search         : DuckDuckGo searches for Teacher/Coach of the Year recognition
                                  (Python tool: search_teacher_coach_award)

Run:
  python run_workflow.py                          # uses data/urls.csv
  python run_workflow.py --input my_urls.csv
  python run_workflow.py --input urls.csv --output-dir results/
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

from strands import Agent
from strands.models.bedrock import BedrockModel

from tools import (
    read_urls_from_csv,
    fetch_article_with_metadata,
    extract_misconduct_fields,
    save_results,
    lookup_nces_location,
    search_teacher_coach_award,
)

# ── Pretty printing helpers ─────────────────────────────────────────────────

GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RED = "\033[91m"
DIM = "\033[2m"
BOLD = "\033[1m"
ITALIC = "\033[3m"
RESET = "\033[0m"

INDENT = "    "

TOOL_LOG = {}


class WorkflowCallbackHandler:
    """Streams agent reasoning, tool calls, and responses to the terminal."""

    def __init__(self, step_name=""):
        self.tool_count = 0
        self.tool_names = []
        self._in_reasoning = False
        self._in_text = False
        self.step_name = step_name

    def __call__(self, **kwargs):
        reasoning_text = kwargs.get("reasoningText", "")
        data = kwargs.get("data", "")
        complete = kwargs.get("complete", False)
        event = kwargs.get("event", {})

        tool_use = event.get("contentBlockStart", {}).get("start", {}).get("toolUse")

        if reasoning_text:
            if not self._in_reasoning:
                print(f"\n{INDENT}{DIM}{ITALIC}Thinking: ", end="", flush=True)
                self._in_reasoning = True
            print(f"{DIM}{ITALIC}{reasoning_text}{RESET}", end="", flush=True)

        if tool_use:
            if self._in_reasoning:
                print(RESET)
                self._in_reasoning = False
            if self._in_text:
                print()
                self._in_text = False
            name = tool_use["name"]
            self.tool_count += 1
            self.tool_names.append(name)
            if self.step_name:
                TOOL_LOG.setdefault(self.step_name, []).append(name)
            print(f"{INDENT}{MAGENTA}  ↳ Tool #{self.tool_count}: {name}{RESET}", flush=True)

        if data:
            if self._in_reasoning:
                print(RESET)
                self._in_reasoning = False
            if not self._in_text:
                print(f"{INDENT}{DIM}", end="", flush=True)
                self._in_text = True
            print(data, end="", flush=True)

        if complete:
            if self._in_reasoning:
                print(RESET)
                self._in_reasoning = False
            if self._in_text:
                print(RESET)
                self._in_text = False


def header(input_csv: str):
    print(f"\n{GREEN}{'━' * 70}{RESET}")
    print(f"  {BOLD}SESAME Data Collector{RESET}")
    print(f"{GREEN}{'━' * 70}{RESET}")
    print(f"{DIM}  Input CSV  : {input_csv}{RESET}")
    print()
    print(f"  {DIM}Pipeline:{RESET}")
    print(f"  {DIM}  Step 1: Article Fetch  (Python tools){RESET}")
    print(f"  {DIM}  Step 2: Data Extraction  (Strands Agent → Claude Sonnet 4.5 + Python validation){RESET}")
    print(f"  {DIM}  Step 3: NCES Location Lookup  (Strands Agent → Claude + Python fuzzy matching){RESET}")
    print(f"  {DIM}  Step 4: Award Search  (Python tool → DuckDuckGo){RESET}")
    print()


def step_start(num, total, name, description):
    print(f"\n  {YELLOW}▶ Step {num}/{total}: {name}{RESET}")
    print(f"  {DIM}  {description}{RESET}")


def step_done(num, name, elapsed):
    print(f"\n  {GREEN}✓ Step {num}: {name} done{RESET} ({elapsed:.1f}s)")
    print()


# ── Model factory ───────────────────────────────────────────────────────────

def create_model():
    return BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        max_tokens=8192,
    )


# ── Default Misconduct_Record structure ─────────────────────────────────────

def _default_record() -> dict:
    """Return a Misconduct_Record with all fields set to their defaults."""
    return {
        "article_date": "unknown",
        "notes": "",
        "arrest_date": "unknown",
        "convicted": "unknown",
        "school_name": "unknown",
        "offender_last_name": "unknown",
        "offender_middle_name": "",
        "offender_first_name": "unknown",
        "offender_age": -1,
        "offender_gender": "unknown",
        "offender_roles": "unknown",
        "grades_taught": "unknown",
        "victim_amount": "unknown",
        "victim_gender": "unknown",
        "victim_age": -1,
        "victim_age_range": "unknown",
        "school_district": "unknown",
        "city": "unknown",
        "state": "unknown",
        "zip": "unknown",
        "teacher_coach_of_year": "no",
    }


# ── Agent definitions (Strands Agents — used in Steps 2 and 3 only) ────────
# Steps 1 and 4 use Python tools directly; no LLM needed.

AGENTS = [
    {
        "name": "Data_Extraction_Agent",
        "description": "Claude reads each article and extracts structured misconduct fields",
        "system_prompt": (
            "You are a structured data extraction specialist. "
            "You read news articles about school employee misconduct and extract "
            "specific fields. Always respond with ONLY a valid JSON object, no prose."
        ),
    },
    {
        "name": "NCES_Location_Agent",
        "description": "Claude reads each article and extracts school/district names for NCES lookup",
        "system_prompt": (
            "You are a location specialist. Extract school and district names from "
            "news articles. Always respond with ONLY a valid JSON array, no prose."
        ),
    },
]


# ── JSON parsing helpers ────────────────────────────────────────────────────

def _parse_json_array(text: str) -> list:
    """Try to parse a JSON array from agent output text.

    Attempts in order:
    1. Direct JSON parse of the full string.
    2. Find the first '[' and extract the matching balanced array using
       bracket counting — handles large nested JSON without regex backtracking.

    Returns an empty list if all attempts fail.
    """
    text = text.strip()

    # Attempt 1: direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Attempt 2: find the first '[' and extract the balanced array
    start = text.find("[")
    if start == -1:
        return []

    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    result = json.loads(text[start:i + 1])
                    if isinstance(result, list):
                        return result
                except (json.JSONDecodeError, TypeError):
                    pass
                break

    return []


def _parse_json_object(text: str, key_hint: str = "") -> dict:
    """Try to parse a JSON object from agent output text.

    Attempts in order:
    1. Direct JSON parse of the full string.
    2. Find the first '{' and extract the matching balanced object using
       bracket counting — handles large nested JSON without regex backtracking.

    Returns an empty dict if all attempts fail.
    """
    text = text.strip()

    # Attempt 1: direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Attempt 2: find the first '{' and extract the balanced object
    start = text.find("{")
    if start == -1:
        return {}

    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    result = json.loads(text[start:i + 1])
                    if isinstance(result, dict):
                        # If a key_hint was given, verify it's present
                        if not key_hint or key_hint in result:
                            return result
                except (json.JSONDecodeError, TypeError):
                    pass
                break

    return {}


# ── Workflow execution ──────────────────────────────────────────────────────

def run_workflow(input_csv: str, output_dir: str):
    """Orchestrate the four-agent SESAME data collector pipeline."""
    global TOOL_LOG
    TOOL_LOG = {}  # Reset tool log at the start of each run

    header(input_csv)

    model = create_model()
    step_times = []
    agent_results = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Step 1: Article Fetch (Python tools — no LLM) ──────────────────────
    step_start(1, 4, "Article Fetch", "Read URLs from CSV and download article text + publication date")

    t0 = time.time()
    step_name_1 = "Article Fetch"

    # Read URLs directly (no agent needed for CSV reading)
    from tools.processor_tools import read_urls_from_csv as _read_csv
    urls_json = _read_csv(input_csv)
    if urls_json.startswith("ERROR:"):
        print(f"{RED}  ✗ Could not read CSV: {urls_json}{RESET}", file=sys.stderr)
        sys.exit(1)

    all_urls = json.loads(urls_json)
    print(f"  {GREEN}✓ Read {len(all_urls)} URL(s) from {input_csv}{RESET}")

    processed_records = []
    articles_fetch_skipped = 0

    from tools.processor_tools import fetch_article_with_metadata as _fetch

    for url in all_urls:
        print(f"  {DIM}  Fetching: {url[:80]}{RESET}")
        TOOL_LOG.setdefault(step_name_1, []).append("fetch_article_with_metadata")
        result_json = _fetch(url)
        if result_json.startswith("ERROR:"):
            print(f"{YELLOW}  ✗ Skipping {url}: {result_json}{RESET}", file=sys.stderr)
            articles_fetch_skipped += 1
            continue
        fetched = json.loads(result_json)
        record = _default_record()
        record["article_date"] = fetched.get("article_date", "unknown")
        record["notes"] = url + "\n\n" + fetched.get("text", "")
        processed_records.append(record)

    dt1 = time.time() - t0
    step_times.append(dt1)
    agent_results.append(f"Fetched {urls_read} articles, skipped {articles_fetch_skipped}")
    step_done(1, "Article Fetch", dt1)

    urls_read = len(processed_records)

    if urls_read == 0:
        print(f"{RED}  ✗ No articles fetched. Aborting.{RESET}", file=sys.stderr)
        sys.exit(1)

    print(f"  {GREEN}✓ {urls_read} record(s) ready for extraction.{RESET}")

    # ── Step 2: Data Extraction (Strands Agent → Claude + Python validation) ─
    step_start(2, 4, "Data Extraction", "Claude Sonnet 4.5 reads each article and extracts structured fields")
    step_name_2 = "Data Extraction"

    t0 = time.time()
    extracted_records = []
    articles_skipped = 0

    from tools.extractor_tools import extract_misconduct_fields as _extract

    for i, record in enumerate(processed_records):
        notes = record.get("notes", "")
        article_text = notes.split("\n\n", 1)[1] if "\n\n" in notes else notes
        article_date = record.get("article_date", "unknown")

        single_prompt = (
            f"Read this article carefully and extract the following fields.\n\n"
            f"Article date: {article_date}\n\n"
            f"Article text:\n{article_text}\n\n"
            f"Extract these fields and respond with ONLY a JSON object (no prose):\n"
            f"- arrest_date: M/DD/YYYY or 'unknown'\n"
            f"- convicted: STRICT RULES — use 'yes' ONLY if the article explicitly states the\n"
            f"  offender received a criminal sentence (e.g. 'sentenced to X years', 'sentenced to\n"
            f"  prison', 'received a sentence of'). Use 'no' ONLY if the article explicitly states\n"
            f"  charges were dropped, dismissed, or the offender died before trial. Use 'unknown'\n"
            f"  for ALL other cases including: arrested, charged, facing charges, under investigation,\n"
            f"  pleaded guilty (unless sentence is also stated), found guilty (unless sentence stated),\n"
            f"  or any case where a sentence is not explicitly mentioned. When in doubt use 'unknown'.\n"
            f"- school_name: district name preferred (e.g. 'Birdville ISD'), else school building name\n"
            f"- state_hint: 2-letter state abbreviation (e.g. 'TX') or 'unknown'\n"
            f"- offender_last_name, offender_middle_name ('' if unknown), offender_first_name\n"
            f"- offender_age: integer or -1\n"
            f"- offender_gender: 'F'/'M'/'unknown'\n"
            f"- offender_roles: comma-separated from {{Teacher, Coach, Substitute, Other staff, Administration}};\n"
            f"  NOTE: 'paraprofessional', 'paraeducator', and 'teacher's assistant' should all be classified as 'Other staff'\n"
            f"- grades_taught: comma-separated from {{HS, MS, ES, prek, Kinder}} or 'unknown'\n"
            f"- victim_amount: 'single'/'multiple'/'unknown'\n"
            f"- victim_gender: 'F'/'M'/'both'/'unknown'\n"
            f"- victim_age: integer (youngest) or -1\n\n"
            f"Respond with ONLY the JSON object, nothing else."
        )

        single_handler = WorkflowCallbackHandler(step_name=step_name_2)
        single_agent = Agent(
            model=model,
            tools=[],  # no tools — agent reads and responds with JSON
            system_prompt=AGENTS[0]["system_prompt"],
            callback_handler=single_handler,
        )

        try:
            agent_response = single_agent(single_prompt)
            response_str = str(agent_response)

            # Parse the JSON the agent returned
            extracted = _parse_json_object(response_str)
            if not extracted:
                print(f"{RED}  ✗ Could not parse JSON from agent response for article {i+1}{RESET}", file=sys.stderr)
                print(f"{DIM}  Preview: {repr(response_str[:300])}{RESET}", file=sys.stderr)
                articles_skipped += 1
                continue

            # Merge extracted fields into the record and validate via tool
            merged = dict(record)
            merged.update(extracted)
            validated_json = _extract(json.dumps(merged))
            validated = json.loads(validated_json)
            extracted_records.append(validated)
            TOOL_LOG.setdefault(step_name_2, []).append("extract_misconduct_fields")
            print(f"  {GREEN}  ✓ Extracted record {len(extracted_records)}: {validated.get('offender_first_name','')} {validated.get('offender_last_name','')}{RESET}")

        except Exception as e:
            print(f"{RED}  ✗ Exception in Data_Extraction_Agent for article {i+1}: {e}{RESET}", file=sys.stderr)
            articles_skipped += 1

    dt2 = time.time() - t0
    step_times.append(dt2)
    agent_results.append(f"Extracted {len(extracted_records)} records from {len(processed_records)} articles")
    step_done(2, "Data Extraction", dt2)

    # ── Step 3: NCES Location Lookup (Strands Agent → Claude + Python fuzzy match) ─
    step_start(3, 4, "NCES Location Lookup", "Claude extracts school names; Python fuzzy-matches against NCES CSV files")
    step_name_3 = "NCES Location Lookup"

    t0 = time.time()
    location_enriched_records = []

    from tools.nces_tools import lookup_nces_location as _lookup_nces

    for i, record in enumerate(extracted_records):
        notes = record.get("notes", "")
        article_text = notes.split("\n\n", 1)[1] if "\n\n" in notes else notes
        state_hint = record.get("state_hint", "")

        name_prompt = (
            f"Read this article and list every school and district name mentioned.\n\n"
            f"Article text:\n{article_text}\n\n"
            f"Respond with ONLY a JSON array of names, district names first. Example:\n"
            f'["Birdville ISD", "Haltom High School"]\n\n'
            f"If no school names are found, respond with: []"
        )

        single_handler = WorkflowCallbackHandler(step_name=step_name_3)
        single_agent = Agent(
            model=model,
            tools=[],
            system_prompt=AGENTS[1]["system_prompt"],
            callback_handler=single_handler,
        )

        try:
            name_response = single_agent(name_prompt)
            names_str = str(name_response).strip()

            # Parse the names array
            names_list = _parse_json_array(names_str)
            if not names_list:
                # Fall back to school_name from record
                names_list = [record.get("school_name", "unknown")]

            school_names_json = json.dumps(names_list)
            loc_result = _lookup_nces(school_names=school_names_json, state_hint=state_hint)
            loc = json.loads(loc_result)

            record["school_district"] = loc.get("school_district", "unknown")
            record["city"] = loc.get("city", "unknown")
            record["state"] = loc.get("state", "unknown")
            record["zip"] = loc.get("zip", "unknown")
            TOOL_LOG.setdefault(step_name_3, []).append("lookup_nces_location")
            print(f"  {GREEN}  ✓ Location for record {i+1}: {record['school_district']}, {record['city']}, {record['state']}{RESET}")

        except Exception as e:
            school_name = record.get("school_name", "unknown")
            print(f"{RED}  ✗ Error in NCES lookup for {school_name!r}: {e}{RESET}", file=sys.stderr)

        location_enriched_records.append(record)

    dt3 = time.time() - t0
    step_times.append(dt3)
    agent_results.append(f"Enriched {len(location_enriched_records)} records with NCES location data")
    step_done(3, "NCES Location Lookup", dt3)

    # ── Step 4: Award Search (Python tool — no LLM) ────────────────────────
    step_start(4, 4, "Award Search", "DuckDuckGo search for Teacher/Coach of the Year recognition")
    step_name_4 = "Award Search"

    t0 = time.time()
    enriched_records = []

    from tools.award_tools import search_teacher_coach_award as _search_award

    for i, record in enumerate(location_enriched_records):
        first_name = record.get("offender_first_name", "unknown")
        middle_name = record.get("offender_middle_name", "")
        last_name = record.get("offender_last_name", "unknown")

        try:
            award = _search_award(
                first_name=first_name,
                middle_name=middle_name,
                last_name=last_name,
                school_district=record.get("school_district", ""),
                state=record.get("state", ""),
            )
            record["teacher_coach_of_year"] = award
            TOOL_LOG.setdefault(step_name_4, []).append("search_teacher_coach_award")
            print(f"  {GREEN}  ✓ Award search for {first_name} {last_name}: {award}{RESET}")
        except Exception as e:
            print(f"{RED}  ✗ Award search error for {first_name} {last_name}: {e}{RESET}", file=sys.stderr)
            record["teacher_coach_of_year"] = "no"

        enriched_records.append(record)

    dt4 = time.time() - t0
    step_times.append(dt4)
    agent_results.append(f"Completed award search for {len(enriched_records)} records")
    step_done(4, "Award Search", dt4)

    # ── Save results directly (not via agent) ──────────────────────────────
    output_file = save_results(
        records_json=json.dumps(enriched_records),
        output_dir=output_dir,
        timestamp=timestamp,
    )

    # output_file is now always data/results.csv
    if not os.path.isabs(output_file):
        output_file = os.path.join(output_dir, "results.csv")

    records_extracted = len(enriched_records)
    articles_processed = len(extracted_records)
    total_time = sum(step_times)

    # ── Execution summary ───────────────────────────────────────────────────
    step_names = ["Article Fetch", "Data Extraction", "NCES Location Lookup", "Award Search"]
    print(f"{GREEN}{'━' * 70}{RESET}")
    print(f"  {BOLD}Execution Summary{RESET}")
    print(f"{GREEN}{'━' * 70}{RESET}")
    print(f"  Steps completed: {GREEN}4/4{RESET}")
    print(f"  Total time     : {total_time:.1f}s")

    for i, (name, dt) in enumerate(zip(step_names, step_times)):
        bar_len = int(dt / total_time * 30) if total_time > 0 else 0
        bar = "█" * bar_len + "░" * (30 - bar_len)
        tools_used = len(TOOL_LOG.get(name, []))
        print(
            f"    {YELLOW}{name:35}{RESET} "
            f"{bar} {dt:.1f}s  "
            f"{MAGENTA}({tools_used} tool calls){RESET}"
        )

    if TOOL_LOG:
        total_tools = sum(len(v) for v in TOOL_LOG.values())
        print(f"\n  {BOLD}Tools invoked ({total_tools} total):{RESET}")
        for name in step_names:
            tools = TOOL_LOG.get(name, [])
            if tools:
                tool_str = ", ".join(tools)
                print(f"    {YELLOW}{name:35}{RESET}: {MAGENTA}{tool_str}{RESET}")

    print(f"\n  {BOLD}Run Statistics:{RESET}")
    print(f"    URLs read          : {CYAN}{urls_read}{RESET}")
    print(f"    Articles processed : {GREEN}{articles_processed}{RESET}")
    print(f"    Articles skipped   : {RED}{articles_skipped}{RESET}")
    print(f"    Records extracted  : {GREEN}{records_extracted}{RESET}")
    print(f"    Output file        : {GREEN}{output_file}{RESET}")

    trace_path = save_trace_log(
        input_csv=input_csv,
        output_file=output_file,
        agent_results=agent_results,
        step_times=step_times,
        total_time=total_time,
        urls_read=urls_read,
        articles_processed=articles_processed,
        articles_skipped=articles_skipped,
        records_extracted=records_extracted,
    )
    print(f"\n  {BOLD}Trace log:{RESET}")
    print(f"    {GREEN}{trace_path}{RESET}")

    print(f"\n{GREEN}{'━' * 70}{RESET}\n")

    # Exit with non-zero status if all articles failed
    if urls_read > 0 and records_extracted == 0:
        print(
            f"{RED}Error: All articles failed to process. "
            f"{articles_skipped} article(s) were skipped and no records were extracted.{RESET}",
            file=sys.stderr,
        )
        sys.exit(1)


# ── Trace logging ──────────────────────────────────────────────────────────

def save_trace_log(
    input_csv: str,
    output_file: str,
    agent_results: list,
    step_times: list,
    total_time: float,
    urls_read: int,
    articles_processed: int,
    articles_skipped: int,
    records_extracted: int,
) -> str:
    """Serialize the full workflow execution trace to a JSON file in ./traces/."""
    traces_dir = os.path.join(os.path.dirname(__file__), "traces")
    os.makedirs(traces_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.json"

    steps = []
    step_names = ["Article Fetch", "Data Extraction", "NCES Location Lookup", "Award Search"]
    step_descriptions = [
        "Read URLs from CSV and download article text + publication date",
        "Claude Sonnet 4.5 reads each article and extracts structured fields",
        "Claude extracts school names; Python fuzzy-matches against NCES CSV files",
        "DuckDuckGo search for Teacher/Coach of the Year recognition",
    ]
    for i, (name, desc, agent_result, dt) in enumerate(zip(step_names, step_descriptions, agent_results, step_times)):
        step_entry = {
            "step": i + 1,
            "name": name,
            "description": desc,
            "execution_time_seconds": round(dt, 2),
            "tools_called": TOOL_LOG.get(name, []),
            "response": str(agent_result)[:500],
        }
        steps.append(step_entry)

    trace = {
        "timestamp": datetime.now().isoformat(),
        "input_csv": input_csv,
        "output_file": output_file,
        "total_time_seconds": round(total_time, 2),
        "step_count": 4,
        "urls_read": urls_read,
        "articles_processed": articles_processed,
        "articles_skipped": articles_skipped,
        "records_extracted": records_extracted,
        "tools_invoked": dict(TOOL_LOG),
        "steps": steps,
    }

    trace_path = os.path.join(traces_dir, filename)
    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2, default=str)

    return trace_path


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SESAME Data Collector — 4-agent sequential workflow"
    )
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(__file__), "data", "urls.csv"),
        help="Path to the input CSV file with a 'url' column (default: data/urls.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Directory for output CSV file (default: data/)",
    )
    args = parser.parse_args()

    load_dotenv()

    run_workflow(
        input_csv=args.input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
