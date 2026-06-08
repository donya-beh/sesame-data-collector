#!/usr/bin/env python3
"""SESAME Data Collector — Flask web application.

Provides a local web interface for uploading a CSV of article URLs,
running the 4-step extraction pipeline with live progress streaming,
previewing results in the browser, and downloading the output CSV.

Run:
    python app.py
Then open http://localhost:5000 in browser.
"""

import csv
import io
import json
import os
import queue
import threading
import time
from datetime import datetime

from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
    stream_with_context,
)

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

# Directory for temporary uploaded CSVs and output
_BASE_DIR = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_BASE_DIR, "data")
_RESULTS_PATH = os.path.join(_DATA_DIR, "results.csv")

# Global state for the currently running job
_job_queue: queue.Queue = queue.Queue()
_job_running = False
_job_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Pipeline runner (runs in a background thread)
# ---------------------------------------------------------------------------

def _run_pipeline(csv_path: str, progress_q: queue.Queue):
    """Run the full extraction pipeline, emitting progress events to the queue."""

    def emit(msg: str, level: str = "info"):
        progress_q.put({"type": "log", "level": level, "message": msg})

    def emit_progress(current: int, total: int, label: str):
        progress_q.put({"type": "progress", "current": current, "total": total, "label": label})

    try:
        # ── Imports ──────────────────────────────────────────────────────
        from strands import Agent
        from strands.models.bedrock import BedrockModel
        from tools.processor_tools import (
            read_urls_from_csv as _read_csv,
            fetch_article_with_metadata as _fetch,
        )
        from tools.extractor_tools import (
            extract_misconduct_fields as _extract,
            save_results as _save,
            CSV_COLUMNS,
        )
        from tools.nces_tools import lookup_nces_location as _lookup_nces
        from tools.award_tools import search_teacher_coach_award as _search_award

        model = BedrockModel(
            model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            max_tokens=8192,
        )

        def _default_record():
            return {
                "article_date": "unknown", "notes": "",
                "arrest_date": "unknown", "convicted": "unknown",
                "school_name": "unknown",
                "offender_last_name": "unknown", "offender_middle_name": "",
                "offender_first_name": "unknown", "offender_age": -1,
                "offender_gender": "unknown", "offender_roles": "unknown",
                "grades_taught": "unknown", "victim_amount": "unknown",
                "victim_gender": "unknown", "victim_age": -1,
                "victim_age_range": "unknown", "school_district": "unknown",
                "city": "unknown", "state": "unknown", "zip": "unknown",
                "teacher_coach_of_year": "no",
            }

        def _parse_json_object(text: str) -> dict:
            text = text.strip()
            try:
                r = json.loads(text)
                if isinstance(r, dict):
                    return r
            except Exception:
                pass
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
                            r = json.loads(text[start:i + 1])
                            if isinstance(r, dict):
                                return r
                        except Exception:
                            pass
                        break
            return {}

        def _parse_json_array(text: str) -> list:
            text = text.strip()
            try:
                r = json.loads(text)
                if isinstance(r, list):
                    return r
            except Exception:
                pass
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
                            r = json.loads(text[start:i + 1])
                            if isinstance(r, list):
                                return r
                        except Exception:
                            pass
                        break
            return []

        # ── Step 1: Read URLs and fetch articles ─────────────────────────
        emit("📂 Step 1: Reading URLs from CSV...")
        urls_json = _read_csv(csv_path)
        if urls_json.startswith("ERROR:"):
            emit(f"❌ Could not read CSV: {urls_json}", "error")
            progress_q.put({"type": "done", "success": False})
            return

        all_urls = json.loads(urls_json)
        total_urls = len(all_urls)
        emit(f"✓ Found {total_urls} URL(s)")
        emit_progress(0, total_urls, "Fetching articles")

        processed_records = []
        failed_urls = []
        for idx, url in enumerate(all_urls):
            emit(f"  Fetching ({idx + 1}/{total_urls}): {url[:80]}...")
            result_json = _fetch(url)
            if result_json.startswith("ERROR:"):
                emit(f"  ⚠ Skipped: {result_json}", "warn")
                failed_urls.append(url)
            else:
                fetched = json.loads(result_json)
                record = _default_record()
                record["article_date"] = fetched.get("article_date", "unknown")
                record["notes"] = url + "\n\n" + fetched.get("text", "")
                processed_records.append(record)
                emit(f"  ✓ Fetched article {len(processed_records)}")
            emit_progress(idx + 1, total_urls, "Fetching articles")

        urls_read = len(processed_records)
        if urls_read == 0:
            emit("❌ No articles could be fetched. Aborting.", "error")
            progress_q.put({"type": "done", "success": False, "failed_urls": failed_urls})
            return

        if failed_urls:
            emit(f"  ⚠ {len(failed_urls)} URL(s) failed to fetch — you can paste their text manually after the run.", "warn")

        emit(f"✓ Step 1 complete: {urls_read} article(s) fetched")

        # ── Step 2: Data extraction ───────────────────────────────────────
        emit(f"\n🔍 Step 2: Extracting misconduct fields ({urls_read} articles)...")
        emit_progress(0, urls_read, "Extracting fields")

        extracted_records = []
        skipped = 0

        for i, record in enumerate(processed_records):
            notes = record.get("notes", "")
            article_text = notes.split("\n\n", 1)[1] if "\n\n" in notes else notes
            article_date = record.get("article_date", "unknown")

            emit(f"  Extracting article {i + 1}/{urls_read}...")

            prompt = (
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

            try:
                agent = Agent(
                    model=model,
                    tools=[],
                    system_prompt=(
                        "You are a structured data extraction specialist. "
                        "You read news articles about school employee misconduct and extract "
                        "specific fields. Always respond with ONLY a valid JSON object, no prose."
                    ),
                )
                response = agent(prompt)
                extracted = _parse_json_object(str(response))
                if extracted:
                    merged = dict(record)
                    merged.update(extracted)
                    validated_json = _extract(json.dumps(merged))
                    validated = json.loads(validated_json)
                    extracted_records.append(validated)
                    name = f"{validated.get('offender_first_name', '')} {validated.get('offender_last_name', '')}".strip()
                    emit(f"  ✓ Extracted: {name or 'unknown'}")
                else:
                    emit(f"  ⚠ Could not parse extraction result for article {i + 1}", "warn")
                    skipped += 1
            except Exception as e:
                emit(f"  ⚠ Error on article {i + 1}: {e}", "warn")
                skipped += 1

            emit_progress(i + 1, urls_read, "Extracting fields")

        emit(f"✓ Step 2 complete: {len(extracted_records)} extracted, {skipped} skipped")

        # ── Step 3: NCES location lookup ──────────────────────────────────
        emit(f"\n📍 Step 3: Looking up school districts ({len(extracted_records)} records)...")
        emit_progress(0, len(extracted_records), "Looking up locations")

        location_records = []
        for i, record in enumerate(extracted_records):
            notes = record.get("notes", "")
            article_text = notes.split("\n\n", 1)[1] if "\n\n" in notes else notes
            state_hint = record.get("state_hint", "")

            emit(f"  Looking up location for record {i + 1}/{len(extracted_records)}...")

            name_prompt = (
                f"Read this article and list every school and district name mentioned.\n\n"
                f"Article text:\n{article_text}\n\n"
                f"Respond with ONLY a JSON array of names, district names first. Example:\n"
                f'["Birdville ISD", "Haltom High School"]\n\n'
                f"If no school names are found, respond with: []"
            )

            try:
                agent = Agent(
                    model=model,
                    tools=[],
                    system_prompt=(
                        "You are a location specialist. Extract school and district names from "
                        "news articles. Always respond with ONLY a valid JSON array, no prose."
                    ),
                )
                name_response = agent(name_prompt)
                names_list = _parse_json_array(str(name_response).strip())
                if not names_list:
                    names_list = [record.get("school_name", "unknown")]

                loc_result = _lookup_nces(
                    school_names=json.dumps(names_list),
                    state_hint=state_hint,
                )
                loc = json.loads(loc_result)
                record["school_district"] = loc.get("school_district", "unknown")
                record["city"] = loc.get("city", "unknown")
                record["state"] = loc.get("state", "unknown")
                record["zip"] = loc.get("zip", "unknown")
                emit(f"  ✓ {record['school_district']}, {record['city']}, {record['state']}")
            except Exception as e:
                emit(f"  ⚠ Location lookup error: {e}", "warn")

            location_records.append(record)
            emit_progress(i + 1, len(extracted_records), "Looking up locations")

        emit(f"✓ Step 3 complete")

        # ── Step 4: Teacher/Coach of the Year ─────────────────────────────
        emit(f"\n🏆 Step 4: Searching Teacher/Coach of the Year ({len(location_records)} records)...")
        emit_progress(0, len(location_records), "Award search")

        enriched_records = []
        for i, record in enumerate(location_records):
            first = record.get("offender_first_name", "unknown")
            middle = record.get("offender_middle_name", "")
            last = record.get("offender_last_name", "unknown")
            emit(f"  Searching: {first} {last}...")
            try:
                award = _search_award(
                    first_name=first,
                    middle_name=middle,
                    last_name=last,
                    school_district=record.get("school_district", ""),
                    state=record.get("state", ""),
                )
                record["teacher_coach_of_year"] = award
                emit(f"  ✓ {first} {last}: {award}")
            except Exception as e:
                emit(f"  ⚠ Award search error: {e}", "warn")
                record["teacher_coach_of_year"] = "no"
            enriched_records.append(record)
            emit_progress(i + 1, len(location_records), "Award search")

        emit(f"✓ Step 4 complete")

        # ── Save results ──────────────────────────────────────────────────
        emit(f"\n💾 Saving results to {_RESULTS_PATH}...")
        os.makedirs(_DATA_DIR, exist_ok=True)
        with open(_RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for r in enriched_records:
                writer.writerow(r)

        emit(f"✓ Saved {len(enriched_records)} record(s)")
        emit(f"\n🎉 Pipeline complete! {len(enriched_records)} record(s) ready.")

        # Send the results data for the preview table
        preview_rows = []
        for r in enriched_records:
            preview_rows.append({k: r.get(k, "") for k in CSV_COLUMNS if k != "notes"})

        progress_q.put({
            "type": "done",
            "success": True,
            "count": len(enriched_records),
            "failed_urls": failed_urls,
            "columns": [c for c in CSV_COLUMNS if c != "notes"],
            "rows": preview_rows,
        })

    except Exception as e:
        emit(f"❌ Unexpected error: {e}", "error")
        import traceback
        emit(traceback.format_exc(), "error")
        progress_q.put({"type": "done", "success": False})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global _job_running

    with _job_lock:
        if _job_running:
            return jsonify({"error": "A job is already running. Please wait."}), 409

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    f = request.files["file"]
    if not f.filename or not f.filename.endswith(".csv"):
        return jsonify({"error": "Please upload a .csv file."}), 400

    # Save uploaded CSV to data/
    os.makedirs(_DATA_DIR, exist_ok=True)
    upload_path = os.path.join(_DATA_DIR, "uploaded_urls.csv")
    f.save(upload_path)

    # Clear the queue and start the pipeline in a background thread
    while not _job_queue.empty():
        try:
            _job_queue.get_nowait()
        except queue.Empty:
            break

    with _job_lock:
        _job_running = True

    def run_and_mark_done():
        global _job_running
        try:
            _run_pipeline(upload_path, _job_queue)
        finally:
            with _job_lock:
                _job_running = False

    thread = threading.Thread(target=run_and_mark_done, daemon=True)
    thread.start()

    return jsonify({"status": "started"})


@app.route("/stream")
def stream():
    """Server-Sent Events endpoint — streams progress from the job queue."""

    def event_generator():
        while True:
            try:
                item = _job_queue.get(timeout=30)
                yield f"data: {json.dumps(item)}\n\n"
                if item.get("type") == "done":
                    break
            except queue.Empty:
                # Send a keepalive comment
                yield ": keepalive\n\n"

    return Response(
        stream_with_context(event_generator()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/download")
def download():
    if not os.path.exists(_RESULTS_PATH):
        return jsonify({"error": "No results file found. Run the pipeline first."}), 404
    return send_file(
        _RESULTS_PATH,
        mimetype="text/csv",
        as_attachment=True,
        download_name="sesame_results.csv",
    )


@app.route("/status")
def status():
    return jsonify({"running": _job_running})


@app.route("/process_manual", methods=["POST"])
def process_manual():
    """Process a single article from manually pasted text.

    Expects JSON: {"url": str, "article_text": str, "article_date": str (optional)}
    Runs Steps 2–4 on the pasted text and appends the result to results.csv.
    Returns the new row as JSON.
    """
    data = request.get_json()
    if not data or not data.get("article_text", "").strip():
        return jsonify({"error": "No article text provided."}), 400

    url = data.get("url", "unknown")
    article_text = data["article_text"].strip()
    article_date = data.get("article_date", "unknown").strip() or "unknown"

    try:
        from strands import Agent
        from strands.models.bedrock import BedrockModel
        from tools.extractor_tools import extract_misconduct_fields as _extract, CSV_COLUMNS
        from tools.nces_tools import lookup_nces_location as _lookup_nces
        from tools.award_tools import search_teacher_coach_award as _search_award

        model = BedrockModel(
            model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            max_tokens=8192,
        )

        def _parse_json_object(text):
            text = text.strip()
            try:
                r = json.loads(text)
                if isinstance(r, dict):
                    return r
            except Exception:
                pass
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
                            r = json.loads(text[start:i + 1])
                            if isinstance(r, dict):
                                return r
                        except Exception:
                            pass
                        break
            return {}

        def _parse_json_array(text):
            text = text.strip()
            try:
                r = json.loads(text)
                if isinstance(r, list):
                    return r
            except Exception:
                pass
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
                            r = json.loads(text[start:i + 1])
                            if isinstance(r, list):
                                return r
                        except Exception:
                            pass
                        break
            return []

        record = {
            "article_date": article_date, "notes": url + "\n\n" + article_text,
            "arrest_date": "unknown", "convicted": "unknown", "school_name": "unknown",
            "offender_last_name": "unknown", "offender_middle_name": "",
            "offender_first_name": "unknown", "offender_age": -1,
            "offender_gender": "unknown", "offender_roles": "unknown",
            "grades_taught": "unknown", "victim_amount": "unknown",
            "victim_gender": "unknown", "victim_age": -1,
            "victim_age_range": "unknown", "school_district": "unknown",
            "city": "unknown", "state": "unknown", "zip": "unknown",
            "teacher_coach_of_year": "no",
        }

        # Step 2: extract fields
        prompt = (
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
            f"- school_name: district name preferred, else school building name\n"
            f"- state_hint: 2-letter state abbreviation or 'unknown'\n"
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
        agent = Agent(
            model=model, tools=[],
            system_prompt=(
                "You are a structured data extraction specialist. "
                "Always respond with ONLY a valid JSON object, no prose."
            ),
        )
        response = agent(prompt)
        extracted = _parse_json_object(str(response))
        if extracted:
            record.update(extracted)
        validated_json = _extract(json.dumps(record))
        record = json.loads(validated_json)

        # Step 3: NCES location
        state_hint = record.get("state_hint", "")
        name_prompt = (
            f"Read this article and list every school and district name mentioned.\n\n"
            f"Article text:\n{article_text}\n\n"
            f"Respond with ONLY a JSON array of names, district names first. "
            f"If none found, respond with: []"
        )
        agent2 = Agent(
            model=model, tools=[],
            system_prompt="Extract school names from articles. Respond with ONLY a JSON array.",
        )
        name_response = agent2(name_prompt)
        names_list = _parse_json_array(str(name_response).strip())
        if not names_list:
            names_list = [record.get("school_name", "unknown")]
        loc = json.loads(_lookup_nces(school_names=json.dumps(names_list), state_hint=state_hint))
        record["school_district"] = loc.get("school_district", "unknown")
        record["city"] = loc.get("city", "unknown")
        record["state"] = loc.get("state", "unknown")
        record["zip"] = loc.get("zip", "unknown")

        # Step 4: award search
        award = _search_award(
            first_name=record.get("offender_first_name", "unknown"),
            middle_name=record.get("offender_middle_name", ""),
            last_name=record.get("offender_last_name", "unknown"),
            school_district=record.get("school_district", ""),
            state=record.get("state", ""),
        )
        record["teacher_coach_of_year"] = award

        # Append to results.csv
        os.makedirs(_DATA_DIR, exist_ok=True)
        file_exists = os.path.exists(_RESULTS_PATH)
        with open(_RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

        preview = {k: record.get(k, "") for k in CSV_COLUMNS if k != "notes"}
        return jsonify({"success": True, "row": preview, "columns": [c for c in CSV_COLUMNS if c != "notes"]})

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(debug=False, port=5001, threaded=True)
