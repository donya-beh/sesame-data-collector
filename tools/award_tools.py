"""
award_tools.py

Tools for searching Teacher/Coach of the Year recognition for offenders via DuckDuckGo.
Provides Strands-compatible @tool-decorated functions for the Teacher_Coach_Agent.
"""

from strands import tool
from ddgs import DDGS


@tool
def search_teacher_coach_award(
    first_name: str,
    middle_name: str,
    last_name: str,
    school_district: str = "",
    state: str = "",
) -> str:
    """Search DuckDuckGo for Teacher/Coach of the Year recognition.

    Builds targeted queries using the offender's name plus school district
    and/or state to reduce false positives from people with the same name.

    Search strategy:
      - Primary queries include school_district and/or state when available,
        e.g. "{name} {district} teacher of the year"
      - Fallback queries use name only if the targeted searches return nothing

    Returns "yes" if recognition is found in any result, "no" otherwise.
    Returns "no" on any exception (network error, rate limit, etc.).
    """
    try:
        # Build the name string
        parts = [p for p in [first_name, middle_name, last_name] if p and p.strip()]
        name = " ".join(parts)

        if not name or name.lower() in ("unknown", ""):
            return "no"

        # Build location context string for targeted queries
        location_parts = []
        if school_district and school_district.strip().lower() not in ("", "unknown"):
            location_parts.append(school_district.strip())
        if state and state.strip().lower() not in ("", "unknown"):
            location_parts.append(state.strip())
        location = " ".join(location_parts)

        # Recognition phrases to look for (case-insensitive)
        recognition_phrases = [
            "teacher of the year",
            "coach of the year",
            "educator of the year",
        ]

        # Build query list — targeted (with location) first, then name-only fallback
        queries = []
        if location:
            queries.append(f'"{name}" {location} teacher of the year')
            queries.append(f'"{name}" {location} coach of the year')
        # Always include name-only queries as fallback
        queries.append(f'"{name}" teacher of the year')
        queries.append(f'"{name}" coach of the year')

        with DDGS() as ddgs:
            for query in queries:
                results = list(ddgs.text(query, max_results=5))
                for result in results:
                    title = (result.get("title") or "").lower()
                    body = (result.get("body") or result.get("snippet") or "").lower()
                    combined = title + " " + body

                    # Check that the offender's last name appears in the result
                    # to reduce false positives from other people
                    if last_name and last_name.lower() not in ("unknown", ""):
                        if last_name.lower() not in combined:
                            continue

                    for phrase in recognition_phrases:
                        if phrase in combined:
                            return "yes"

        return "no"

    except Exception:
        return "no"
