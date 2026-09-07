#!/usr/bin/env python3
"""
Measures the thing that actually matters: does the service pick the right
response type?

Four sets are scored separately, because the failure modes are different and
averaging them hides the interesting one.

    answered      20 questions the corpus answers cleanly
    not_covered   25 adversarial questions it cannot answer
    conflict       7 questions landing on the three planted contradictions
    near_miss      3 questions that look like conflicts and are not

Run:  python evaluate.py            (uses whatever backend is configured)
      USE_DENSE_EMBEDDINGS=0 python evaluate.py   (lexical-only fallback)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.main import app  # noqa: E402

TESTS = Path(__file__).resolve().parent / "tests"

GREEN, RED, YELLOW, DIM, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[0m"


def load(name: str) -> Dict[str, Any]:
    return json.loads((TESTS / name).read_text(encoding="utf-8"))


def run_set(
    client: TestClient,
    title: str,
    questions: List[Dict[str, Any]],
    expected: str,
    section_key: str | None = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    passed = 0
    failures: List[str] = []
    citation_hits = 0
    citation_total = 0

    print(f"\n{title}")
    print("-" * len(title))

    for item in questions:
        response = client.post("/ask", json={"question": item["question"], "top_k": 5}).json()
        got = response["response_type"]
        ok = got == expected
        passed += ok
        if not ok:
            failures.append(f'{item["id"]}: got {got} — {item["question"]}')

        # For answerable questions, also check the right clause was cited.
        if section_key and item.get(section_key):
            citation_total += 1
            cited = {c["section_id"] for c in response["citations"]}
            if item[section_key] in cited:
                citation_hits += 1

        mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        if verbose or not ok:
            print(f"  {mark} {item['id']}  {item['question'][:78]}")
            if not ok:
                print(f"{DIM}        expected {expected}, got {got}{RESET}")
                print(f"{DIM}        reason: {response['reasoning'][:110]}{RESET}")

    total = len(questions)
    pct = 100.0 * passed / total if total else 0.0
    colour = GREEN if pct == 100 else (YELLOW if pct >= 80 else RED)
    print(f"  {colour}{passed}/{total} correct ({pct:.0f}%){RESET}")
    if citation_total:
        cpct = 100.0 * citation_hits / citation_total
        print(f"  {DIM}expected clause present in citations: {citation_hits}/{citation_total} ({cpct:.0f}%){RESET}")

    return {"passed": passed, "total": total, "failures": failures}


def run_conflict_set(client: TestClient, verbose: bool = False) -> Dict[str, Any]:
    data = load("conflict_questions.json")
    items = data["conflict_questions"]
    passed = 0
    failures: List[str] = []
    section_hits = 0
    section_total = 0

    print("\nCONFLICT — questions landing on a planted contradiction")
    print("-" * 54)

    for item in items:
        response = client.post("/ask", json={"question": item["question"], "top_k": 5}).json()
        got = response["response_type"]
        topic_ok = (response.get("conflict") or {}).get("topic") == item["expected_topic"]
        ok = got == "conflict" and topic_ok
        passed += ok
        if not ok:
            failures.append(f'{item["id"]}: got {got} — {item["question"]}')

        # Every clause in the contradiction must be surfaced, not just one.
        if response.get("conflict"):
            reported = {c["section_id"] for c in response["conflict"]["claims"]}
            for expected_section in item["expected_sections"]:
                section_total += 1
                section_hits += expected_section in reported

        mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        if verbose or not ok:
            print(f"  {mark} {item['id']}  {item['question'][:78]}")
            if not ok:
                print(f"{DIM}        got {got}, topic={(response.get('conflict') or {}).get('topic')}{RESET}")

    pct = 100.0 * passed / len(items)
    colour = GREEN if pct == 100 else (YELLOW if pct >= 80 else RED)
    print(f"  {colour}{passed}/{len(items)} correct ({pct:.0f}%){RESET}")
    if section_total:
        spct = 100.0 * section_hits / section_total
        print(f"  {DIM}conflicting clauses surfaced: {section_hits}/{section_total} ({spct:.0f}%){RESET}")

    # Near misses: must not be reported as conflicts.
    near = data["near_miss_questions"]
    print("\nNEAR MISS — must NOT be reported as a conflict")
    print("-" * 45)
    near_passed = 0
    for item in near:
        response = client.post("/ask", json={"question": item["question"], "top_k": 5}).json()
        ok = response["response_type"] != item["must_not_be"]
        near_passed += ok
        mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        if verbose or not ok:
            print(f"  {mark} {item['id']}  {item['question'][:78]}")
        if not ok:
            failures.append(f'{item["id"]}: wrongly flagged as conflict')
    npct = 100.0 * near_passed / len(near)
    colour = GREEN if npct == 100 else RED
    print(f"  {colour}{near_passed}/{len(near)} correct ({npct:.0f}%){RESET}")

    return {
        "passed": passed + near_passed,
        "total": len(items) + len(near),
        "failures": failures,
    }


def main() -> int:
    verbose = "-v" in sys.argv or "--verbose" in sys.argv

    with TestClient(app) as client:
        health = client.get("/health").json()
        print("=" * 62)
        print("RULEBOOK QA — EVALUATION")
        print("=" * 62)
        print(f"sections indexed     : {health['sections_indexed']}")
        print(f"indexed words        : {health['indexed_words']}")
        print(f"retrieval backend    : {health['retrieval_backend']}")
        print(f"answer backend       : {health['answer_backend']}")
        print(f"contested quantities : {len(health['contested_quantities'])} found by the detector")
        for key in health["contested_quantities"]:
            print(f"                       - {key}")

        results = {
            "answered": run_set(
                client,
                "ANSWERED — questions the corpus answers cleanly",
                load("answerable_questions.json")["questions"],
                "answered",
                section_key="expected_section",
                verbose=verbose,
            ),
            "not_covered": run_set(
                client,
                "NOT COVERED — 25 adversarial questions the corpus cannot answer",
                load("not_covered_questions.json")["questions"],
                "not_covered",
                verbose=verbose,
            ),
            "conflict": run_conflict_set(client, verbose=verbose),
        }

    total_passed = sum(r["passed"] for r in results.values())
    total = sum(r["total"] for r in results.values())
    pct = 100.0 * total_passed / total

    print("\n" + "=" * 62)
    print(f"OVERALL: {total_passed}/{total} ({pct:.1f}%)")
    print("=" * 62)

    failures = [f for r in results.values() for f in r["failures"]]
    if failures:
        print(f"\n{RED}Failures:{RESET}")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
