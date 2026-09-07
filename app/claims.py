"""
Contradiction detection.

The naive approach — "ask an LLM whether these two passages disagree" — is
unreliable and unexplainable. This module does something narrower and testable.

**The model.** A rulebook contradicts itself when two clauses fix *different
values for the same regulated quantity*. So the corpus is read as a set of
`Claim`s, where a claim is:

    (regulated quantity, value, the sentence that states it, section id)

Two claims on the same quantity with different values are a conflict. A claim
that asserts a *power to waive* a quantity conflicts with any claim that fixes
it.

**What is declared vs what is discovered.** The regulated *quantities* are
declared below — attendance-for-exam-eligibility, scholarship-retention CGPA,
and so on. The *values*, the *sentences* and *which sections collide* are all
discovered by parsing the corpus. The detector is never told that AR-3.2 and
AR-7.4 disagree; it finds that by reading them. Delete a contradiction from the
corpus and the detector stops reporting it, with no code change.

**Why quantities are declared.** A fully generic "two numbers, same unit, similar
text" detector was tried first and over-triggered badly. AR-4.3 (medical
certificate within *seven* working days of a mid-semester test) and AR-5.4
(within *ten* working days of an end-semester examination) are near-identical in
wording and both state a deadline in days — but they govern different events and
are perfectly consistent. Discriminating them needs to know *what the number
measures*, which is exactly what a declared quantity encodes. The near-misses are
kept as negatives in `tests/contradictions_log.md`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .corpus_loader import Chunk

# --------------------------------------------------------------------------- #
# Number parsing
# --------------------------------------------------------------------------- #

_UNITS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def words_to_number(phrase: str) -> float | None:
    """'seventy-five' -> 75.0, 'ten' -> 10.0. Handles the 1-99 range this corpus uses."""
    tokens = re.split(r"[\s-]+", phrase.strip().lower())
    total = 0.0
    matched = False
    for token in tokens:
        if token in _TENS:
            total += _TENS[token]
            matched = True
        elif token in _UNITS:
            total += _UNITS[token]
            matched = True
        elif token in {"and", "per", "cent"}:
            continue
        else:
            break
    return total if matched else None


_WORD_NUM = r"(?:[a-z]+(?:-[a-z]+)?)"


def extract_number(text: str, pattern: str) -> tuple[float, str] | None:
    """
    Run a quantity pattern and return (value, matched_sentence_fragment).

    Digit forms win over word forms: the corpus writes 'seventy-five per cent
    (75%)', and 75 is less ambiguous to parse than the words beside it.
    """
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    groups = match.groupdict()
    digits = groups.get("digits")
    if digits:
        try:
            return float(digits.replace(",", "")), match.group(0)
        except ValueError:
            pass
    words = groups.get("words")
    if words:
        value = words_to_number(words)
        if value is not None:
            return value, match.group(0)
    return None


# --------------------------------------------------------------------------- #
# Regulated quantities
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Quantity:
    """A single regulated quantity that clauses may fix a value for."""

    key: str
    label: str
    unit: str
    # Every group must have at least one term present for a clause to be *about*
    # this quantity. Groups are ANDed, terms within a group are ORed.
    required_terms: Sequence[Sequence[str]]
    value_pattern: str
    # Words that mean "this clause claims the power to set the value aside".
    override_terms: Sequence[str] = ()
    question_terms: Sequence[Sequence[str]] = ()

    def matches_clause(self, text: str) -> bool:
        low = text.lower()
        return all(any(term in low for term in group) for group in self.required_terms)

    def matches_question(self, question: str) -> bool:
        low = question.lower()
        groups = self.question_terms or self.required_terms
        return all(any(term in low for term in group) for group in groups)


_PCT = rf"(?:(?P<words>{_WORD_NUM}(?:-{_WORD_NUM})?)\s+per\s+cent\s*)?\(?(?P<digits>\d{{1,3}})\s*(?:%|per\s+cent)"
_CGPA = r"(?:not\s+less\s+than|of|below|maintains?)\s+(?P<digits>\d\.\d{1,2})"
_HOURS = rf"(?:(?P<words>{_WORD_NUM}(?:-{_WORD_NUM})?)\s*)?\(?(?P<digits>\d{{1,3}})\)?\s*hours"
_DAYS = rf"(?:within\s+)?(?:(?P<words>{_WORD_NUM}(?:-{_WORD_NUM})?)|(?P<digits>\d{{1,3}}))\s+working\s+days"

QUANTITIES: List[Quantity] = [
    Quantity(
        key="attendance_for_exam_eligibility",
        label="Minimum attendance required to appear in the end-semester examination",
        unit="%",
        required_terms=[
            ["attend", "attendance", "contact hours"],
            ["end-semester examination", "examination", "eligible", "eligibility", "appear"],
        ],
        value_pattern=_PCT,
        override_terms=["waive", "relax", "notwithstanding"],
        question_terms=[
            ["attendance", "attend", "classes", "present"],
            ["exam", "examination", "sit", "appear", "eligible", "eligibility", "write"],
        ],
    ),
    Quantity(
        key="scholarship_retention_cgpa",
        label="CGPA required to keep a Merit Scholarship (tuition remission) for the next year",
        unit="CGPA",
        required_terms=[
            ["scholarship", "remission"],
            ["cumulative grade point average", "cgpa"],
            ["renew", "continue", "continuation", "maintain", "cease", "following academic year"],
        ],
        value_pattern=_CGPA,
        question_terms=[
            ["scholarship", "remission", "fee waiver"],
            ["cgpa", "grade point", "marks", "keep", "renew", "retain", "continue", "lose"],
        ],
    ),
    Quantity(
        key="hostel_overnight_leave_notice",
        label="Notice a hostel resident must give before an overnight absence",
        unit="hours",
        required_terms=[
            ["hostel", "resident"],
            ["leave register", "leave", "absent", "absence", "depart"],
            ["before", "prior"],
        ],
        value_pattern=_HOURS,
        question_terms=[
            ["hostel"],
            ["leave", "absent", "absence", "overnight", "weekend", "go home", "notice"],
        ],
    ),
    # Declared but *not* contradictory — proves the detector discriminates by the
    # event a deadline attaches to rather than firing on "two numbers in days".
    Quantity(
        key="medical_certificate_deadline_midsem",
        label="Deadline to submit a medical certificate after a missed mid-semester test",
        unit="working days",
        required_terms=[
            ["medical certificate"],
            ["mid-semester test", "the test"],
        ],
        value_pattern=_DAYS,
    ),
    Quantity(
        key="medical_certificate_deadline_endsem",
        label="Deadline to submit a medical certificate after a missed end-semester examination",
        unit="working days",
        required_terms=[
            ["medical certificate"],
            ["end-semester examination"],
        ],
        value_pattern=_DAYS,
    ),
]

QUANTITIES_BY_KEY: Dict[str, Quantity] = {q.key: q for q in QUANTITIES}


# --------------------------------------------------------------------------- #
# Claims
# --------------------------------------------------------------------------- #


@dataclass
class Claim:
    quantity_key: str
    section_id: str
    document: str
    kind: str  # "value" | "override"
    value: float | None
    sentence: str

    @property
    def stated_value(self) -> str:
        quantity = QUANTITIES_BY_KEY[self.quantity_key]
        if self.kind == "override":
            return "may be waived entirely"
        if quantity.unit == "%":
            return f"{self.value:g}%"
        if quantity.unit == "CGPA":
            return f"CGPA {self.value:.2f}"
        return f"{self.value:g} {quantity.unit}"


_SENTENCE_SPLIT = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")


def _sentences(text: str) -> List[str]:
    cleaned = re.sub(r"\*\*|\s+", lambda m: "" if m.group(0) == "**" else " ", text).strip()
    return [s.strip() for s in _SENTENCE_SPLIT.split(cleaned) if s.strip()]


def extract_claims(chunk: Chunk) -> List[Claim]:
    """Read one section and return every claim it makes about a regulated quantity."""
    claims: List[Claim] = []
    sentences = _sentences(chunk.text)

    for quantity in QUANTITIES:
        if not quantity.matches_clause(chunk.text):
            continue

        # An override is a claim about the quantity that fixes no number.
        if quantity.override_terms:
            low = chunk.text.lower()
            if any(term in low for term in quantity.override_terms) and re.search(
                r"\bwaiv\w*|\brelax\w*", low
            ):
                override_sentence = next(
                    (s for s in sentences if re.search(r"waiv\w*|relax\w*", s, re.I)),
                    sentences[0] if sentences else chunk.text,
                )
                claims.append(
                    Claim(
                        quantity_key=quantity.key,
                        section_id=chunk.section_id,
                        document=chunk.document,
                        kind="override",
                        value=None,
                        sentence=override_sentence,
                    )
                )
                continue

        # Otherwise look for a fixed value, preferring the sentence that also
        # carries the quantity's subject terms.
        for sentence in sentences:
            if not quantity.matches_clause(sentence):
                continue
            found = extract_number(sentence, quantity.value_pattern)
            if found is None:
                continue
            value, _ = found
            claims.append(
                Claim(
                    quantity_key=quantity.key,
                    section_id=chunk.section_id,
                    document=chunk.document,
                    kind="value",
                    value=value,
                    sentence=sentence,
                )
            )
            break

    return claims


class ClaimIndex:
    """All claims in the corpus, grouped by regulated quantity."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.by_section: Dict[str, List[Claim]] = {}
        self.by_quantity: Dict[str, List[Claim]] = {}
        for chunk in chunks:
            claims = extract_claims(chunk)
            if claims:
                self.by_section[chunk.section_id] = claims
            for claim in claims:
                self.by_quantity.setdefault(claim.quantity_key, []).append(claim)

    def contested_quantities(self) -> Dict[str, List[Claim]]:
        """Quantities where the corpus as a whole disagrees with itself."""
        contested: Dict[str, List[Claim]] = {}
        for key, claims in self.by_quantity.items():
            if self._is_contested(claims):
                contested[key] = claims
        return contested

    @staticmethod
    def _is_contested(claims: Sequence[Claim]) -> bool:
        if len(claims) < 2:
            return False
        values = {c.value for c in claims if c.kind == "value"}
        has_override = any(c.kind == "override" for c in claims)
        # Two different fixed values, or a fixed value plus a power to waive it.
        return len(values) > 1 or (has_override and len(values) >= 1)

    def conflict_for(
        self, retrieved_section_ids: Sequence[str], question: str
    ) -> tuple[str, List[Claim]] | None:
        """
        Decide whether the question lands on a contested quantity.

        Requires both: the question is *about* the quantity, and retrieval
        actually surfaced at least one clause that speaks to it. The second
        condition stops the system announcing a conflict it cannot cite.
        """
        retrieved = set(retrieved_section_ids)
        best: tuple[str, List[Claim]] | None = None

        for key, claims in self.contested_quantities().items():
            quantity = QUANTITIES_BY_KEY[key]
            if not quantity.matches_question(question):
                continue
            if not any(c.section_id in retrieved for c in claims):
                continue
            # Prefer the quantity with more of its clauses actually retrieved.
            grounded = sum(1 for c in claims if c.section_id in retrieved)
            if best is None or grounded > sum(1 for c in best[1] if c.section_id in retrieved):
                best = (key, claims)

        return best
