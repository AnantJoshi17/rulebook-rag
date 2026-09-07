# Planted Contradictions — Ground Truth Log

Three contradictions were deliberately written into the corpus. Each is recorded
here with its exact location so it can be tested against. The detector in
`app/claims.py` is **not** given this file — it finds these by comparing extracted
normative claims, so this log is ground truth for evaluation only.

---

## Contradiction 1 — Minimum attendance to sit the end-semester examination

A three-way conflict, spanning two documents and two formats (markdown + PDF).

| # | Location | File | What it says |
|---|---|---|---|
| a | **AR-3.2** | `01_academic_regulations.md` | Not less than **75%** attendance in a course is required to appear in the end-semester examination. States the threshold is *mandatory* and applies to every course individually. |
| b | **AR-7.4** | `01_academic_regulations.md` | With a medical certificate, a student is eligible to appear provided attendance is not less than **65%**. |
| c | **EC-3.1** | `05_examination_committee_charter.pdf` | *Notwithstanding any provision of Document AR*, the Examination Committee may **waive the attendance requirement in whole or in part**. |

**Why it is a contradiction:** AR-3.2 states an unqualified mandatory floor of 75%
and says no student below it may appear. AR-7.4 permits a student at 65% to appear.
EC-3.1 permits a waiver down to no floor at all. All three govern the same question —
*what attendance do I need to sit the exam?* — and give three different answers.

**Test question:** "What is the minimum attendance I need to sit the end-semester exam?"

---

## Contradiction 2 — Notice period for overnight absence from the hostel

Both clauses sit inside the same document, five sections apart.

| # | Location | File | What it says |
|---|---|---|---|
| a | **HH-4.1** | `02_hostel_handbook.md` | A resident intending to be absent overnight shall apply in the leave register not less than **48 hours** before departure. |
| b | **HH-9.3** | `02_hostel_handbook.md` | A resident travelling home or elsewhere over a weekend shall record the absence not less than **24 hours** before departure. |

**Why it is a contradiction:** A weekend trip home *is* an overnight absence. HH-4.1
demands 48 hours' notice for it and HH-9.3 demands 24. A resident leaving on Friday
evening cannot satisfy both readings, and neither clause is expressed as an exception
to the other.

**Test question:** "How much notice do I have to give before leaving the hostel overnight?"

---

## Contradiction 3 — CGPA required to keep a Merit Scholarship

Spans two documents, and the two clauses are worded as if describing different things
while in fact governing the same outcome.

| # | Location | File | What it says |
|---|---|---|---|
| a | **SP-2.1** | `03_scholarship_policy.md` | A Merit Scholarship is renewed only where the holder has a CGPA of not less than **8.00**. Below 8.00 the holder *ceases to hold* the scholarship. |
| b | **FS-6.2** | `04_fee_schedule.md` | Where a student holds a Merit Scholarship, the tuition remission **continues** into the following year provided the student maintains a CGPA of not less than **7.50**. |

**Why it is a contradiction:** The scholarship *is* the remission — SP-1.1 defines it as
"a remission of a stated portion of the tuition fee." A student at CGPA 7.70 has lost
the scholarship under SP-2.1 but keeps the remission under FS-6.2, and FS-6.2 explicitly
says no fresh application is needed. The two documents disagree on the same threshold.

**Test question:** "What CGPA do I need to keep my merit scholarship next year?"

---

## Near-miss cases (deliberately *not* contradictions)

These were written to look like conflicts but are consistent. A detector that flags
them is over-triggering, so they are included in the evaluation set as negatives.

| Sections | Why it is **not** a conflict |
|---|---|
| AR-4.3 (7 working days for a medical certificate after a mid-sem test) vs AR-5.4 (10 working days after an end-semester exam) | Different events, different deadlines. Both can hold at once. |
| HH-2.4 (24 hours' notice for room inspection) vs HH-4.1 (48 hours' notice for leave) | Same unit, unrelated subjects — one binds the Warden, one binds the resident. |
| FS-3.1 (no admit card if dues unpaid) vs AR-5.2 (admit card requires attendance and no dues) | Consistent and mutually reinforcing. |
