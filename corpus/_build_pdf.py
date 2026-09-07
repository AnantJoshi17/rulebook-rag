"""
Generates 05_examination_committee_charter.pdf.

The corpus deliberately mixes formats (markdown, tables, PDF). This script
builds the PDF member of the corpus so the repository stays reproducible:
anyone cloning it can regenerate the PDF with `python corpus/_build_pdf.py`.
"""

from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).parent / "05_examination_committee_charter.pdf"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceAfter=10)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=6)
BODY = ParagraphStyle(
    "BODY", parent=styles["BodyText"], fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=6
)

CONTENT = [
    ("h1", "Examination Committee Charter"),
    ("body", "<b>Document ID:</b> EC &nbsp;&nbsp; <b>Version:</b> 2026.1 &nbsp;&nbsp; "
             "<b>Approved by:</b> Academic Council, 14 March 2026"),

    ("h2", "Section EC-1. Constitution"),
    ("h2", "EC-1.1 Establishment"),
    ("body", "There shall be an Examination Committee of the University, constituted by the Academic "
             "Council, to discharge the functions assigned to it by these regulations and by Document AR."),
    ("h2", "EC-1.2 Composition"),
    ("body", "The Examination Committee shall comprise the Dean of Academics as Chairperson, the "
             "Controller of Examinations as Member Secretary, three Heads of Department nominated by "
             "the Academic Council, one senior faculty member nominated by the Vice-Chancellor, and the "
             "Dean of Students. Members other than ex officio members hold office for two years."),
    ("h2", "EC-1.3 Quorum"),
    ("body", "The quorum for a meeting of the Examination Committee is four members, of whom at least "
             "one shall be a Head of Department. Decisions are taken by a majority of members present "
             "and voting. In the event of an equality of votes the Chairperson has a casting vote."),
    ("h2", "EC-1.4 Frequency of Meetings"),
    ("body", "The Committee shall meet not fewer than four times in an academic year, and additionally "
             "whenever a matter is referred to it under AR-5.6 or AR-8.1."),

    ("h2", "Section EC-2. Functions"),
    ("h2", "EC-2.1 Examination Offences"),
    ("body", "The Committee shall enquire into every case of an examination offence reported under "
             "AR-5.6 and AR-8.3, and into every second instance of plagiarism referred under AR-8.1."),
    ("h2", "EC-2.2 Procedure"),
    ("body", "The Committee shall give the student written notice of the allegation, shall afford the "
             "student an opportunity to be heard in person, and shall record its findings in writing. "
             "The student may be accompanied by one person of the student's choosing, who shall not "
             "address the Committee."),
    ("h2", "EC-2.3 Penalties"),
    ("body", "On a finding that an examination offence has been committed, the Committee may cancel the "
             "result of the student in the course concerned, cancel the result of the student in the "
             "semester concerned, or debar the student from appearing in examinations for a stated "
             "period not exceeding two semesters. The Committee shall record reasons for the penalty "
             "imposed."),
    ("h2", "EC-2.4 Appeal"),
    ("body", "A student aggrieved by a decision of the Committee may appeal to the Vice-Chancellor "
             "within fifteen working days of communication of the decision. The Vice-Chancellor may "
             "confirm, reduce or set aside the penalty, but shall not enhance it."),

    ("h2", "Section EC-3. Powers of Relaxation"),
    ("h2", "EC-3.1 General Power to Relax"),
    ("body", "<b>Notwithstanding any provision of Document AR, the Examination Committee may, on the "
             "written application of a student and for reasons recorded in writing, waive the minimum "
             "attendance requirement in a course in whole or in part and permit the student to appear "
             "in the end-semester examination for that course.</b> The Committee shall satisfy itself "
             "that the circumstances of the case are exceptional and that the student is otherwise "
             "prepared to appear in the examination."),
    ("h2", "EC-3.2 Application for Relaxation"),
    ("body", "An application under EC-3.1 shall be submitted to the Member Secretary not later than "
             "seven working days before the commencement of the end-semester examination period, "
             "together with such documentary evidence as the student wishes the Committee to consider."),
    ("h2", "EC-3.3 Record of Relaxations"),
    ("body", "Every relaxation granted under EC-3.1 shall be recorded in the minutes of the Committee "
             "and reported to the Academic Council at its next meeting. The Academic Council may direct "
             "that a relaxation be reviewed."),
    ("h2", "EC-3.4 Limits"),
    ("body", "The Committee shall not relax the passing standard in AR-6.2, the credit requirement in "
             "AR-10.1, or the maximum duration in AR-10.2."),

    ("h2", "Section EC-4. Results"),
    ("h2", "EC-4.1 Approval of Results"),
    ("body", "The Committee shall approve the end-semester results placed before it by the Controller of "
             "Examinations before publication. Approval may be given by circulation where the Chairperson "
             "so directs."),
    ("h2", "EC-4.2 Withheld Results"),
    ("body", "Where the result of a student has been withheld under AR-5.6 or under the Fee Schedule, the "
             "Controller of Examinations shall place the matter before the Committee at its next meeting "
             "with a statement of the reason for withholding."),
    ("h2", "EC-4.3 Correction of Results"),
    ("body", "Where an error in a published result is detected, the Controller of Examinations shall "
             "report the error to the Chairperson, who may authorise immediate correction and shall "
             "report the correction to the Committee at its next meeting."),

    ("h2", "Section EC-5. Records"),
    ("h2", "EC-5.1 Minutes"),
    ("body", "Minutes of every meeting shall be recorded by the Member Secretary, confirmed at the "
             "following meeting, and retained permanently."),
    ("h2", "EC-5.2 Confidentiality"),
    ("body", "Proceedings of the Committee relating to an individual student are confidential. The "
             "outcome shall be communicated in writing to the student, to the Head of Department "
             "concerned and to the Controller of Examinations, and to no other person."),
]


def build() -> None:
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Examination Committee Charter",
    )
    flow = []
    for kind, text in CONTENT:
        if kind == "h1":
            flow.append(Paragraph(text, H1))
        elif kind == "h2":
            flow.append(Paragraph(text, H2))
        else:
            flow.append(Paragraph(text, BODY))
    flow.append(Spacer(1, 0.5 * cm))
    doc.build(flow)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
