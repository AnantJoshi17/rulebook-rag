"""
Deciding when the rulebook simply does not say.

A similarity threshold alone does not work here, and that is the whole difficulty
of this problem. Ask *"what happens if I miss the exam because of a family
wedding?"* and retrieval returns AR-5.4 — medical absence from the end-semester
examination — with a high score, because the question really is about missing an
exam. The passage is topically perfect and answers a different question. A
threshold-only system answers confidently and wrongly.

So there are two gates:

**Gate 1 — weak match.** Nothing scored above `weak_match_threshold`. The corpus
is not even in the neighbourhood. This catches the easy cases.

**Gate 2 — unsupported circumstance.** The question names a specific
circumstance, and the token naming it appears *nowhere in the corpus vocabulary*.
"wedding", "strike", "quarantine", "internship abroad" — if the word is absent
from a 7,000-word rulebook, no clause in that rulebook is about it, however well
the surrounding words match. This is deliberately conservative: it fires only on
content words, never on question scaffolding ("what", "happens", "if", "can").

Gate 2 is what separates this from a confident-sounding chatbot. Its cost is
recall — a question phrased entirely in rulebook vocabulary can still be
unanswerable and will slip through. That limitation is measured in `evaluate.py`
and stated in the README rather than hidden.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Set

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Words that carry no circumstance of their own. If one of these is missing from
# the corpus it means nothing, so they never trigger Gate 2.
_QUESTION_SCAFFOLDING: Set[str] = {
    "what", "when", "where", "who", "whom", "which", "why", "how", "whether",
    "can", "could", "may", "might", "shall", "should", "will", "would", "must",
    "do", "does", "did", "is", "are", "was", "were", "be", "been", "am",
    "have", "has", "had", "get", "gets", "got", "getting",
    "happen", "happens", "happened", "need", "needs", "needed", "want", "wants",
    "allow", "allowed", "allows", "let", "lets", "make", "makes", "made",
    "take", "takes", "taken", "give", "gives", "given", "go", "goes", "going",
    "come", "comes", "put", "puts", "say", "says", "said", "tell", "tells",
    "know", "knows", "think", "thinks", "please", "kindly", "anyone", "someone",
    "something", "anything", "everything", "somebody", "anybody",
    "i", "me", "my", "mine", "myself", "we", "our", "us", "you", "your",
    "it", "its", "they", "them", "their", "he", "she", "his", "her",
    "if", "then", "else", "but", "and", "or", "not", "no", "yes", "any", "some",
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "from", "with",
    "about", "into", "over", "under", "after", "before", "during", "while",
    "still", "just", "only", "also", "even", "very", "much", "many", "more",
    "long", "far", "often", "soon", "later", "earlier", "back", "again", "ever",
    "night", "day", "days", "time", "times", "way", "ways", "thing", "things",
    "item", "items", "away", "kind", "sort", "type", "types", "amount", "number",
    "option", "options", "point", "points", "part", "parts", "place", "places",
    "stuff", "bit", "lot", "lots", "one", "ones", "person", "people", "student",
    "students", "supposed", "meant", "able", "allowed", "exactly", "normally",
    "most", "less", "least", "own", "same", "other", "another", "such",
    "there", "here", "this", "that", "these", "those", "am", "im", "ive",
    "possible", "possibly", "case", "cases", "situation", "situations",
    "instead", "rather", "actually", "really", "sure", "okay", "ok",
    "sir", "maam", "thanks", "thank", "hello", "hi",
}

_IGNORED = _QUESTION_SCAFFOLDING | set(ENGLISH_STOP_WORDS)

_TOKEN = re.compile(r"[a-z][a-z\-']+")


# Suffix rules cannot reach these, and every one of them turns up in the way
# students actually phrase questions ("when are tests held", "fees I already paid").
_IRREGULAR: dict[str, str] = {
    "held": "hold", "paid": "pay", "sat": "sit", "lost": "lose", "kept": "keep",
    "left": "leave", "caught": "catch", "brought": "bring", "taught": "teach",
    "sent": "send", "spent": "spend", "met": "meet", "found": "find",
    "written": "write", "wrote": "write", "taken": "take", "took": "take",
    "given": "give", "gave": "give", "chose": "choose", "chosen": "choose",
    "withdrew": "withdraw", "withdrawn": "withdraw", "began": "begin",
    "fell": "fall", "fallen": "fall", "sought": "seek", "struck": "strike",
}


def _morphological_variants(token: str) -> List[str]:
    """Cheap stemming: check a few surface forms before declaring a word absent."""
    forms = {token}
    if token in _IRREGULAR:
        forms.add(_IRREGULAR[token])
    for suffix, replacements in (
        ("ies", ["y"]),
        ("ied", ["y"]),
        ("es", ["", "e"]),
        ("s", [""]),
        ("ing", ["", "e"]),
        ("ed", ["", "e"]),
        ("ly", [""]),
    ):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stem = token[: -len(suffix)]
            for replacement in replacements:
                forms.add(stem + replacement)
    # And the other direction, for a question that uses the singular.
    forms.update({token + "s", token + "es", token + "ed", token + "ing"})
    return sorted(forms)


@dataclass
class CoverageVerdict:
    covered: bool
    reason: str
    unsupported_terms: List[str]


class CoverageGate:
    def __init__(self, corpus_vocabulary: Set[str], weak_match_threshold: float) -> None:
        self.vocabulary: Set[str] = corpus_vocabulary
        self.weak_match_threshold = weak_match_threshold

    def _is_present(self, token: str) -> bool:
        if any(form in self.vocabulary for form in _morphological_variants(token)):
            return True
        # Students write "exam"; a regulation writes "examination". Treating those
        # as different words produced false "not covered" verdicts on perfectly
        # answerable questions, so a query token that prefixes a corpus token
        # counts as present. Four characters is the floor — shorter prefixes
        # collide with unrelated words.
        for form in _morphological_variants(token):
            if len(form) >= 4 and any(word.startswith(form) for word in self.vocabulary):
                return True
        # And the reverse, for "registrations" against a corpus that says "register".
        if len(token) >= 6 and any(
            token.startswith(word) for word in self.vocabulary if len(word) >= 5
        ):
            return True
        return False

    def salient_terms(self, question: str) -> List[str]:
        seen: List[str] = []
        for token in _TOKEN.findall(question.lower()):
            if token in _IGNORED or len(token) < 3:
                continue
            if token not in seen:
                seen.append(token)
        return seen

    def check(self, question: str, top_score: float) -> CoverageVerdict:
        if top_score < self.weak_match_threshold:
            return CoverageVerdict(
                covered=False,
                reason=(
                    f"No section scored above {self.weak_match_threshold:.2f} "
                    f"(best was {top_score:.2f}) — the rulebook does not discuss this."
                ),
                unsupported_terms=[],
            )

        unsupported = [t for t in self.salient_terms(question) if not self._is_present(t)]
        if unsupported:
            listed = ", ".join(f"'{t}'" for t in unsupported)
            return CoverageVerdict(
                covered=False,
                reason=(
                    f"The question turns on {listed}, which appears nowhere in the corpus. "
                    "Nearby sections cover related but different circumstances."
                ),
                unsupported_terms=unsupported,
            )

        return CoverageVerdict(covered=True, reason="", unsupported_terms=[])


def build_gate(corpus_texts: Iterable[str], weak_match_threshold: float) -> CoverageGate:
    """
    Build the vocabulary from the raw corpus using the *same* tokenizer that is
    applied to questions.

    Reusing the retriever's TF-IDF vocabulary was tried first and caused a false
    "not covered": that tokenizer keeps "b.tech" as one token, while the question
    side splits it, so "tech" looked absent from a corpus that says it forty
    times. Two tokenizers, one comparison, is always a bug.
    """
    vocabulary: Set[str] = set()
    for text in corpus_texts:
        vocabulary.update(_TOKEN.findall(text.lower()))
    return CoverageGate(vocabulary, weak_match_threshold)
