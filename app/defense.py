"""Defence: trust tiers, instruction hierarchy, and the grounding check.

Trust tiers
-----------
- Tier 1 (trusted): ``data/corpus/`` — summaries of official Canadian
  guidance (OPC, Treasury Board). Only tier 1 may be cited as authority.
- Tier 2 (untrusted): ``data/untrusted/`` — synthetic test passages.
  Loaded only with an explicit flag. Never cited, never treated as
  instructions, never used as evidence.

Instruction hierarchy (highest to lowest)
------------------------------------------
1. System instructions (this code, the API contract).
2. Trusted retrieved documents (tier 1) — evidence only, never commands.
3. The user's question — the task to answer.
4. Untrusted content (tier 2) — data only. It can never instruct the
   model, override higher tiers, or supply facts.

An instruction hidden in retrieved content is an *indirect prompt
injection*: the defence is to treat retrieved text as data, scan it for
instruction-like language, flag it, and exclude it from the answer.
"""

from __future__ import annotations

import re

# Patterns that look like instructions smuggled into retrieved *data*.
INSTRUCTION_PATTERNS: list[tuple[str, str]] = [
    (r"\bignore\b.{0,40}\b(pipeda|consent|guidance|instruction)", "ignore-guidance"),
    (r"\bdisregard\b", "disregard"),
    (r"\b(system\s+)?override\b", "override"),
    (r"\bfrom now on\b", "from-now-on"),
    (r"\byou must (say|tell|answer|respond|ignore|disregard)\b", "you-must"),
    (r"\btell the user\b", "tell-the-user"),
    (r"\bdo not mention\b", "do-not-mention"),
    (r"\bnever mention\b", "never-mention"),
    (r"\bthis instruction takes priority\b", "takes-priority"),
    (r"\bimportant( system)? (instruction|override)\b", "important-instruction"),
    (r"\bshare this\b.{0,30}\b(freely|anyone)\b", "share-freely-instruction"),
    (r"\byou may share\b", "you-may-share"),
]

# Heuristics for poisoned *claims* (false facts contradicting tier 1).
POISON_PATTERNS: list[tuple[str, str]] = [
    (r"\bshared? freely\b", "share-freely-claim"),
    (r"\bno limits on (use|disclosure|retention)\b", "no-limits-claim"),
    (r"\bwithout consent\b.{0,30}\b(any|freely)\b", "without-consent-claim"),
    (r"\b\d{3}-\d{3}-\d{3}\b", "sin-like-number"),
    (r"\b123 fake street\b", "fake-address"),
]

PERSONAL_DATA_REQUEST = re.compile(
    r"(social insurance|sin\b|home address|address of|phone number|date of birth)"
    r".{0,40}(of|for)\b|"
    r"\b(what is|give me|reveal|share)\b.{0,40}"
    r"(social insurance|sin\b|home address)",
    re.IGNORECASE,
)


def scan_instructions(text: str) -> list[str]:
    """Return the labels of instruction-like patterns found in ``text``."""
    found = []
    low = text.lower()
    for pattern, label in INSTRUCTION_PATTERNS:
        if re.search(pattern, low):
            found.append(label)
    return found


def scan_poison(text: str) -> list[str]:
    """Return the labels of poison-like claims found in ``text``."""
    found = []
    low = text.lower()
    for pattern, label in POISON_PATTERNS:
        if re.search(pattern, low):
            found.append(label)
    return found


def is_personal_data_request(question: str) -> bool:
    """True if the question asks for a specific person's private data."""
    return bool(PERSONAL_DATA_REQUEST.search(question))


def filter_trusted(chunks: list[dict]) -> list[dict]:
    """Keep only tier-1 chunks. Tier-2 chunks can never be evidence."""
    return [c for c in chunks if c.get("trust_tier") == 1]


def check_grounding(answer: str, citations: list[dict], trusted_doc_ids: set[str]) -> bool:
    """Grounding check: every citation in the answer must trace to a
    cited, trusted passage. Returns False if any citation is untrusted,
    uncited in the answer text, or missing from the trusted set."""
    if not citations:
        return "[not supported" in answer.lower() or "cannot answer" in answer.lower()
    for cit in citations:
        if cit["doc_id"] not in trusted_doc_ids:
            return False
        if f"[{cit['doc_id']}]" not in answer:
            return False
    return True
