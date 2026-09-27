"""Deterministic guardrails for the chat app. No Databricks or model dependencies, so they are cheap to run on
every request and easy to unit-test (tests/test_guardrails.py).

    find_identifiers   text that must not leave the boundary (patient names incl. misspellings, MRN, SSN, ...)
    screen_input       prompt-injection patterns and length limit
    check_grounded     every digit-bearing fact in a model answer must appear in the retrieved records
    RateLimiter        per-user sliding window (the endpoint-level limits are shared by all app users)
"""
import re
import threading
import time
from collections import defaultdict, deque
from difflib import SequenceMatcher


class GuardrailBlocked(Exception):
    """A request was stopped by a guardrail. The message is safe to show the user."""


# --------------------------------------------------------------------------------------------- identifiers
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
PHONE_RE = re.compile(r"\b(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
LONG_ID_RE = re.compile(r"\b\d{7,10}\b")  # MRN-shaped numbers
DATE_RE = re.compile(r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:\d{2}|\d{4})\b")  # e.g. a birth date

FUZZY_NAME_RATIO = 0.85
MIN_NAME_PART = 4  # single first/last names shorter than this ("Lee", "Roe") would match too many ordinary words


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _name_hit(text: str, names: set[str]) -> bool:
    words = _words(text)
    joined = " ".join(words)
    for name in names:
        if name in joined:
            return True
        parts = name.split()
        # a distinctive single part of a name (first or last) on its own
        if any(len(p) >= MIN_NAME_PART and p in words for p in parts):
            return True
        # misspelled full name: compare every window of the same word count
        n = len(parts)
        for i in range(len(words) - n + 1):
            if SequenceMatcher(None, " ".join(words[i:i + n]), name).ratio() >= FUZZY_NAME_RATIO:
                return True
    return False


def find_identifiers(text: str, patient_names: set[str]) -> list[str]:
    """Kinds of identifier found in `text`; an empty list means it looks safe to send to a hosted model."""
    flagged = []
    if patient_names and _name_hit(text, patient_names):
        flagged.append("patient_name")
    for kind, rx in (("ssn", SSN_RE), ("phone", PHONE_RE), ("email", EMAIL_RE), ("mrn_like", LONG_ID_RE),
                     ("date", DATE_RE)):
        if rx.search(text):
            flagged.append(kind)
    return flagged


# --------------------------------------------------------------------------------------------- input screen
MAX_QUERY_CHARS = 500

INJECTION_PATTERNS = [
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier|your)\b.*\b(?:instructions?|rules?|prompts?)",
    r"disregard (?:all |any |the )?(?:previous|prior|above|earlier|your)?\s*(?:instructions?|rules?|prompts?)",
    r"(?:reveal|show|print|repeat|tell me)\b.*\b(?:system prompt|your instructions|your prompt|hidden prompt)",
    r"\bsystem prompt\b",
    r"you are now\b",
    r"\bpretend (?:to be|you are)\b",
    r"\b(?:developer|debug|admin|god) mode\b",
    r"\bjailbreak\b",
    r"\b(?:dump|exfiltrate|export)\b.*\b(?:patients?|records?|database|table|data)\b",
    r"\ball (?:of )?(?:the )?patients?['’]?s? (?:records?|data|names?|mrns?|information)\b",
    r"\bevery patient['’]?s? (?:records?|data|names?|mrns?|information)\b",
    r"\bwithout (?:any )?(?:restrictions?|filters?|checks?)\b",
    r"\bbypass\b.*\b(?:filter|policy|security|guardrail|access|restriction)s?\b",
]
_INJECTION_RE = [re.compile(p, re.I) for p in INJECTION_PATTERNS]


def screen_input(query: str) -> None:
    """Raise GuardrailBlocked for oversized input or a known prompt-injection / bulk-extraction pattern.
    Pattern matching is easily bypassed by a determined attacker; it stops the obvious attempts."""
    if len(query) > MAX_QUERY_CHARS:
        raise GuardrailBlocked(f"Questions are limited to {MAX_QUERY_CHARS} characters.")
    for rx in _INJECTION_RE:
        if rx.search(query):
            raise GuardrailBlocked("That request looks like an attempt to override the assistant's rules or to "
                                   "extract records in bulk, so it was blocked.")


# --------------------------------------------------------------------------------------------- grounding
_FACT_RE = re.compile(r"[A-Za-z]*\d[\w./:-]*")  # anything containing a digit: numbers, MRNs, codes, times, dates


def _norm(token: str) -> str:
    token = token.strip(".,;:()[]\"'").lower()
    return re.sub(r"(?<![\d])0+(?=\d)", "", token)  # 06:30 == 6:30


def check_grounded(answer: str, records: list[str]) -> list[str]:
    """Digit-bearing facts in `answer` that do not appear in `records`. An empty list means every
    number, ID, code, time and date the model stated is supported by the retrieved records."""
    support = _norm(" ".join(records))
    support_tokens = {_norm(t) for t in _FACT_RE.findall(" ".join(records))}
    missing = []
    for t in _FACT_RE.findall(answer):
        n = _norm(t)
        if n and n not in support_tokens and n not in support:
            missing.append(t.strip(".,;:"))
    return sorted(set(missing))


# --------------------------------------------------------------------------------------------- rate limit
class RateLimiter:
    """Sliding window per key (the signed-in user's email). In-memory: it resets on redeploy and is per app
    instance, which is enough for a single-instance app."""

    def __init__(self, limit: int = 20, window_s: float = 60.0, clock=time.time):
        self.limit, self.window_s, self.clock = limit, window_s, clock
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self.clock()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] >= self.window_s:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True


# --------------------------------------------------------------------------------------------- hash chain
import hashlib
import json

CHAIN_FIELDS = ["event_id", "event_time", "user_email", "status", "path", "intent", "model", "model_version",
                "chunk_ids", "flags", "question", "answer"]


def row_hash(prev_hash: str | None, record: dict) -> str:
    """SHA-256 over the previous row's hash plus this row's content. Editing or removing any row changes every
    hash after it, which verify_chain() then reports."""
    payload = json.dumps([record.get(f) for f in CHAIN_FIELDS], separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(((prev_hash or "") + "|" + payload).encode("utf-8")).hexdigest()


def verify_chain(rows: list[dict]) -> list[str]:
    """`rows` in seq order, each with the CHAIN_FIELDS plus seq, prev_hash and row_hash. Returns problems found
    (empty means the chain is intact). Detects edited rows, removed rows and reordered rows."""
    problems, prev, expected_seq = [], None, 1
    for r in rows:
        if int(r["seq"]) != expected_seq:
            problems.append(f"seq {r['seq']}: expected {expected_seq} (a row is missing or out of order)")
            expected_seq = int(r["seq"])
        if (r.get("prev_hash") or None) != prev:
            problems.append(f"seq {r['seq']}: prev_hash does not match the previous row")
        if row_hash(r.get("prev_hash"), r) != r["row_hash"]:
            problems.append(f"seq {r['seq']}: row content does not match its hash (edited)")
        prev, expected_seq = r["row_hash"], expected_seq + 1
    return problems
