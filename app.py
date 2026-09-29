#!/usr/bin/env python3
"""Meridian Retail support-draft review tool.

Smallest useful scope: pick one supplied case, show the supplied facts and the
supplied policy, and print a draft reply for a human to review.

What this tool deliberately cannot do (see policy P4):
  * it never sends a message to a customer
  * it never approves a refund
  * it never deletes or changes a record

It makes a single text-generation call and nothing else. There is no web UI, no
database, no retrieval system, no login, and no agent loop.

Usage:
    python app.py --case C1
    python app.py --case C2 --offline-demo
    python app.py --case C3 --simulate-timeout

Provider: OpenCode Zen, the Big Pickle model, over the OpenAI-compatible
chat completions endpoint. See https://opencode.ai/docs/zen/.

Environment (read from a project-only .env, see setup_env.py):
    APP_OPENCODE_API_KEY   Zen API key, passed explicitly to the client
    APP_OPENCODE_MODEL     model id, defaults to space-bunny-free
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, TextIO

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CASES_FILE = BASE_DIR / "cases.json"
POLICY_FILE = BASE_DIR / "policy.md"
DEMO_FILE = BASE_DIR / "demo_outputs.json"

CASE_IDS = ("C1", "C2", "C3")
DEFAULT_CASE = "C1"

ENV_KEY = "APP_OPENCODE_API_KEY"
ENV_MODEL = "APP_OPENCODE_MODEL"
# space-bunny-free is the default because it is the one Zen model that answers
# direct API calls on a free-tier key. big-pickle, mimo-*-free and
# nemotron-*-free all return 403 FreeTierError ("can only be used from within
# OpenCode"); paid models return 402 until the account has funds.
DEFAULT_MODEL = "space-bunny-free"

# OpenCode Zen, Big Pickle. Documented as the OpenAI-compatible chat
# completions route; the client appends /chat/completions to this base.
ZEN_BASE_URL = "https://opencode.ai/zen/v1"
REQUEST_TIMEOUT_SECONDS = 20.0
MAX_RETRIES = 0
MAX_TOKENS = 2048

REQUIRED_KEYS = ("draft_reply", "evidence_refs", "review_status")
ALLOWED_STATUSES = ("READY_FOR_HUMAN_REVIEW", "NEEDS_INFORMATION", "BLOCKED")
UNAVAILABLE_STATUS = "UNAVAILABLE"

FACTS_HEADER = "=== SUPPLIED FACTS ==="
POLICY_HEADER = "=== SUPPLIED POLICY ==="
RESULT_HEADER = "=== DRAFT RESULT FOR HUMAN REVIEW ==="
NO_API_NOTICE = (
    "NO MODEL API WAS CALLED - prerecorded synthetic demo output from demo_outputs.json."
)

EXIT_OK = 0
EXIT_UNAVAILABLE = 1
EXIT_NO_KEY = 2

SYSTEM_PROMPT = (
    "You draft support replies for a human reviewer at a retail company.\n"
    "Hard rules:\n"
    "- Use ONLY the supplied case facts and the supplied policy text below.\n"
    "- Never invent facts, dates, order details, amounts, or policy numbers.\n"
    "- You cannot send a message, approve a refund, or delete or change any record. "
    "You only write draft text for a human to review.\n"
    "Reply with ONE JSON object and nothing else, no prose, no markdown fence, with "
    'exactly these three keys:\n'
    '  "draft_reply": string, the drafted reply text for a human to review.\n'
    '  "evidence_refs": array of strings, each a policy ID copied from the supplied '
    "policy (for example P1).\n"
    '  "review_status": one of "READY_FOR_HUMAN_REVIEW", "NEEDS_INFORMATION", '
    '"BLOCKED".\n'
    "Use NEEDS_INFORMATION when a fact the policy requires is missing.\n"
    "Use BLOCKED when you must not answer without a human decision.\n"
    "Otherwise use READY_FOR_HUMAN_REVIEW."
)


class InputError(Exception):
    """Supplied input files are missing or unusable."""


class DraftInvalid(Exception):
    """Model output failed structural or policy validation."""


# --------------------------------------------------------------------------
# Loading supplied inputs (always local, never any network)
# --------------------------------------------------------------------------


def policy_ids(policy_text: str) -> List[str]:
    """Return the policy IDs declared in the policy text, in file order."""
    return re.findall(r"^\s*(P\d+)\.", policy_text, re.MULTILINE)


def _policy_id_for(policy_text: str, keyword: str) -> str:
    """Best-effort lookup of the policy ID whose line mentions a keyword."""
    for line in policy_text.splitlines():
        if keyword.lower() in line.lower():
            match = re.match(r"^\s*(P\d+)\.", line)
            if match:
                return match.group(1)
    return "the policy"


def load_policy(path: Path = POLICY_FILE) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"cannot read policy file {path}: {exc}") from exc
    if not policy_ids(text):
        raise InputError(f"policy file {path} contains no policy IDs (expected lines like 'P1. ...')")
    return text


def load_case(case_id: str, path: Path = CASES_FILE) -> Dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"cannot read cases file {path}: {exc}") from exc
    try:
        cases = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InputError(f"cases file {path} is not valid JSON: {exc}") from exc
    if not isinstance(cases, dict) or case_id not in cases:
        available = ", ".join(sorted(cases)) if isinstance(cases, dict) else "none"
        raise InputError(f"case {case_id!r} not found in {path} (available: {available})")
    case = cases[case_id]
    if not isinstance(case, dict):
        raise InputError(f"case {case_id!r} in {path} is not a JSON object")
    return case


def _load_demo_file(path: Path = DEMO_FILE) -> Dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"cannot read demo file {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InputError(f"demo file {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise InputError(f"demo file {path} is not a JSON object")
    return data


def load_demo_output(case_id: str, path: Path = DEMO_FILE) -> Dict[str, Any]:
    data = _load_demo_file(path)
    cases = data.get("cases")
    if not isinstance(cases, dict) or case_id not in cases:
        raise InputError(f"no prerecorded demo output for case {case_id!r} in {path}")
    demo = cases[case_id]
    if not isinstance(demo, dict):
        raise InputError(f"demo output for case {case_id!r} in {path} is not a JSON object")
    return demo


def load_demo_label(case_id: str, path: Path = DEMO_FILE) -> str:
    labels = _load_demo_file(path).get("labels")
    if isinstance(labels, dict) and isinstance(labels.get(case_id), str):
        return labels[case_id]
    return "SYNTHETIC DEMO (hand-written, not model output)."


# --------------------------------------------------------------------------
# Validation
#
# A structurally correct draft is not proof of a correct draft, so this module
# runs two passes: a schema pass, then a content pass against the supplied
# facts and policy.
# --------------------------------------------------------------------------


def parse_model_json(text: str) -> Dict[str, Any]:
    """Parse raw model text into a dict, or raise DraftInvalid."""
    if not isinstance(text, str) or not text.strip():
        raise DraftInvalid("model returned empty text")
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise DraftInvalid(f"model output is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise DraftInvalid("model output JSON is not an object")
    return parsed


def validate_draft(draft: Any, policy_text: str = "") -> Dict[str, Any]:
    """Check the exact shape and types of a draft. Raises DraftInvalid."""
    if not isinstance(draft, dict):
        raise DraftInvalid("draft is not a JSON object")

    keys = set(draft)
    expected = set(REQUIRED_KEYS)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise DraftInvalid(f"wrong key set (missing={missing}, unexpected={extra})")

    reply = draft["draft_reply"]
    if not isinstance(reply, str):
        raise DraftInvalid('"draft_reply" must be a string')
    if not reply.strip():
        raise DraftInvalid('"draft_reply" must not be empty')

    refs = draft["evidence_refs"]
    if not isinstance(refs, list):
        raise DraftInvalid('"evidence_refs" must be a list')
    if any(not isinstance(ref, str) or not ref.strip() for ref in refs):
        raise DraftInvalid('"evidence_refs" must contain only non-empty strings')
    if policy_text:
        known = set(policy_ids(policy_text))
        unknown = sorted({ref for ref in refs if ref not in known})
        if unknown:
            raise DraftInvalid(f'"evidence_refs" cites policy IDs not in the policy: {unknown}')

    status = draft["review_status"]
    if not isinstance(status, str):
        raise DraftInvalid('"review_status" must be a string')
    if status not in ALLOWED_STATUSES:
        raise DraftInvalid(f'"review_status" must be one of {list(ALLOWED_STATUSES)}, got {status!r}')

    return {
        "draft_reply": reply.strip(),
        "evidence_refs": [ref.strip() for ref in refs],
        "review_status": status,
    }


NEGATION_SUBSTRINGS = (
    "not ",
    "n't",
    "cannot",
    "unable",
    "nothing has",
    "never",
    "remains",
    "still open",
    "under review",
    "pending",
    "no action",
    "does not",
    "do not",
    "will not",
    "must not",
)
# Standalone negations need a word boundary: a bare "no " substring test would
# be fine, but "no" as a word has to be matched without catching "know" or
# "another".
NEGATION_WORDS_RE = re.compile(r"\b(?:no|none|nor|neither)\b", re.IGNORECASE)


def _is_negated(sentence: str) -> bool:
    """True when a sentence hedges or denies rather than asserts."""
    lowered = sentence.lower()
    if any(marker in lowered for marker in NEGATION_SUBSTRINGS):
        return True
    return bool(NEGATION_WORDS_RE.search(lowered))

RESOLUTION_CLAIM_PATTERNS = (
    re.compile(r"\b(?:is|was|are|were|has been|have been)\s+(?:now\s+|already\s+)?(?:been\s+)?(?:resolved|closed)\b", re.I),
    re.compile(r"\b(?:resolved|closed|fixed|completed)\s+(?:your|the|this|it)\b", re.I),
    re.compile(r"\bmark(?:ed|ing)?\s+(?:the\s+[\w-]+\s+)?as\s+(?:resolved|closed|complete)\b", re.I),
    re.compile(r"\bwe(?:'ve| have)?\s+(?:now\s+)?(?:resolved|closed|fixed|completed)\b", re.I),
    re.compile(r"\bconfirm(?:ed|ing)?\b[\w\s,'\"]{0,30}\b(?:resolved|closed)\b", re.I),
)

REFUND_CLAIM_PATTERNS = (
    re.compile(r"\brefunds?\s+(?:has\s+|have\s+|was\s+|were\s+|is\s+|are\s+)?(?:been\s+)?(?:approved|granted|processed|issued|released|refunded)\b", re.I),
    re.compile(r"\b(?:approved|granted|processed|issued|released|refunded)\s+(?:your|the|this)\s+refund\b", re.I),
    re.compile(r"\bwe(?:'ve| have)?\s+(?:now\s+)?(?:approved|processed|issued|released|refunded)\b", re.I),
    re.compile(r"\byou(?:'re| are)\s+(?:eligible|entitled)\s+for\s+a\s+refund\b", re.I),
)


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def content_warnings(draft_reply: str, case: Dict[str, Any], policy_text: str = "") -> List[str]:
    """Second pass: catch drafts that are well formed but say something unsafe.

    Two things are refused outright, because doing either would exceed the
    supplied policy: asserting that an open issue is resolved, and asserting
    that a refund was approved or issued.

    Warnings describe the rule that was broken and never quote the offending
    sentence, so a rejected claim is not echoed back into the output.
    """
    warnings: List[str] = []
    status = str(case.get("issue_status", "")).strip().lower()
    resolution_id = _policy_id_for(policy_text, "resolved")
    refund_id = _policy_id_for(policy_text, "approve")

    for sentence in _sentences(draft_reply):
        if _is_negated(sentence):
            continue
        if status == "open":
            for pattern in RESOLUTION_CLAIM_PATTERNS:
                if pattern.search(sentence):
                    warnings.append(
                        "draft states the issue is resolved or closed while the supplied "
                        f"record status is 'open' (conflicts with {resolution_id})"
                    )
                    break
        for pattern in REFUND_CLAIM_PATTERNS:
            if pattern.search(sentence):
                warnings.append(
                    f"draft claims a refund was approved or issued (conflicts with {refund_id})"
                )
                break
    return warnings


ORDER_ID_RE = re.compile(r"\b[A-Z]{2,}-\d+\b")
ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def fact_warnings(draft_reply: str, case: Dict[str, Any]) -> List[str]:
    """Third pass: catch drafts that state a fact the case never supplied.

    A model can hand back perfectly shaped JSON that cites the wrong delivery
    date or a different order id, and a hand-written prerecorded demo can drift
    out of step with cases.json after an edit. Either way the draft would tell
    a human something untrue, so both are refused.
    """
    warnings: List[str] = []

    expected_order = str(case.get("order_id") or "").strip()
    stray_orders = sorted(
        {m for m in ORDER_ID_RE.findall(draft_reply) if m != expected_order}
    )
    if stray_orders:
        warnings.append(
            f"draft cites order ids that are not in the supplied facts: {stray_orders}"
        )

    expected_date = str(case.get("delivery_date") or "").strip()
    cited_dates = sorted(set(ISO_DATE_RE.findall(draft_reply)))
    if expected_date:
        stray_dates = [d for d in cited_dates if d != expected_date]
        if stray_dates:
            warnings.append(
                f"draft states date(s) other than the supplied delivery date "
                f"{expected_date}: {stray_dates}"
            )
    elif cited_dates:
        warnings.append(
            f"the supplied facts have no delivery date, but the draft states {cited_dates}"
        )

    return warnings


def draft_warnings(draft_reply: str, case: Dict[str, Any], policy_text: str = "") -> List[str]:
    """Every content and fact check, in one list."""
    return content_warnings(draft_reply, case, policy_text) + fact_warnings(draft_reply, case)


# --------------------------------------------------------------------------
# Model call
# --------------------------------------------------------------------------


def build_client(api_key: str):
    """Create the OpenCode Zen client with the key passed explicitly.

    The SDK's global key environment variable is deliberately neither
    read nor set, and max_retries is 0 so a failure surfaces immediately
    instead of being retried behind the reviewer's back.
    """
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        base_url=ZEN_BASE_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )


def normalise_model(model: str) -> str:
    """Accept either the raw Zen model id or the opencode/ config form.

    OpenCode's own config uses opencode/big-pickle, but the API wants the raw
    id. Accepting both saves a confusing 404 when someone copies the id from
    their opencode.json.
    """
    cleaned = (model or "").strip()
    if cleaned.startswith("opencode/"):
        cleaned = cleaned[len("opencode/") :]
    return cleaned or DEFAULT_MODEL


def build_prompt(case: Dict[str, Any], policy_text: str) -> str:
    facts = json.dumps(case, indent=2, sort_keys=True)
    return (
        "SUPPLIED CASE FACTS (the only facts you may use):\n"
        f"{facts}\n\n"
        "SUPPLIED POLICY (the only policy you may apply):\n"
        f"{policy_text}\n\n"
        "Return only the JSON object now."
    )


def extract_text(response: Any) -> str:
    """Pull the assistant text out of a chat completions response."""
    choices = getattr(response, "choices", None)
    if not choices:
        raise DraftInvalid("model response contained no choices")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise DraftInvalid("model response choice contained no message")
    text = getattr(message, "content", None)
    if isinstance(text, str) and text.strip():
        return text
    # Reasoning models can return reasoning only, with no final content.
    reasoning = getattr(message, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning.strip():
        raise DraftInvalid(
            "model returned reasoning but no final content; the run was treated as unusable"
        )
    raise DraftInvalid("model response message contained no content")


def make_model_call(api_key: str, model: str, case: Dict[str, Any], policy_text: str) -> Callable[[], str]:
    """Return a zero-argument callable that performs one model call.

    Kept as a factory so tests can inject a fake in its place; production
    callers pass the real key and never touch the SDK directly.
    """
    client = build_client(api_key)

    def call() -> str:
        response = client.chat.completions.create(
            model=normalise_model(model),
            max_tokens=MAX_TOKENS,
            messages=[
                # Big Pickle is documented as not supporting the developer
                # role, so the instructions ride in a system message.
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(case, policy_text)},
            ],
        )
        return extract_text(response)

    return call


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def use_utf8(stream: TextIO) -> None:
    """Make a stream safe for whatever text the model returned.

    The Windows console defaults to cp1252, which cannot encode an emoji or a
    CJK character. Without this, rendering a perfectly valid draft could raise
    UnicodeEncodeError and print a traceback, which is exactly the failure mode
    this tool is meant to avoid. errors="replace" guarantees a substitution
    rather than an exception.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass


def _wrap(text: str, indent: str = "  ", width: int = 78) -> str:
    lines = []
    for block in text.splitlines() or [""]:
        wrapped = textwrap.wrap(block, width=width - len(indent)) or [""]
        lines.extend(indent + line for line in wrapped)
    return "\n".join(lines) if lines else indent


def render_facts(case: Dict[str, Any], out: TextIO) -> None:
    print(FACTS_HEADER, file=out)
    for key, value in case.items():
        print(f"{key}: {'null' if value is None else value}", file=out)
    print(file=out)


def render_policy(policy_text: str, out: TextIO) -> None:
    print(POLICY_HEADER, file=out)
    print(policy_text.rstrip(), file=out)
    print(file=out)


def render_result(result: Dict[str, Any], out: TextIO) -> None:
    print(RESULT_HEADER, file=out)
    print(f"review_status: {result['review_status']}", file=out)
    refs = result.get("evidence_refs") or []
    print(f"evidence_refs: {', '.join(refs) if refs else '(none)'}", file=out)
    print(file=out)
    print("draft_reply:", file=out)
    reply = result.get("draft_reply") or ""
    if reply.strip():
        print(_wrap(reply), file=out)
    else:
        print("  (no draft produced)", file=out)
    if result.get("notice"):
        print(file=out)
        print(_wrap(result["notice"]), file=out)
    if result.get("reason"):
        print(file=out)
        print(f"REASON: {result['reason']}", file=out)
    if result["review_status"] == UNAVAILABLE_STATUS:
        print(file=out)
        print("MANUAL FALLBACK", file=out)
        print(
            _wrap(
                "No draft was produced. Nothing was sent, approved, refunded, or changed. "
                "A human reviewer should read the supplied facts and policy above, confirm any "
                "missing or unverified details with the customer through the normal support "
                "channel, and record the outcome in the case system. Do not treat this run as "
                "an approval of anything."
            ),
            file=out,
        )
    print(file=out)
    print(
        "Reminder: drafts only. This tool cannot send messages, approve refunds, "
        "or change records.",
        file=out,
    )


def unavailable_result(reason: str, notice: Optional[str] = None) -> Dict[str, Any]:
    return {
        "draft_reply": "",
        "evidence_refs": [],
        "review_status": UNAVAILABLE_STATUS,
        "reason": reason,
        "notice": notice,
        "api_called": False,
        "exit_code": EXIT_UNAVAILABLE,
    }


# --------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------


def load_dotenv_quietly() -> None:
    """Load the project-only .env if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise InputError(
            "python-dotenv is not installed. Run: pip install -r requirements.txt"
        ) from exc
    load_dotenv(BASE_DIR / ".env", override=False)


def run(
    case_id: str = DEFAULT_CASE,
    model_fn: Optional[Callable[[], str]] = None,
    simulate_timeout: bool = False,
    offline_demo: bool = False,
    out: Optional[TextIO] = None,
    err: Optional[TextIO] = None,
) -> Dict[str, Any]:
    """Run one case end to end and return the result dict.

    model_fn is the only seam the tests need: a zero-argument callable that
    returns raw model text. Pass one and no SDK client is ever built.
    """
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    use_utf8(out)
    use_utf8(err)

    try:
        policy_text = load_policy(POLICY_FILE)
        case = load_case(case_id, CASES_FILE)
    except InputError as exc:
        print(f"input error: {exc}", file=err)
        return unavailable_result(str(exc))

    render_facts(case, out)
    render_policy(policy_text, out)

    # --simulate-timeout: no API call at all.
    if simulate_timeout:
        result = unavailable_result(
            f"simulated timeout after {REQUEST_TIMEOUT_SECONDS:.0f}s (no API call was made).",
            notice="SIMULATED TIMEOUT - no model API was called.",
        )
        render_result(result, out)
        return result

    # --offline-demo: prerecorded synthetic output, no API call at all.
    if offline_demo:
        try:
            demo = load_demo_output(case_id, DEMO_FILE)
            label = load_demo_label(case_id, DEMO_FILE)
        except InputError as exc:
            result = unavailable_result(str(exc), notice=NO_API_NOTICE)
            render_result(result, out)
            return result
        notice = f"{NO_API_NOTICE}\n{label}"
        try:
            validated = validate_draft(demo, policy_text)
        except DraftInvalid as exc:
            result = unavailable_result(
                f"prerecorded demo output failed validation: {exc}", notice=notice
            )
            render_result(result, out)
            return result
        warnings = draft_warnings(validated["draft_reply"], case, policy_text)
        if warnings:
            result = unavailable_result(
                "prerecorded demo output failed the content check: " + "; ".join(warnings),
                notice=notice,
            )
            render_result(result, out)
            return result
        result = {**validated, "reason": None, "notice": notice, "api_called": False, "exit_code": EXIT_OK}
        render_result(result, out)
        return result

    # Live path: read the application key from the project .env only.
    if model_fn is None:
        try:
            load_dotenv_quietly()
        except InputError as exc:
            result = unavailable_result(str(exc))
            render_result(result, out)
            return result
        api_key = os.environ.get(ENV_KEY, "").strip()
        if not api_key:
            result = unavailable_result(
                f"{ENV_KEY} is not set. Run: python setup_env.py"
            )
            render_result(result, out)
            return result
        model = os.environ.get(ENV_MODEL, "").strip() or DEFAULT_MODEL
        model_fn = make_model_call(api_key, model, case, policy_text)

    try:
        raw = model_fn()
    except DraftInvalid:
        raise
    except Exception as exc:
        result = unavailable_result(f"model provider call failed: {type(exc).__name__}: {exc}")
        render_result(result, out)
        return result

    notice = None
    try:
        draft = parse_model_json(raw)
        validated = validate_draft(draft, policy_text)
    except DraftInvalid as exc:
        result = unavailable_result(f"model output rejected: {exc}")
        render_result(result, out)
        return result

    warnings = draft_warnings(validated["draft_reply"], case, policy_text)
    if warnings:
        result = unavailable_result("model output failed the content check: " + "; ".join(warnings))
        render_result(result, out)
        return result

    result = {**validated, "reason": None, "notice": notice, "api_called": True, "exit_code": EXIT_OK}
    render_result(result, out)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description=(
            "Draft a Meridian Retail support reply for human review. "
            "Cannot send, approve, or delete anything."
        ),
    )
    parser.add_argument(
        "--case",
        choices=CASE_IDS,
        default=DEFAULT_CASE,
        help=f"case id to draft for (default: {DEFAULT_CASE})",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--simulate-timeout",
        action="store_true",
        help="pretend the provider timed out; makes no API call",
    )
    mode.add_argument(
        "--offline-demo",
        action="store_true",
        help="print the prerecorded synthetic demo result; makes no API call",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run(
        case_id=args.case,
        simulate_timeout=args.simulate_timeout,
        offline_demo=args.offline_demo,
    )["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
