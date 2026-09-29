"""Tests for app.py.

No test in this file makes a network call. Every live-path test injects a fake
model function, which is the only seam run() needs: a zero-argument callable
returning raw model text.
"""

from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import app

BASE_DIR = Path(app.__file__).resolve().parent

# Built from the real case so the fixture cannot drift out of step with
# cases.json. The demo-output test below is what guards demo_outputs.json.
_C1 = app.load_case("C1")
_C1_DATE = _C1["delivery_date"]

GOOD_C1_TEXT = json.dumps(
    {
        "draft_reply": (
            "Thanks for reporting the broken zip on order MR-1042. The jacket was "
            f"delivered on {_C1_DATE} and the issue is logged as open. Damage reported "
            "within two days of delivery can be referred for a human return review, so "
            "a specialist will assess the next step. This draft is not a refund "
            "decision: no refund has been approved and your order record has not been "
            "changed. A human colleague will follow up with the outcome."
        ),
        "evidence_refs": ["P1", "P2", "P4"],
        "review_status": "READY_FOR_HUMAN_REVIEW",
    }
)


class FakeModel:
    """Records whether it was called and returns canned raw text."""

    def __init__(self, text=None, exc=None):
        self.text = text
        self.exc = exc
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.text


def run_with(model, **kwargs):
    out, err = io.StringIO(), io.StringIO()
    result = app.run(model_fn=model, out=out, err=err, **kwargs)
    return result, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------
# 1. Successful C1 result through the fake model
# --------------------------------------------------------------------------


def test_successful_c1_draft_is_displayed():
    model = FakeModel(GOOD_C1_TEXT)
    result, out, err = run_with(model, case_id="C1")

    assert result["review_status"] == "READY_FOR_HUMAN_REVIEW"
    assert result["evidence_refs"] == ["P1", "P2", "P4"]
    assert "MR-1042" in result["draft_reply"]
    assert result["exit_code"] == app.EXIT_OK
    assert model.calls == 1

    for header in (app.FACTS_HEADER, app.POLICY_HEADER, app.RESULT_HEADER):
        assert header in out
    assert "broken zip" in out
    assert "P1." in out
    assert "READY_FOR_HUMAN_REVIEW" in out
    assert "Traceback" not in out
    assert err == ""


def test_each_case_id_is_accepted():
    for case_id in app.CASE_IDS:
        model = FakeModel(good_text_for(case_id))
        result, out, _ = run_with(model, case_id=case_id)
        assert result["review_status"] != app.UNAVAILABLE_STATUS, (case_id, result["reason"])
        assert f"case_id: {case_id}" in out


def test_parser_accepts_c1_c2_c3_and_rejects_others():
    parser = app.build_parser()
    for case_id in app.CASE_IDS:
        assert parser.parse_args(["--case", case_id]).case == case_id
    with pytest.raises(SystemExit):
        parser.parse_args(["--case", "C9"])


def test_flags_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        app.build_parser().parse_args(["--simulate-timeout", "--offline-demo"])


# --------------------------------------------------------------------------
# 2. Invalid model output
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,text",
    [
        ("not_json", "Sorry, I cannot help with that."),
        ("not_an_object", json.dumps([{"review_status": "BLOCKED"}])),
        ("missing_key", json.dumps({"draft_reply": "hi", "review_status": "BLOCKED"})),
        (
            "extra_key",
            json.dumps(
                {
                    "draft_reply": "hi",
                    "evidence_refs": ["P1"],
                    "review_status": "BLOCKED",
                    "refund_approved": True,
                }
            ),
        ),
        (
            "refs_not_a_list",
            json.dumps({"draft_reply": "hi", "evidence_refs": "P1", "review_status": "BLOCKED"}),
        ),
        (
            "refs_wrong_item_type",
            json.dumps({"draft_reply": "hi", "evidence_refs": [1], "review_status": "BLOCKED"}),
        ),
        (
            "reply_wrong_type",
            json.dumps({"draft_reply": 42, "evidence_refs": ["P1"], "review_status": "BLOCKED"}),
        ),
        (
            "reply_empty",
            json.dumps({"draft_reply": "   ", "evidence_refs": ["P1"], "review_status": "BLOCKED"}),
        ),
        (
            "bad_status",
            json.dumps({"draft_reply": "hi", "evidence_refs": ["P1"], "review_status": "APPROVED"}),
        ),
        (
            "unavailable_status_rejected",
            json.dumps({"draft_reply": "hi", "evidence_refs": ["P1"], "review_status": "UNAVAILABLE"}),
        ),
        (
            "fabricated_policy_ref",
            json.dumps({"draft_reply": "hi", "evidence_refs": ["P99"], "review_status": "BLOCKED"}),
        ),
        ("empty_text", "   "),
    ],
)
def test_invalid_model_output_becomes_safe_unavailable_state(label, text):
    result, out, err = run_with(FakeModel(text), case_id="C1")

    assert result["review_status"] == app.UNAVAILABLE_STATUS, label
    assert result["draft_reply"] == "", label
    assert result["evidence_refs"] == [], label
    assert result["exit_code"] == app.EXIT_UNAVAILABLE, label
    assert "model output rejected" in result["reason"], label
    assert "MANUAL FALLBACK" in out, label
    assert "Traceback" not in out and "Traceback" not in err, label
    # the rejected draft must not be shown anywhere
    assert "ready for human" not in out.lower(), label


def test_provider_failure_becomes_safe_unavailable_state_without_traceback():
    boom = FakeModel(exc=RuntimeError("connection reset by peer"))
    result, out, err = run_with(boom, case_id="C1")

    assert result["review_status"] == app.UNAVAILABLE_STATUS
    assert result["draft_reply"] == ""
    assert "model provider call failed" in result["reason"]
    assert "RuntimeError" in result["reason"]
    assert "MANUAL FALLBACK" in out
    assert "Traceback" not in out
    assert "Traceback" not in err
    assert result["exit_code"] == app.EXIT_UNAVAILABLE


def test_no_stale_draft_is_shown_after_a_failure():
    # first a good run, then a failing run into the same output stream
    out = io.StringIO()
    app.run(case_id="C1", model_fn=FakeModel(GOOD_C1_TEXT), out=out, err=io.StringIO())
    out2 = io.StringIO()
    result = app.run(
        case_id="C1", model_fn=FakeModel(exc=TimeoutError("timed out")), out=out2, err=io.StringIO()
    )
    assert result["review_status"] == app.UNAVAILABLE_STATUS
    assert "specialist will assess" not in out2.getvalue()


# --------------------------------------------------------------------------
# 3. --simulate-timeout
# --------------------------------------------------------------------------


def test_simulate_timeout_makes_no_api_call():
    model = FakeModel(GOOD_C1_TEXT)
    result, out, _ = run_with(model, case_id="C2", simulate_timeout=True)

    assert model.calls == 0, "simulate-timeout must not call the model"
    assert result["api_called"] is False
    assert result["review_status"] == app.UNAVAILABLE_STATUS
    assert result["draft_reply"] == ""
    assert result["evidence_refs"] == []
    assert "simulated timeout" in result["reason"].lower()
    assert "MANUAL FALLBACK" in out
    assert app.RESULT_HEADER in out
    assert result["exit_code"] == app.EXIT_UNAVAILABLE


def test_simulate_timeout_clears_draft_and_shows_fallback_message():
    result, out, _ = run_with(FakeModel(), case_id="C1", simulate_timeout=True)
    assert "draft_reply:" in out
    assert "(no draft produced)" in out
    assert "no draft was produced" in out.lower()


def test_simulate_timeout_does_not_need_a_key(monkeypatch):
    monkeypatch.delenv(app.ENV_KEY, raising=False)
    out, err = io.StringIO(), io.StringIO()
    result = app.run(case_id="C1", simulate_timeout=True, out=out, err=err)
    assert "is not set" not in (result.get("reason") or "")
    assert result["review_status"] == app.UNAVAILABLE_STATUS


# --------------------------------------------------------------------------
# 4. All input files
# --------------------------------------------------------------------------


def test_all_input_files_exist():
    for name in ("app.py", "cases.json", "policy.md", "demo_outputs.json", "requirements.txt"):
        assert (BASE_DIR / name).is_file(), f"missing input file: {name}"


def test_cases_and_policy_load_and_contain_expected_ids():
    for case_id in app.CASE_IDS:
        case = app.load_case(case_id)
        assert case["case_id"] == case_id
        for field in ("order_id", "product", "customer_question", "issue_status"):
            assert field in case
    assert "delivery_date" in app.load_case("C2")
    assert app.load_case("C2")["delivery_date"] is None
    assert app.load_case("C1")["issue_status"] == "open"

    policy = app.load_policy()
    assert app.policy_ids(policy) == ["P1", "P2", "P3", "P4"]


def test_unknown_case_id_raises_input_error():
    with pytest.raises(app.InputError):
        app.load_case("C9")


def test_missing_input_file_reports_cleanly(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "POLICY_FILE", tmp_path / "absent.md")
    out, err = io.StringIO(), io.StringIO()
    result = app.run(case_id="C1", model_fn=FakeModel(), out=out, err=err)
    assert result["review_status"] == app.UNAVAILABLE_STATUS
    assert "cannot read policy file" in result["reason"]
    assert "Traceback" not in err.getvalue()
    assert "input error" in err.getvalue()


# --------------------------------------------------------------------------
# 5. Offline demo outputs
# --------------------------------------------------------------------------


def test_offline_demo_makes_no_api_call_and_prints_no_api_line():
    model = FakeModel(GOOD_C1_TEXT)
    result, out, _ = run_with(model, case_id="C1", offline_demo=True)

    assert model.calls == 0, "offline-demo must not call the model"
    assert result["api_called"] is False
    assert "NO MODEL API WAS CALLED" in out
    assert result["review_status"] == "READY_FOR_HUMAN_REVIEW"
    assert result["evidence_refs"] == ["P1", "P2", "P4"]
    assert result["exit_code"] == app.EXIT_OK
    assert "SYNTHETIC" in out


def test_offline_demo_works_for_every_case():
    for case_id, expected_status in (
        ("C1", "READY_FOR_HUMAN_REVIEW"),
        ("C2", "NEEDS_INFORMATION"),
        ("C3", "BLOCKED"),
    ):
        result, out, _ = run_with(FakeModel(), case_id=case_id, offline_demo=True)
        assert result["review_status"] == expected_status, case_id
        assert result["draft_reply"].strip()
        assert "NO MODEL API WAS CALLED" in out
        assert result["api_called"] is False


def test_offline_outputs_cover_every_case_and_pass_validation():
    policy = app.load_policy()
    data = json.loads((BASE_DIR / "demo_outputs.json").read_text(encoding="utf-8"))
    assert set(data["cases"]) == set(app.CASE_IDS)
    for case_id in app.CASE_IDS:
        demo = app.load_demo_output(case_id)
        assert set(demo) == set(app.REQUIRED_KEYS), case_id
        validated = app.validate_draft(demo, policy)
        assert validated["review_status"] in app.ALLOWED_STATUSES
        case = app.load_case(case_id)
        # draft_warnings, not just content_warnings: this is the guard that
        # catches demo_outputs.json drifting away from cases.json.
        assert app.draft_warnings(validated["draft_reply"], case, policy) == [], case_id
        assert app.load_demo_label(case_id).startswith("SYNTHETIC")


def test_demo_outputs_never_contradict_the_supplied_cases():
    """Guards against an edit to cases.json silently invalidating the demos."""
    policy = app.load_policy()
    for case_id in app.CASE_IDS:
        case = app.load_case(case_id)
        reply = app.load_demo_output(case_id)["draft_reply"]
        assert app.fact_warnings(reply, case) == [], case_id


# --------------------------------------------------------------------------
# 6b. Fabricated facts, not just unsafe claims
# --------------------------------------------------------------------------


def test_draft_with_wrong_delivery_date_is_rejected():
    wrong = "Your jacket on order MR-1042 was delivered on 2020-01-01 and can be returned."
    result, out, _ = run_with(FakeModel(_text(wrong)), case_id="C1")
    assert result["review_status"] == app.UNAVAILABLE_STATUS
    assert "delivery date" in result["reason"]
    # the draft itself is never shown; REASON names the stray date as a
    # diagnostic, which is the point of that line
    assert "can be returned" not in out
    assert "MANUAL FALLBACK" in out


def test_draft_citing_another_order_id_is_rejected():
    wrong = "Order MR-9999 is eligible for a return review."
    result, _, _ = run_with(FakeModel(_text(wrong)), case_id="C1")
    assert result["review_status"] == app.UNAVAILABLE_STATUS
    assert "order ids" in result["reason"]


def test_draft_inventing_a_date_when_none_supplied_is_rejected():
    # C2 has no delivery date, so any ISO date is an invention
    invented = "Our records show the jacket arrived on 2026-09-20."
    result, _, _ = run_with(FakeModel(_text(invented)), case_id="C2")
    assert result["review_status"] == app.UNAVAILABLE_STATUS
    assert "no delivery date" in result["reason"]


def test_fact_warnings_accept_relative_wording():
    reply = "Thanks for reporting the broken zip; the jacket arrived yesterday."
    assert app.fact_warnings(reply, app.load_case("C1")) == []


def test_offline_demo_does_not_need_a_key(monkeypatch):
    monkeypatch.delenv(app.ENV_KEY, raising=False)
    out, err = io.StringIO(), io.StringIO()
    result = app.run(case_id="C3", offline_demo=True, out=out, err=err)
    assert result["review_status"] == "BLOCKED"
    assert "is not set" not in (result.get("reason") or "")


# --------------------------------------------------------------------------
# 6. Content safety beyond the schema
# --------------------------------------------------------------------------


def _text(reply, status="READY_FOR_HUMAN_REVIEW"):
    return json.dumps(
        {"draft_reply": reply, "evidence_refs": ["P1"], "review_status": status}
    )


def good_text_for(case_id):
    """A draft that is valid for that specific case.

    Must be per-case: the fact check rejects a draft that cites a delivery date
    the case does not supply, so one canned text cannot serve all three.
    """
    case = app.load_case(case_id)
    order = case["order_id"]
    date = case["delivery_date"]
    if date:
        reply = (
            f"Thanks for reporting the broken zip on order {order}. Our records show "
            f"the jacket was delivered on {date} and the issue is logged as open. Damage "
            "reported within two days of delivery can be referred for a human return "
            "review, so a specialist will assess the next step. No refund has been approved "
            "and no record has been changed by this draft."
        )
    else:
        reply = (
            f"Thanks for letting us know about the broken zip on order {order}. Our records "
            "do not include a delivery date for this order, so could you confirm when it "
            "arrived? A human colleague can then review whether to refer it. Nothing has been "
            "approved and no record has been changed by this draft."
        )
    return json.dumps(
        {
            "draft_reply": reply,
            "evidence_refs": ["P1", "P2", "P4"],
            "review_status": "READY_FOR_HUMAN_REVIEW",
        }
    )


def test_open_issue_draft_claiming_resolution_is_rejected():
    result, out, _ = run_with(
        FakeModel(_text("Good news, your issue is now resolved.")), case_id="C1"
    )
    assert result["review_status"] == app.UNAVAILABLE_STATUS
    assert "content check" in result["reason"]
    assert "resolved" in result["reason"]
    assert "now resolved" not in out
    assert "MANUAL FALLBACK" in out


def test_draft_claiming_approved_refund_is_rejected():
    result, _, _ = run_with(
        FakeModel(_text("Your refund has been approved and the money is on its way.")),
        case_id="C1",
    )
    assert result["review_status"] == app.UNAVAILABLE_STATUS
    assert "refund" in result["reason"]


def test_negated_resolution_claim_is_allowed():
    # C3 legitimately has to say it cannot confirm resolution
    reply = (
        "I cannot confirm that your issue is resolved. Order MR-1126 is still open "
        "and the replacement request remains under review."
    )
    result, _, _ = run_with(FakeModel(_text(reply, "BLOCKED")), case_id="C3")
    assert result["review_status"] == "BLOCKED"


@pytest.mark.parametrize(
    "reply",
    [
        "No refund has been approved and no record has been changed.",
        "None of this has been approved or refunded.",
        "Your refund has not been approved.",
        "This is not a refund approval.",
        "I am unable to approve a refund.",
    ],
)
def test_negated_refund_claims_are_allowed(reply):
    assert app.content_warnings(reply, app.load_case("C1"), app.load_policy()) == []


def test_negation_word_matching_does_not_catch_substrings():
    # "no" must not be found inside know/another/cannot
    reply = "We know another option: your refund has been approved."
    warnings = app.content_warnings(reply, app.load_case("C1"), app.load_policy())
    assert any("refund" in w for w in warnings)


def test_bare_refund_claim_is_still_rejected():
    reply = "Your refund has been approved."
    warnings = app.content_warnings(reply, app.load_case("C1"), app.load_policy())
    assert any("refund" in w for w in warnings)


def test_content_warnings_use_policy_ids_not_hardcoded_numbers():
    policy = "P7. An open issue must not be reported as resolved.\n"
    warnings = app.content_warnings("The issue is resolved.", app.load_case("C1"), policy)
    assert warnings and "P7" in warnings[0]

    refunds = "P8. Support cannot approve refunds.\n"
    warnings = app.content_warnings("Your refund has been approved.", app.load_case("C1"), refunds)
    assert warnings and "P8" in warnings[0]


def test_content_warning_never_quotes_the_rejected_claim():
    policy = app.load_policy()
    bad = "Your issue is now resolved and the refund has been approved."
    warnings = app.content_warnings(bad, app.load_case("C1"), policy)
    joined = " ".join(warnings)
    assert "now resolved" not in joined
    assert "refund has been approved" not in joined


# --------------------------------------------------------------------------
# 7. Client configuration and key handling
# --------------------------------------------------------------------------


def test_client_is_built_with_explicit_key_base_url_timeout_and_no_retries():
    pytest.importorskip("openai")
    client = app.build_client("sk-zen-fake-not-real")
    assert client.api_key == "sk-zen-fake-not-real"
    assert str(client.base_url).rstrip("/") == app.ZEN_BASE_URL
    assert client.timeout == 20.0
    assert client.max_retries == 0


def test_zen_base_url_is_the_documented_endpoint():
    assert app.ZEN_BASE_URL == "https://opencode.ai/zen/v1"


def test_default_model_is_space_bunny_free():
    assert app.DEFAULT_MODEL == "space-bunny-free"


def test_opencode_config_model_id_is_normalised():
    assert app.normalise_model("opencode/space-bunny-free") == "space-bunny-free"
    assert app.normalise_model("opencode/big-pickle") == "big-pickle"
    assert app.normalise_model("  space-bunny-free ") == "space-bunny-free"
    assert app.normalise_model("") == "space-bunny-free"


def test_extract_text_reads_chat_completion_content():
    message = SimpleNamespace(content='{"draft_reply": "x"}')
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    assert app.extract_text(response) == '{"draft_reply": "x"}'


def test_extract_text_rejects_reasoning_only_response():
    message = SimpleNamespace(content="", reasoning_content="thinking hard")
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    with pytest.raises(app.DraftInvalid) as exc:
        app.extract_text(response)
    assert "reasoning" in str(exc.value)


def test_extract_text_rejects_empty_choices():
    with pytest.raises(app.DraftInvalid):
        app.extract_text(SimpleNamespace(choices=[]))


def test_app_source_never_touches_the_global_key(monkeypatch):
    source = (BASE_DIR / "app.py").read_text(encoding="utf-8")
    for line in source.splitlines():
        stripped = line.strip()
        if "OPENCODE_API_KEY" not in stripped:
            continue
        assert "APP_OPENCODE_API_KEY" in stripped, f"global key referenced: {stripped}"
    assert not re.search(r"(?<!APP_)OPENCODE_API_KEY", source)


def test_run_does_not_read_the_global_key_or_set_env(monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-global-should-be-ignored")
    monkeypatch.setenv(app.ENV_KEY, "sk-app-fake")
    monkeypatch.setattr(app, "load_dotenv_quietly", lambda: None)
    seen = {}

    def spy(api_key, model, case, policy):
        seen["key"] = api_key
        seen["model"] = model
        return FakeModel(GOOD_C1_TEXT)

    monkeypatch.setattr(app, "make_model_call", spy)
    out = io.StringIO()
    result = app.run(case_id="C1", out=out, err=io.StringIO())

    assert result["review_status"] == "READY_FOR_HUMAN_REVIEW"
    assert seen["key"] == "sk-app-fake"
    assert seen["key"] != "sk-global-should-be-ignored"
    assert os.environ.get("OPENCODE_API_KEY") == "sk-global-should-be-ignored"


def test_missing_key_reports_setup_instructions(monkeypatch):
    monkeypatch.delenv(app.ENV_KEY, raising=False)
    monkeypatch.setattr(app, "load_dotenv_quietly", lambda: None)
    out, err = io.StringIO(), io.StringIO()
    result = app.run(case_id="C1", out=out, err=err)
    assert result["review_status"] == app.UNAVAILABLE_STATUS
    assert "setup_env.py" in result["reason"]
    assert "Traceback" not in err.getvalue()


def test_model_defaults_to_space_bunny_free(monkeypatch):
    monkeypatch.delenv(app.ENV_MODEL, raising=False)
    monkeypatch.setenv(app.ENV_KEY, "sk-app-fake")
    monkeypatch.setattr(app, "load_dotenv_quietly", lambda: None)
    seen = {}

    def spy(api_key, model, case, policy):
        seen["model"] = model
        return FakeModel(GOOD_C1_TEXT)

    monkeypatch.setattr(app, "make_model_call", spy)
    app.run(case_id="C1", out=io.StringIO(), err=io.StringIO())
    assert seen["model"] == "space-bunny-free"


def test_prompt_contains_facts_policy_and_json_instruction():
    case = app.load_case("C1")
    prompt = app.build_prompt(case, app.load_policy())
    assert "MR-1042" in prompt
    assert "P1." in prompt
    assert "JSON" in prompt
    for status in app.ALLOWED_STATUSES:
        assert status in app.SYSTEM_PROMPT
    assert "cannot send" in app.SYSTEM_PROMPT.lower()


def test_app_has_no_write_or_send_capabilities():
    source = (BASE_DIR / "app.py").read_text(encoding="utf-8").lower()
    for forbidden in (
        "import requests",
        "import urllib",
        "http.client",
        "smtplib",
        "boto3",
        "sqlalchemy",
        "sqlite3",
        "fastapi",
        "flask",
        "streamlit",
        "input(",
    ):
        assert forbidden not in source, f"unexpected capability: {forbidden}"


def test_env_example_has_placeholders_only():
    text = (BASE_DIR / ".env.example").read_text(encoding="utf-8")
    assert f"{app.ENV_KEY}=your-zen-api-key-here" in text
    assert app.DEFAULT_MODEL in text
    for line in text.splitlines():
        if line.strip().startswith(f"{app.ENV_KEY}="):
            value = line.split("=", 1)[1].strip()
            assert "your-" in value and "sk-" not in value


def test_gitignore_excludes_env_and_caches():
    text = (BASE_DIR / ".gitignore").read_text(encoding="utf-8")
    for entry in (".env", "__pycache__/", ".venv/", ".pyenv/", ".pytest_cache/"):
        assert entry in text, f"missing gitignore entry: {entry}"
    assert "!.env.example" in text


# --------------------------------------------------------------------------
# 9. Non-ASCII model output must not crash the renderer
# --------------------------------------------------------------------------


@pytest.mark.parametrize("snippet", ["jacket\u2019s zip", "refund \u2014 approved?", "\U0001f600", "\u4e2d\u6587"])
def test_non_ascii_draft_renders_without_unicode_error(snippet):
    reply = f"Thanks for the report on order MR-1042. {snippet} A colleague will review."
    result, out, _ = run_with(FakeModel(_text(reply)), case_id="C1")
    assert result["review_status"] == "READY_FOR_HUMAN_REVIEW"
    assert "Traceback" not in out
    assert "UnicodeEncodeError" not in out


def test_use_utf8_is_safe_on_streams_without_reconfigure():
    app.use_utf8(io.StringIO())  # must not raise


def test_run_reconfigures_the_real_stdout_to_utf8():
    buffer = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    app.use_utf8(buffer)
    assert buffer.encoding.lower().replace("-", "") == "utf8"
    assert buffer.errors == "replace"


# --------------------------------------------------------------------------
# 8. Wire format, checked without a network call
#
# The real OpenCode client is driven through a mock transport, so the request
# URL, auth header, and payload are asserted exactly as Zen would receive them.
# --------------------------------------------------------------------------


def _zen_client(response_json, status=200, capture=None):
    import httpx2
    from openai import OpenAI

    def handler(request):
        if capture is not None:
            capture["url"] = str(request.url)
            capture["auth"] = request.headers.get("authorization")
            capture["body"] = json.loads(request.content)
        return httpx2.Response(status, json=response_json)

    return OpenAI(
        api_key="sk-zen-fake",
        base_url=app.ZEN_BASE_URL,
        timeout=app.REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )


def _completion(content):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "big-pickle",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }


def test_live_call_targets_the_documented_zen_endpoint(monkeypatch):
    capture = {}
    client = _zen_client(_completion(GOOD_C1_TEXT), capture=capture)
    monkeypatch.setattr(app, "build_client", lambda api_key: client)

    call = app.make_model_call("sk-zen-fake", "big-pickle", app.load_case("C1"), app.load_policy())
    text = call()

    assert capture["url"] == "https://opencode.ai/zen/v1/chat/completions"
    assert capture["auth"] == "Bearer sk-zen-fake"
    assert capture["body"]["model"] == "big-pickle"
    assert [m["role"] for m in capture["body"]["messages"]] == ["system", "user"]
    assert json.loads(text)["review_status"] == "READY_FOR_HUMAN_REVIEW"


def test_live_call_uses_the_zen_model_and_validates(monkeypatch):
    monkeypatch.setenv(app.ENV_KEY, "sk-zen-fake")
    monkeypatch.setenv(app.ENV_MODEL, "opencode/space-bunny-free")
    monkeypatch.setattr(app, "load_dotenv_quietly", lambda: None)
    capture = {}
    monkeypatch.setattr(
        app, "build_client", lambda api_key: _zen_client(_completion(GOOD_C1_TEXT), capture=capture)
    )

    out = io.StringIO()
    result = app.run(case_id="C1", out=out, err=io.StringIO())

    assert result["review_status"] == "READY_FOR_HUMAN_REVIEW"
    assert result["api_called"] is True
    # the opencode/ config form is normalised to the raw API id
    assert capture["body"]["model"] == "space-bunny-free"


def test_live_call_provider_error_becomes_safe_unavailable_state(monkeypatch):
    monkeypatch.setenv(app.ENV_KEY, "sk-zen-fake")
    monkeypatch.setattr(app, "load_dotenv_quietly", lambda: None)
    monkeypatch.setattr(
        app,
        "build_client",
        lambda api_key: _zen_client({"error": {"message": "invalid key"}}, status=401),
    )

    out, err = io.StringIO(), io.StringIO()
    result = app.run(case_id="C1", out=out, err=err)

    assert result["review_status"] == app.UNAVAILABLE_STATUS
    assert result["draft_reply"] == ""
    assert "model provider call failed" in result["reason"]
    assert "Traceback" not in out.getvalue()
    assert "Traceback" not in err.getvalue()
    assert "MANUAL FALLBACK" in out.getvalue()


def test_live_call_rejects_unparseable_completion(monkeypatch):
    monkeypatch.setenv(app.ENV_KEY, "sk-zen-fake")
    monkeypatch.setattr(app, "load_dotenv_quietly", lambda: None)
    monkeypatch.setattr(
        app,
        "build_client",
        lambda api_key: _zen_client(_completion("I would rather not answer in JSON.")),
    )

    out = io.StringIO()
    result = app.run(case_id="C1", out=out, err=io.StringIO())
    assert result["review_status"] == app.UNAVAILABLE_STATUS
    assert "model output rejected" in result["reason"]
