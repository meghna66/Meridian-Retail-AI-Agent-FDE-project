# Meridian Retail support-draft review tool

A small local tool that drafts one customer support reply from supplied case
facts and a supplied policy, and prints it **for a human to review**.

## What it does

1. Reads one case from `cases.json` (C1, C2 or C3).
2. Reads `policy.md`.
3. Sends both to the OpenCode Zen API and asks for a JSON draft.
4. Validates the shape, the types, and the content of the draft.
5. Prints the draft with a status a human can act on.

Model: OpenCode Zen, default `space-bunny-free`. See "Which model, and why this
default" below.

## What it cannot do

By design, and enforced by policy P4:

- it never sends a message to a customer
- it never approves a refund
- it never deletes or changes a record

There is no web UI, no database, no RAG pipeline, no login, no agent
framework, and no production integration. It makes one text-generation call
and nothing else. If the provider fails or the output fails validation, it
prints a safe `UNAVAILABLE` state and a manual fallback message rather than a
traceback or a stale draft.

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
python setup_env.py
```

`setup_env.py` asks for the key with hidden input (nothing is echoed), asks for
the model with `space-bunny-free` as the default, and writes a project-only
`.env`. The key value is never printed, not even masked. The file is created
owner-only where the platform allows it and is gitignored.

| Variable | Purpose |
| --- | --- |
| `APP_OPENCODE_API_KEY` | Zen API key, passed explicitly to the client |
| `APP_OPENCODE_MODEL` | model id, default `space-bunny-free` |

Get a key from <https://opencode.ai/auth>. Docs: <https://opencode.ai/docs/zen/>.

The global `OPENCODE_API_KEY` is deliberately never read or set. The client is
built with `base_url=https://opencode.ai/zen/v1`, `timeout=20` and
`max_retries=0`, so a failure surfaces immediately instead of being retried
silently. Requests go to `POST /zen/v1/chat/completions` with
`Authorization: Bearer <key>`. An `opencode/space-bunny-free` value is accepted
and normalised to the raw `space-bunny-free` API id, since that is the form
that appears in `opencode.json`.

## Which model, and why this default

Zen's free models are **not** equally available. Measured against a real
free-tier key, direct API call, no retries:

| Model | Result |
| --- | --- |
| `space-bunny-free` | **works** |
| `big-pickle` | `403 FreeTierError` - "can only be used from within OpenCode" |
| `mimo-v2.5-free` | `403 FreeTierError` |
| `nemotron-3-ultra-free` | `403 FreeTierError` |
| `kimi-k3` (paid) | `402 Insufficient account funds` |

So `space-bunny-free` is the default: it is the one model that answers direct
API calls for free. `big-pickle` and the other free models are reachable only
from inside the OpenCode CLI/TUI with a free-tier key, and paid models need
funds on the account. To use `big-pickle` instead, set
`APP_OPENCODE_MODEL=big-pickle` in `.env` with a billed Zen key.

Caveats worth knowing before you rely on this:

- Every model in that free list is described by OpenCode as available "for a
  limited time" and several are stealth models. They can change or disappear.
- `space-bunny-free` is a coding-tuned model, not a support-writing one. The
  drafts it produced for C1-C3 were policy-compliant, but do not expect the
  polish of a support-tuned model. That is the trade for running free.
- Run `--offline-demo` when you need a reproducible, provider-independent
  result for review or demo purposes.

## Run

```bash
python app.py --case C1                      # live call
python app.py --case C2 --offline-demo       # prerecorded synthetic result, no API call
python app.py --case C3 --simulate-timeout   # force the safe unavailable state, no API call
```

Output has three labelled sections:

```
=== SUPPLIED FACTS ===
=== SUPPLIED POLICY ===
=== DRAFT RESULT FOR HUMAN REVIEW ===
```

`--simulate-timeout` and `--offline-demo` are mutually exclusive and neither
makes a network call.

### Review statuses

| Status | Meaning |
| --- | --- |
| `READY_FOR_HUMAN_REVIEW` | a draft exists and a human should read it |
| `NEEDS_INFORMATION` | a fact the policy requires is missing |
| `BLOCKED` | must not be answered without a human decision |
| `UNAVAILABLE` | no draft. No API call, a failed call, a timeout, or rejected output |

### Output contract

The model is asked for one JSON object with exactly three keys:

```json
{
  "draft_reply": "string, the drafted text",
  "evidence_refs": ["P1", "P2"],
  "review_status": "READY_FOR_HUMAN_REVIEW"
}
```

## Validation is two passes

A structurally correct draft is not proof of a correct draft, so `app.py`
checks twice before displaying anything:

1. **Schema.** exactly the three keys above, `draft_reply` a non-empty string,
   `evidence_refs` a list of non-empty strings, `review_status` one of the three
   allowed values, and every cited policy ID actually present in `policy.md`.
2. **Content.** a draft is refused outright if it asserts that an open issue is
   resolved, or asserts that a refund was approved or issued. Claims that are
   clearly negated ("I cannot confirm that your issue is resolved") are allowed,
   because refusing those would break C3.

A draft that fails either pass is never displayed. The run degrades to
`UNAVAILABLE` with the reason and a manual fallback.

## Tests

```bash
.venv/Scripts/python -m pytest -q
```

No test makes a network call. Live-path tests inject a fake model function,
which is the single seam `run()` needs. The suite covers a successful C1
result, twelve kinds of invalid model output, provider failure, timeout
behaviour, the input files, the offline demo outputs, the content guards, the
client configuration, non-ASCII model output, and that `setup_env.py` writes
the key without printing it. The real OpenCode client is also driven through an
`httpx2` mock transport, which asserts the exact Zen URL, the `Bearer` header,
the model id, and the `[system, user]` message roles without opening a socket.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | the tool |
| `policy.md` | supplied policy (input) |
| `cases.json` | supplied cases C1-C3 (input) |
| `demo_outputs.json` | prerecorded synthetic results for `--offline-demo` |
| `setup_env.py` | writes the project-only `.env` |
| `.env.example` | placeholders only |
| `test_app.py`, `test_setup_env.py` | pytest suite |
