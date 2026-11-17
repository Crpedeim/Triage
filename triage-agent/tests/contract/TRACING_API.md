# Tracing API: `POST /api/v1/traces`

Send one LLM call, or one multi-step / multi-agent run, to the platform. The trace is stored,
its cost is computed server-side, and one evaluation job per matching evaluator is queued in
the same database transaction. The worker then runs the evaluators in the background.

This document is the contract. It is written from, and must stay in sync with:
`src/lib/trace-input.ts` (request schema and cost rules) and
`src/app/api/v1/traces/route.ts` (auth, status codes, response).

## Endpoint

```
POST {BASE_URL}/api/v1/traces
Content-Type: application/json
Authorization: Bearer <project API key>
```

`BASE_URL` is the deployment's origin, for example the live URL in the README.

## Authentication

Every request needs a project API key in the `Authorization` header, as `Bearer <key>`.

- Keys look like `evp_live_` followed by 32 base64url characters (24 random bytes).
- A key belongs to exactly one project; the trace is stored in that project. There is no
  project id in the body.
- Only a SHA-256 hash of the key is stored, so a lost key cannot be shown again; create a
  new one on the project page (**API keys**).
- A missing header, a header not starting with `Bearer `, or an unknown key all return
  `401 {"error":"invalid api key"}`.

## Request body

A JSON object. Unknown fields are ignored (dropped, not stored). In particular a
client-sent `costUsd`, on the trace or on a span, is ignored: cost is always computed by
the server.

### Trace fields

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `name` | string, ≥1 char | **yes** | | Operation name, e.g. `"triage.run"`. Evaluators can be scoped to one trace name. |
| `input` | string | **yes** | | Prompt or input text. May be `""`. |
| `output` | string | **yes** | | Final output text. May be `""`. |
| `model` | string, ≥1 char | **yes** | | Model of the call. With `spans`, use the main / final model; it is stored but does not set the cost (see Cost). |
| `promptTokens` | integer ≥ 0 | no | `0` | |
| `completionTokens` | integer ≥ 0 | no | `0` | Include hidden reasoning ("thinking") tokens, since providers bill them as output. |
| `latencyMs` | integer ≥ 0 | no | `null` | End-to-end wall-clock time. |
| `status` | string | no | `"ok"` | Free-form, e.g. `"ok"` or `"error"`. |
| `metadata` | object (string keys, any JSON values) | no | `null` | Stored as JSON. Must be an object, not an array or scalar. |
| `spans` | array of Span, at most 500 | no | none | Steps inside the run. See below. |

### Span fields

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `id` | string, 1–128 chars | no | none | Your id for this span, unique within the request. Needed only if another span names it as `parentId`. |
| `parentId` | string, 1–128 chars | no | none | `id` of the parent span **in the same request**. Omit for a top-level span. |
| `name` | string, 1–200 chars | **yes** | | e.g. `"orchestrator"`, `"planner"`, `"kb_search"`. |
| `kind` | string, 1–32 chars | no | none | Free-form category, e.g. `"agent"`, `"llm"`, `"tool"`, `"retrieval"`. Shown in the UI. |
| `status` | `"ok"` or `"error"` | no | `"ok"` | Any other value is a 422. |
| `model` | string, ≥1 char | no | none | Model this span called. A span without `model` costs 0. |
| `startedAt` | ISO 8601 string or epoch milliseconds | no | time the request is received | Used to draw the waterfall. Send it for a meaningful timeline. |
| `latencyMs` | integer ≥ 0 | **yes** | | Duration of this span. |
| `promptTokens` | integer ≥ 0 | no | `0` | This span's **own** model calls only, not its children's. |
| `completionTokens` | integer ≥ 0 | no | `0` | Same; include thinking tokens. |
| `metadata` | object | no | `null` | Anything else: agent role, tool arguments, error message, etc. |

Span rules, checked after the field types above pass:

- `id` values must be unique within the request (`duplicate span id "<id>"`).
- `parentId` must equal the `id` of a span in the same request
  (`parentId "<x>" is not the id of a span in this request`).
- Parents must not form a cycle, including a span naming itself (`span parents form a cycle`).
- Spans may be listed in any order; a parent does not have to come before its children.
- Nesting depth is not limited.

### Multi-agent runs

One run = one trace. Model each agent as a span with `kind: "agent"`, and each model call,
tool call or retrieval inside it as a child span (`parentId` = the agent's `id`), with its
own `model`, tokens, `latencyMs` and `status`. Agents can nest (an orchestrator span whose
children are sub-agent spans). Per-span cost is computed from each span's own model and
tokens, so put tokens on the leaf spans that made the calls and leave them at 0 on agent
spans that only coordinate, otherwise the same tokens are counted twice.

## Cost

Computed on the server from token counts and the price table in `src/lib/pricing.ts`
(USD per 1M tokens, input and output). Never accepted from the client.

- **Span cost** = `promptTokens × input price + completionTokens × output price` of the
  span's `model`. 0 if the span has no `model` or the model is not in the price table.
- The rate is the one in force when the request is received (some prices are scheduled
  to change on a date), and models with a long-context tier use it when that span's (or
  trace's) `promptTokens` is above the threshold. So send one span per model call: tokens
  summed over several calls can cross a tier that no single call did.
- **Trace cost**: if `spans` is present and non-empty, the **sum of the span costs**.
  Otherwise, computed from the trace's `model`, `promptTokens` and `completionTokens`.
- Unknown models cost 0 (the trace is still stored).
- Stored precision: trace cost 6 decimal places, span cost 8. The `costUsd` in the response
  is the unrounded value.

Priced models (USD per 1M tokens; `src/lib/pricing.ts` records the source of each price):

| Model | Input | Output (incl. thinking) | Source |
|---|---|---|---|
| `gemini-3.1-pro-preview` | $2.00 (prompt ≤ 200k tokens), $4.00 above | $12.00 (≤ 200k), $18.00 above | Google pricing page, checked 2026-11-17 |
| `gemini-3.8-flash` | $0.75 through 2026-12-31, $1.50 from 2027-01-01 | $3.75, then $7.50 from 2027-01-01 | Google pricing page, checked 2026-11-17 |
| `gemini-2.5-flash` | $0.30 | $2.50 | Google pricing page, checked 2026-11-17 |
| `gemini-embedding-001` | $0.15 | — | Google pricing page, checked 2026-11-17 |
| `claude-sonnet-4-6` | $3.00 | $15.00 | original table, not re-checked |
| `gpt-4o` | $2.50 | $10.00 | original table, not re-checked |

Model ids must match exactly (`gemini-3.8-flash`, not `models/gemini-3.8-flash`).

## Responses

All responses are JSON.

| Status | Body | When |
|---|---|---|
| `202 Accepted` | `{"id": string, "evalsQueued": integer, "costUsd": number, "spans": integer}` | Stored. `id` is the trace id; `evalsQueued` = evaluation jobs queued for it; `spans` = number of spans stored. |
| `400 Bad Request` | `{"error":"invalid json"}` | Body is not valid JSON. |
| `401 Unauthorized` | `{"error":"invalid api key"}` | Missing, malformed or unknown key. |
| `422 Unprocessable Entity` | `{"error":"validation failed","details":{"formErrors":[...],"fieldErrors":{...}}}` | Body is JSON but breaks the schema. `fieldErrors` is keyed by the top-level field; every span problem is listed under `fieldErrors.spans`. |
| `500 Internal Server Error` | `{"error":"internal error"}` | Database or queue failure. Nothing was stored (trace, spans and jobs are written in one transaction); safe to retry. |

Example 422 (a span names a parent that is not in the request):

```json
{
  "error": "validation failed",
  "details": {
    "formErrors": [],
    "fieldErrors": { "spans": ["parentId \"plannr\" is not the id of a span in this request"] }
  }
}
```

`202` means stored and queued, not evaluated: evaluator results appear on the trace page
when the worker has run them (`/projects/<projectId>/traces/<id>`).

Which evaluators run: every active evaluator of the key's project that is either unscoped
or scoped to this trace's `name`.

Requests are not idempotent: sending the same body twice stores two traces.

## Example: a multi-agent run

```bash
curl -sS -X POST "$BASE_URL/api/v1/traces" \
  -H "Authorization: Bearer $EVAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d @- <<'JSON'
{
  "name": "triage.run",
  "input": "Customer: I was charged twice for my March invoice.",
  "output": "Routed to billing (priority high): duplicate charge on the March invoice.",
  "model": "gemini-3.1-pro-preview",
  "promptTokens": 4200,
  "completionTokens": 380,
  "latencyMs": 5120,
  "metadata": { "ticketId": "T-1042", "app": "triage" },
  "spans": [
    { "id": "root", "name": "orchestrator", "kind": "agent",
      "startedAt": "2026-11-17T10:00:00.000Z", "latencyMs": 5120 },
    { "id": "plan", "parentId": "root", "name": "planner", "kind": "llm",
      "model": "gemini-3.1-pro-preview", "startedAt": "2026-11-17T10:00:00.050Z",
      "latencyMs": 1900, "promptTokens": 3000, "completionTokens": 300 },
    { "id": "classify", "parentId": "root", "name": "classifier", "kind": "agent",
      "startedAt": "2026-11-17T10:00:02.000Z", "latencyMs": 2100 },
    { "id": "classify.llm", "parentId": "classify", "name": "classify", "kind": "llm",
      "model": "gemini-3.8-flash", "startedAt": "2026-11-17T10:00:02.010Z",
      "latencyMs": 800, "promptTokens": 1200, "completionTokens": 80 },
    { "parentId": "classify", "name": "kb_search", "kind": "tool", "status": "error",
      "startedAt": "2026-11-17T10:00:02.900Z", "latencyMs": 400,
      "metadata": { "error": "timeout after 400 ms" } }
  ]
}
JSON
```

Response (`202`):

```json
{ "id": "<trace id>", "evalsQueued": 2, "costUsd": 0.0108, "spans": 5 }
```

`evalsQueued` depends on the project's evaluators. `costUsd` here is
`3000 × $2.00/1M + 300 × $12.00/1M` (planner, gemini-3.1-pro-preview) plus
`1200 × $0.75/1M + 80 × $3.75/1M` (classify, gemini-3.8-flash) = `0.0096 + 0.0012 = 0.0108`;
the agent and tool spans have no model and cost 0. If the prices in `src/lib/pricing.ts`
change, this number changes with them.

## Minimal example

```bash
curl -sS -X POST "$BASE_URL/api/v1/traces" \
  -H "Authorization: Bearer $EVAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"summarize","input":"Long text...","output":"Short summary.","model":"gemini-3.8-flash","promptTokens":1200,"completionTokens":80,"latencyMs":950}'
```
