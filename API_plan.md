# API Plan: Multi-Provider LLM Integration (Claude + OpenAI + Local)

## Goals
- Add optional cloud LLM backends (Anthropic Claude, OpenAI) that integrate into the existing LLMRouter/LLMWorker pipeline.
- Keep existing local LLM backends (llama.cpp and PyTorch/Transformers) fully supported.
- Support multi-provider routing and graceful fallback based on availability, cost, and policy.
- Preserve real-time responsiveness by keeping all network LLM calls off the control loop.
- Keep cloud usage opt-in, with safe defaults (`cloud_enabled: false`) and clear configuration.
- Leverage cloud-native features (tool use, extended thinking, prompt caching) where they improve agent quality.

## Non-goals
- Removing or breaking the current local LLM stack.
- Making cloud LLM usage the default.
- Rewriting prompt construction — cloud backends adapt to the existing prompt pipeline.

## Current State (Summary)
- **LLMRouter** (`src/maxim/models/language/router.py`) is the central LLM abstraction. It loads config, selects backends via `_get_backend()`, formats prompts via `_PROMPT_BUILDERS`, and extracts JSON from completions.
- **LLMWorker** (`src/maxim/agents/llm_worker.py`) is the primary inference orchestrator. It runs on a dedicated thread/lane, uses `LLMRouter.generate_json()`, and handles prompt budgeting (`PromptBudgeter`), energy tracking, timeouts, fallback responses, and reasoning carryover.
- **ExecAgent** (`src/maxim/agents/exec_agent.py`) uses `ChatLLMAgent` directly (bypassing LLMWorker/LLMRouter). This is a simpler, older path that should eventually converge with LLMWorker.
- **LLMEnergyTracker** (`src/maxim/energy/llm_tracker.py`) already has model multipliers for Claude (Haiku 0.5, Sonnet 1.0/1.2, Opus 2.0/2.5) and GPT-4o (1.5), anticipating cloud integration.
- LLM configuration cascades: env vars → JSON config files → builtin profiles → defaults.
- Two local backends exist: `_LlamaCppBackend` and `_PyTorchTransformersBackend`.

## Proposed Architecture

### 1) Extend LLMRouter with Cloud Backends (not a parallel gateway)

The codebase already has the right abstraction layers. Rather than introducing a separate "gateway" class, we extend `LLMRouter._get_backend()` to dispatch to new cloud backend classes alongside the existing local ones.

Each backend conforms to the existing interface pattern (`complete()`, `warmup()`, `unload()`) and adds a structured response type for usage metadata:

```python
@dataclass
class LLMResponse:
    """Structured response from any LLM backend."""
    content: str
    input_tokens: int = 0       # Actual count from API, or estimate for local
    output_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0
    provider: str = ""          # "anthropic", "openai", "llama_cpp", "pytorch"
    stop_reason: str = ""       # "end_turn", "max_tokens", "tool_use", etc.
    tool_calls: list[dict] | None = None  # Native tool use results (Claude/OpenAI)
    cached_input_tokens: int = 0          # Optional: provider cache hit tokens
    uncached_input_tokens: int = 0        # Optional: provider cache miss tokens
```

Key design decisions:
- **Cloud backends set `requires_prompt_formatting = False`** — they receive raw `(system, messages)` and handle formatting internally. Local backends keep `requires_prompt_formatting = True` and receive formatted prompt strings via `_PROMPT_BUILDERS`.
- **`LLMResponse` is returned by `complete_with_usage()`** — energy tracking gets real token counts from cloud APIs instead of `len(text) // 3` estimates, while `complete()` remains `str` for compatibility.
- **`_get_backend()` gains new dispatch branches** for `"anthropic"`, `"openai"`, `"openai_compatible"`.

### 2) Provider Implementations

**Existing (unchanged):**
- `_LlamaCppBackend`: Local GGUF models via llama-cpp-python.
- `_PyTorchTransformersBackend`: HuggingFace Transformers (Blackwell GPU support).

**New cloud backends:**

#### Anthropic Claude (`src/maxim/models/language/anthropic_backend.py`)
- Uses `anthropic` SDK (optional dependency).
- **Native tool use**: ExecAgent's `ProposedGoal` schema maps to Claude's `tools` parameter. Returns structured tool calls with guaranteed schema compliance — no fragile JSON extraction from text.
- **Extended thinking**: Available for complex planning in ExecAgent (multi-step goal decomposition).
- **Prompt caching**: Foundational context (CONSTITUTION.md + AGENTS.md) is identical across calls. Claude's prompt caching reduces cost for these repeated prefixes.
- **System prompt**: Claude treats system as a dedicated parameter — maps naturally to ExecAgent's `SYSTEM_PROMPT` and LLMWorker's foundational context.
- Models: `claude-sonnet-4-5-20250514` (planning default), `claude-haiku-4-5-20251001` (fast routing).

#### OpenAI (`src/maxim/models/language/openai_backend.py`)
- Uses `openai` SDK (optional dependency).
- Supports function calling for structured output.
- Models: `gpt-4o` (planning), `gpt-4o-mini` (routing).

#### OpenAI-Compatible (`openai_compatible` type in config)
- Uses `openai` SDK with custom `base_url`.
- For self-hosted endpoints (vLLM, Ollama with OpenAI-compat, local gateways).
- No prompt template formatting (server handles it).

#### Other APIs (Future — Phase 4+)
- **Deferred until core providers are stable.** Do not add a plugin registry before `_AnthropicBackend` and `_OpenAIBackend` are battle-tested.
- When ready, additional providers (Gemini, Mistral API, Azure OpenAI) are added as new backend classes inside `src/maxim/models/language/`, registered in a hardcoded allowlist within `_get_backend()`. No dynamic module loading from config — this prevents arbitrary code execution if `llm.json` is tampered with.
- Keep the interface identical to `_OpenAIBackend` and `_AnthropicBackend` so routing and energy tracking remain uniform.
- Fail closed: unknown provider names are rejected with a clear error and no network call.

### 3) Routing and Fallback Policy

Explicit policy object instead of ad-hoc logic:

```python
@dataclass
class RoutingPolicy:
    """Governs how requests are dispatched across providers."""
    provider_priority: list[str]       # ["anthropic", "openai", "local"]
    fallback_on_rate_limit: bool = True
    fallback_on_timeout: bool = True
    fallback_on_budget_exceeded: str = "local"  # "local", "reject", or "queue"
    require_cloud_opt_in: bool = True  # Must set cloud_enabled: true explicitly
    context_window_routing: bool = True  # Auto-route large prompts to high-context providers

    # Cost guardrails — all in USD. 0 = unlimited (NOT recommended for cloud_enabled).
    # When cloud_enabled: true, at least one cost limit MUST be non-zero or startup fails.
    max_cost_per_request: float = 0.50  # Hard ceiling per individual API call
    max_cost_per_hour: float = 1.00     # Rolling hourly window
    max_cost_per_day: float = 10.00     # Rolling daily window
    max_cost_per_month: float = 100.00  # Calendar month

    # Graduated degradation thresholds (fraction of any budget window)
    cost_warning_threshold: float = 0.80   # 80%: log warning, downgrade model tier
    cost_critical_threshold: float = 0.95  # 95%: restrict to cheapest cloud model only
    # At 100%: fall to local per fallback_on_budget_exceeded
```

Routing logic:
1. Check `require_cloud_opt_in` — reject cloud dispatch if not explicitly enabled.
2. **Pre-flight cost check**: Estimate **LLM request cost only** from prompt token count + `max_tokens` + `thinking.budget_tokens`. Reject or downgrade model if estimated cost exceeds `max_cost_per_request`.
   - **Parallel actions aggregate check (future)**: If batching multiple **LLM requests** is introduced, aggregate **LLM cost only** before dispatching any of them. The sum of all LLM costs must fit within `max_cost_per_request` (per-batch) and not push the hourly/daily window past its threshold. If the aggregate exceeds budget, reduce the batch (drop lowest-priority LLM requests) or downgrade model before dispatch.
   - Tool execution cost modeling is **out of scope** for Phase 0–2 and should be added separately if needed.
3. **Budget window check**: Check `CostTracker` against hourly/daily/monthly windows:
   - Below `cost_warning_threshold`: dispatch normally.
   - Between warning and critical: log warning, auto-downgrade model tier (e.g., Sonnet → Haiku).
   - Between critical and 100%: restrict to cheapest cloud model only.
   - At 100%: route per `fallback_on_budget_exceeded` (local/reject/queue).
4. Check prompt size vs provider `n_ctx` — a 50K-token context can't go to a 4K local model.
5. Try providers in `provider_priority` order. On rate limit (429) or timeout, fall to next provider with exponential backoff at the provider level.
6. If all providers fail, use `LLMWorker._generate_llm_fallback()` (existing graceful degradation).

### Budget Visibility (Prompt Context)
Budgets should be visible to the agent so it can make cost-aware choices, without being able to bypass enforcement.

Add a `BudgetContext` block to the prompt (ExecAgent + LLMWorker):
- `remaining_per_request`, `remaining_hour`, `remaining_day`, `remaining_month`
- `current_spend_hour`, `current_spend_day`, `current_spend_month`
- `active_budget_tier` (normal / warning / critical / blocked)
- `is_budget_blocked` (true/false)
- `effective_model` — the model that will actually be used *after* any degradation (e.g., if budget is at 95%, this shows "claude-haiku-4-5-20251001" even if the profile says Sonnet). This lets the agent adapt its prompt complexity to the model it will actually get.

**Spend rate and horizon projection** (enables long-horizon reasoning):
- `spend_rate_3h` — EMA (alpha=0.3) over last 3 hours. Captures bursty behavior.
- `spend_rate_24h` — EMA (alpha=0.2) over last 24 hours. Better for daily projections.
- `spend_rate_7d` — EMA (alpha=0.1) over last 7 days. Better for monthly projections.
- `hours_until_daily_limit` — `remaining_day / spend_rate_24h`.
- `hours_until_monthly_limit` — `remaining_month / spend_rate_7d`.
- `min_spend_samples` — if fewer than N cloud calls in the window (default 5), that window's EMA is considered immature.

**EMA maturity fallback hierarchy**: Longer windows take time to mature. Rather than showing `null` for immature windows (which hides the monthly projection for the entire first week), fall back to the next-shorter mature window:
- `hours_until_monthly_limit`: prefer `spend_rate_7d`. If immature, use `spend_rate_24h`. If that's also immature, use `spend_rate_3h`. If all immature, show `null`.
- `hours_until_daily_limit`: prefer `spend_rate_24h`. If immature, use `spend_rate_3h`. If immature, show `null`.
- When a shorter-window fallback is used, append `"(estimated from shorter window)"` to the BudgetContext field so the LLM knows the projection is less reliable.

These fields let the LLM reason about horizons: "I have 5.8 hours of budget at current rate — this is a 2-hour task, I can afford Sonnet" vs "I have 0.3 hours left — switch to Haiku or defer to local." Without trend data, BudgetContext is a snapshot the agent can't extrapolate from.

**Per-step budget reservation** (when a multi-step plan is active):
- `plan_steps_remaining` — how many sub_goals are left in the current plan.
- `plan_duration_estimate_minutes` — estimated remaining plan duration. **Source**: In Phase 2, use a simple heuristic: `plan_steps_remaining × avg_step_duration_minutes` where `avg_step_duration_minutes` is the rolling average of recent step execution times (default: 2.0 minutes if no history). In Phase 3+, upgrade to NAc's `TemporalDelta` predictions per action type (e.g., `internet_search` averages 8s, `llm_planning` averages 15s) summed across remaining steps.
- `plan_per_step_budget` — `remaining_in_tightest_window / plan_steps_remaining`, **unless** `plan_duration_estimate_minutes` exceeds the tightest window; in that case use the next-longer window (daily or monthly) or a time-weighted prorated budget.
- `plan_spent_so_far` — USD spent on already-executed steps of this plan.

This prevents front-loading: if step 1 burns a disproportionate share, the updated `plan_per_step_budget` for step 2 shrinks, naturally pressuring the agent toward cheaper actions for remaining steps. Only injected when a `PlanCostEstimate` is active (Phase 2+); omitted for single-action requests.

**Token overhead gating**: BudgetContext adds ~80-120 tokens to every prompt. To avoid wasting context on local-only calls where cost is zero:
- Only inject BudgetContext when the provider has `cost_visible: true` (default true for cloud providers, configurable for `openai_compatible`).
- In `PromptBudgeter`, assign BudgetContext `SectionPriority.NICE_TO_HAVE` so it is the first section dropped if the prompt approaches the context window limit. If `SectionPriority` does not exist yet, add it alongside `PromptBudgeter` as a minimal enum.

**Ordering note**: If `effective_model` is included, routing must select and finalize the provider/model **before** prompt injection. If routing is delayed, omit `effective_model` to avoid lying in the prompt.

**Budget reserve** (long-horizon stabilization):
- Add `reserved_budget_ratio` (default 0.2) to keep a buffer in each window.
- `remaining_*` used for planning should be `max(0, raw_remaining * (1 - reserved_budget_ratio))` unless in emergency mode.
- **Interaction with graduated degradation**: This reserve is intentionally more conservative than RoutingPolicy's 80%/95% thresholds. The reserve applies to the *planning-visible* budget (BudgetContext), not to enforcement. So the agent sees tighter constraints than the system enforces — it self-limits before RoutingPolicy forces degradation. This is by design: the agent panics slightly early, reducing the chance that RoutingPolicy's hard cliff gets hit. Do NOT align the reserve with degradation thresholds; the gap between soft agent-side conservatism and hard server-side enforcement is the buffer that prevents thrashing.

Enforcement remains server-side (RoutingPolicy + CostTracker). The prompt is informative only.

### 4) Streaming Support

Cloud API calls can take 5-15s vs <1s for local models. Streaming mitigates this:

- Both `anthropic` and `openai` SDKs support streaming natively.
- Add `stream=True` option to `complete()` that yields partial tokens.
- LLMWorker can start validating/parsing JSON as it arrives, or cancel early if response goes off-track.
- Not required for Phase 0/1 — add in Phase 3 as an optimization.

### 5) Prompt Architecture (PromptPrompt + ExecutivePrompt)
Introduce a small prompt class hierarchy that wraps existing prompt-building logic rather than replacing it.

Design:
- `PromptPrompt` base class defines a stable interface:
  - `build_system(context) -> str`
  - `build_user(context) -> str`
  - `build_messages(context) -> list[dict]` (for cloud providers)
  - `build_text_prompt(context) -> str` (for local providers, uses existing `_PROMPT_BUILDERS`)
  - `inject(context, profile) -> InjectedPrompt` (applies injection slots)
- `ExecutivePrompt(PromptPrompt)` is the dedicated subclass for ExecAgent planning.
- Existing prompt construction in `LLMWorker` and `LLMRouter` remains intact and is wrapped, not rewritten.

Rationale:
- Minimizes refactor risk.
- Allows cloud backends to consume clean message objects.
- Preserves local prompt formatting for llama.cpp and Transformers.

### 6) Prompt Injection Slots (Minimal — Phase 0-2)
Add two injection slots to prompt profiles for the immediate needs. Expand later only if real usage patterns demand it.

**Phase 0-2 slots (start small):**
- `system_suffix`: Appended to the system prompt (e.g., "Return ONLY valid JSON.").
- `context_prefix`: Prepended to the user context (e.g., agent identity, foundational principles).

**Phase 3+ expansion (if needed):**
- `pre_tools`, `post_tools` — for provider-specific tool schema adjustments.
- Additional slots added only when a concrete use case requires them.

Applied by `PromptPrompt.inject()` to produce:
- `InjectedPrompt.system` (system + `system_suffix`)
- `InjectedPrompt.messages` (for cloud providers)
- `InjectedPrompt.text_prompt` (for local providers, via `_PROMPT_BUILDERS`)

**Security constraint**: Injection slot contents loaded from `llm.json` are subject to **local-only** review at config load time (regex + blocklist), even if FearAgent supports LLM review. Slots are append-only — they can add context but MUST NOT contain instructions that contradict CONSTITUTION.md principles (e.g., "ignore safety", "no constraints"). If a slot fails review, it is silently dropped and a warning is logged. LLM-based review is allowed only when `cloud_enabled: true` and explicitly opted in.

## Configuration Plan (Backward Compatible)

Extend current config without breaking existing keys. Add `cloud_enabled`, provider blocks, and per-agent profiles.

### Example data/util/llm.json:
```json
{
  "enabled": true,
  "cloud_enabled": false,
  "profile": "exec_planning",
  "providers": {
    "anthropic": {
      "type": "anthropic",
      "api_key_env": "ANTHROPIC_API_KEY",
      "timeout_s": 30,
      "max_retries": 2
    },
    "openai": {
      "type": "openai",
      "api_key_env": "OPENAI_API_KEY",
      "base_url": "https://api.openai.com/v1",
      "timeout_s": 60,
      "max_retries": 2
    },
    "local": {
      "type": "llama_cpp"
    }
  },
  "routing": {
    "provider_priority": ["anthropic", "openai", "local"],
    "fallback_on_rate_limit": true,
    "fallback_on_timeout": true,
    "require_cloud_opt_in": true,
    "max_cost_per_request": 0.50,
    "max_cost_per_hour": 1.00,
    "max_cost_per_day": 10.00,
    "max_cost_per_month": 100.00,
    "cost_warning_threshold": 0.80,
    "cost_critical_threshold": 0.95
  },
  "profiles": {
    "exec_planning": {
      "provider": "anthropic",
      "model": "claude-sonnet-4-5-20250514",
      "temperature": 0.2,
      "max_tokens": 1024,
      "n_ctx": 200000
    },
    "fast_routing": {
      "provider": "anthropic",
      "model": "claude-haiku-4-5-20251001",
      "temperature": 0.0,
      "max_tokens": 256,
      "n_ctx": 200000
    },
    "openai_planning": {
      "provider": "openai",
      "model": "gpt-4o",
      "temperature": 0.2,
      "max_tokens": 1024,
      "n_ctx": 128000
    },
    "local_small": {
      "provider": "local",
      "backend": "llama_cpp",
      "model_path": "data/models/LLM/SmolLM-1.7B-Instruct.Q4_K_M.gguf",
      "prompt_style": "chatml",
      "n_ctx": 4096
    }
  },
  "pricing": {
    "claude-sonnet-4-5-20250514": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00, "cached_input": 0.08},
    "gpt-4o": {"input": 2.50, "output": 10.00, "cached_input": 1.25},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.075}
  }
}
```

### Environment variable overrides (additive, optional):
- `MAXIM_LLM_PROVIDER` — provider name ("anthropic", "openai", "local")
- `MAXIM_LLM_CLOUD_ENABLED` — explicit opt-in for cloud providers ("true"/"false")
- `MAXIM_LLM_API_KEY` — API key (provider-agnostic override)
- `ANTHROPIC_API_KEY` — Anthropic SDK standard env var
- `OPENAI_API_KEY` — OpenAI SDK standard env var
- `MAXIM_LLM_BASE_URL` — custom endpoint for openai_compatible
- `MAXIM_LLM_MODEL` — model override
- `MAXIM_LLM_TIMEOUT_S` — request timeout
- `MAXIM_LLM_MAX_RETRIES` — retry count

### Per-Agent Profiles (resolved open question)

Per-agent profiles are the right design. ExecAgent needs deeper reasoning; LLMWorker needs fast routing. This matches how the system already works — they have different code paths with different latency/quality tradeoffs.

```json
{
  "agent_profiles": {
    "exec_agent": "exec_planning",
    "llm_worker": "fast_routing"
  }
}
```

Optional prompt injection slot example:
```json
{
  "prompt_profiles": {
    "exec_planning": {
      "system_suffix": "Return ONLY valid JSON. No prose, no explanations.",
      "context_prefix": "You are Maxim. Follow AGENTS.md and CONSTITUTION.md."
    }
  }
}
```
Note: Injection slot contents are reviewed locally (regex + blocklist) at config load time. Override-style instructions are silently dropped.

Providers may set `cost_visible` (boolean) to control whether BudgetContext is injected.
Defaults:
- `anthropic`/`openai`: true
- `openai_compatible`: false (must opt-in explicitly)
- `local`: false

## Phase 0: Foundation — Extend LLMRouter (Week 1-2)

Goal: Cloud backends work through the existing `LLMRouter` pipeline. LLMWorker gets cloud providers for free.

1. **Define `LLMResponse` dataclass** in `router.py` with usage metadata.
2. **Implement `_AnthropicBackend`** in new `src/maxim/models/language/anthropic_backend.py`:
   - `complete(prompt, *, max_tokens, temperature, stop, system=None) -> str`
   - `complete_with_usage(prompt, *, max_tokens, temperature, stop, system=None) -> LLMResponse`
   - `warmup() -> bool` (validates API key and connectivity)
   - `unload()` (no-op for cloud)
   - `requires_prompt_formatting = False`
   - `complete_with_tools(messages, tools, *, system, max_tokens) -> LLMResponse` (native tool use)
3. **Implement `_OpenAIBackend`** in new `src/maxim/models/language/openai_backend.py`:
   - Same interface as Anthropic backend.
4. **Extend `_get_backend()`** in `LLMRouter` with new dispatch branches:
   ```python
   if backend in ("anthropic", "claude"):
       from maxim.models.language.anthropic_backend import _AnthropicBackend
       self._backend = _AnthropicBackend(self.cfg)
   elif backend in ("openai", "gpt"):
       from maxim.models.language.openai_backend import _OpenAIBackend
       self._backend = _OpenAIBackend(self.cfg)
   ```
5. **Add `complete_with_usage()` compatibility layer** in `LLMRouter` so call sites that expect `str` remain untouched.
6. **Update `generate_json()`** in `LLMRouter`: skip `_PROMPT_BUILDERS` formatting when `backend.requires_prompt_formatting is False`.
7. **Update `LLMEnergyTracker` integration** in `LLMWorker._process_request()`: use real token counts from `LLMResponse` where available.
8. **Add config loading** for `cloud_enabled`, `providers`, `routing`, `agent_profiles`, and `prompt_profiles` in `load_llm_config()`.
9. **Add `pyproject.toml` optional deps**:
   ```toml
   llm-anthropic = ["anthropic>=0.40.0"]
   llm-openai = ["openai>=1.0.0", "tiktoken>=0.7.0"]  # tiktoken bundled with OpenAI for token counting
   ```

## Phase 1: LLMWorker with Cloud Routing (Week 3-4)

Goal: LLMWorker uses cloud providers with intelligent routing and fallback.

1. **Implement `RoutingPolicy`** and wire into `LLMRouter._get_backend()` selection logic.
2. **Add provider-level state**: rate limit backoff timers, per-provider error counts, health status.
3. **Context window routing**: `PromptBudgeter` checks `n_ctx` per profile. If prompt exceeds local model window, auto-route to cloud provider (if `cloud_enabled`).
4. **Budget-aware dispatch**: Check `CostTracker` budget status (hourly/daily/monthly) before cloud calls. Route to `fallback_on_budget_exceeded` target when over budget.
5. **API key re-reading**: On auth failures (401/403), re-read env var instead of using cached value. Supports key rotation without restart.
6. **Token counting**: Use provider tokenizers when available (local library), and fall back to response usage for accounting. Avoid extra network calls just to count tokens. Optional deps: `tiktoken` for OpenAI-compatible estimates.

## Phase 1.5: ExecAgent Early Path (Week 5, after Phase 1)

Goal: ExecAgent planning uses the new prompt path early without waiting for full LLMWorker migration. This phase is **sequential after Phase 1** because it depends on the routing policy and cloud backends being stable.

1. **Introduce `ExecutivePrompt`** (`PromptPrompt` subclass) for ExecAgent context.
2. **ExecAgent calls LLMRouter** via a dedicated prompt path using `ExecutivePrompt`.
3. **Preserve existing threading and rate limiting** so behavior remains stable.
4. **Validate JSON parsing** with both local and cloud providers.

## Phase 2: ExecAgent Migration (Week 6-7)

Goal: Migrate ExecAgent from direct `ChatLLMAgent` usage to the `LLMWorker` pipeline.

1. **ExecAgent submits to LLMWorker** instead of calling `ChatLLMAgent.generate_json()` directly.
2. **Benefits**: ExecAgent gains prompt budgeting, energy tracking, timeouts, fallback, and reasoning carryover for free.
3. **Conversation history**: LLMWorker's `conversation_history_text` field in `LLMRequest` replaces `ChatLLMAgent._history`.
4. **Per-agent profile**: ExecAgent uses `agent_profiles.exec_agent` (e.g. `exec_planning` → Claude Sonnet). LLMWorker uses `agent_profiles.llm_worker` (e.g. `fast_routing` → Claude Haiku or local).
5. **Deprecate `ChatLLMAgent`** as a direct inference path (keep as a wrapper for backward compat).

## Phase 3: Cloud-Native Features (Week 8+)

Goal: Leverage cloud API features that fundamentally improve agent quality.

### Claude Native Tool Use
ExecAgent's `ProposedGoal` has a fixed schema (`tool_name`, `params`, `priority`, `reasoning`, `sub_goals`). With Claude's tool use:
- Define the schema as a Claude `tool` definition.
- Claude returns structured tool calls with **guaranteed schema compliance**.
- Eliminates fragile `_extract_json_object()` parsing, ChatML token stripping, brace-matching repair, and all the failure modes that come with prompt-hacking JSON out of text completions.

### Extended Thinking
For complex planning (multi-step sub_goals, statistical pattern investigation):
- Enable Claude's extended thinking on `exec_planning` profile.
- Claude reasons internally before committing to a plan — better goal decomposition.
- Budget controlled via `thinking.budget_tokens` in profile config.
- **Default `thinking.budget_tokens`: 5000.** Hard ceiling: 20000. Extended thinking tokens are billed at output-token rates, so a 20K thinking budget on Opus costs ~$1.50 per call. The pre-request cost check in `RoutingPolicy` MUST include thinking budget in its estimate.
- Extended thinking is disabled by default; enable per-profile with `"thinking": {"enabled": true, "budget_tokens": 5000}`.

### Prompt Caching
The foundational context (`_load_foundational_context()` — CONSTITUTION.md + AGENTS.md principles) is identical across every call:
- Mark as cacheable in Anthropic API requests.
- Reduces per-call cost by avoiding re-processing ~500 tokens of constitutional context.
- Add profile-level config for caching (provider-specific):
  - `"prompt_cache": {"enabled": true, "scope": "foundational"}`
  - Note: TTL is server-controlled (Anthropic manages cache lifetime). The client opts in via `cache_control` markers on message blocks; there is no client-side `ttl_s` parameter.

### Streaming
- Add `stream=True` path in cloud backends.
- LLMWorker processes partial responses as they arrive.
- Enables early cancellation if response diverges from expected format.
- Reduces perceived latency for time-sensitive requests.

## Safety, Security, and Compliance

### API Key Management
- API keys read from environment variables only (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).
- On auth failures, re-read env var (supports rotation without restart).
- Never log API keys, auth headers, or full prompts at default log level.
- Redact keys in structured logging output.

### Cloud Opt-In
- `cloud_enabled: false` is the default. A stray env var alone will NOT send data to cloud APIs.
- Both `cloud_enabled: true` in config AND a valid API key must be present.
- Add a startup guard: if the redaction policy or data classification is missing, disable cloud dispatch and log a warning.
- Cloud opt-in in production should be gated on passing redaction tests in CI.

### Data Privacy
- StructuredContext (conversation history, memory contents, user data) is sent to cloud endpoints.
- Add a `CloudRedactionFilter` that strips sensitive fields before cloud dispatch.
- Introduce explicit data classification so redaction is meaningful:
  - `DataSensitivity` enum: `PUBLIC`, `INTERNAL`, `PRIVATE`, `SECRET`.
  - Default policy: redact `PRIVATE` and `SECRET` fields for cloud providers unless explicitly allowed.
  - Tag sources at creation time: memory entries, percept transcripts, tool inputs/outputs, file paths, system info.
- **Default for untagged data: `INTERNAL` (allowed for cloud).** This ensures the system works out of the box without requiring every existing code path to add sensitivity tags. However, the following known-sensitive fields are hardcoded `PRIVATE` regardless of tags:
  - Raw conversation transcripts from `ConversationManager`
  - Memory contents from Hippocampus (personal memories, associations)
  - File system paths and system environment info
  - API keys, tokens, and credential-adjacent strings (regex-matched)
- Data that is safe to send untagged (treated as `INTERNAL`):
  - Tool descriptions and available tool lists
  - Mode context and strategy descriptions
  - Statistical patterns and agent state summaries
  - CONSTITUTION.md and AGENTS.md foundational context
- `CloudRedactionFilter` is configurable per-provider: less redaction for self-hosted `openai_compatible` endpoints (which stay on your network), stricter for public cloud APIs. Default: strict for all providers.
- **Per-tool sensitivity map**: Tool inputs and outputs carry different sensitivity. Instead of a single default for all tools, define a per-tool sensitivity map:
  ```python
  TOOL_SENSITIVITY: dict[str, DataSensitivity] = {
      "internet_search": DataSensitivity.PUBLIC,
      "http_fetch": DataSensitivity.PUBLIC,
      "read_file": DataSensitivity.PRIVATE,
      "write_file": DataSensitivity.PRIVATE,
      "list_directory": DataSensitivity.PRIVATE,
      "glob": DataSensitivity.PRIVATE,
      "bash": DataSensitivity.SECRET,
      "execute_file": DataSensitivity.SECRET,
      "maxim_command": DataSensitivity.INTERNAL,  # refined by command map below
      "respond": DataSensitivity.INTERNAL,
      # Default for unlisted tools: INTERNAL
  }

  MAXIM_COMMAND_SENSITIVITY: dict[str, DataSensitivity] = {
      "request_shutdown": DataSensitivity.SECRET,
      "request_sleep": DataSensitivity.PRIVATE,
      "request_observe": DataSensitivity.INTERNAL,
      "goto_pose": DataSensitivity.PRIVATE,
      "move": DataSensitivity.SECRET,
      "look_at_image": DataSensitivity.PRIVATE,
      # Default for unlisted commands: INTERNAL
  }
  ```
  The redaction filter uses this map to decide whether tool call results are included in cloud prompts. Tools tagged `PRIVATE`/`SECRET` have their outputs redacted or summarized before cloud dispatch.
- **Runtime policy enforcement**: If config hot-reload removes the redaction policy, cloud dispatch halts immediately (not just at startup). Cloud calls re-check policy validity on every dispatch.

### Audit Log
Every cloud API call produces a structured audit entry, persisted to the existing structured logging system:

```python
@dataclass
class CloudAuditEntry:
    timestamp: float
    provider: str
    model: str
    data_categories_sent: list[str]      # ["tool_descriptions", "mode_context"]
    data_categories_redacted: list[str]  # ["conversation_history", "memory_contents"]
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    request_id: str
    agent: str                           # "exec_agent", "llm_worker"
    redaction_policy: str                # "strict", "relaxed", policy name
```

Audit entries are logged at INFO level via `log_structured()` and written to a separate append-only file (e.g., `data/logs/cloud_audit.jsonl`). Do **not** mix audit logs with `cost_state.json`.

### SSRF Protection
- Explicit allowlist for `base_url` in `openai_compatible` provider.
- Reject `localhost`, private IP ranges, and non-HTTPS URLs by default.
- **Post-DNS-resolution validation**: URL string checks alone are vulnerable to DNS rebinding (e.g., `evil.com` resolving to `127.0.0.1`). Use `socket.getaddrinfo()` to resolve the hostname *before* passing to the HTTP client, then validate the resolved IP against the private range blocklist. Reject if the resolved address is private/loopback.
- Allow local endpoints only with an explicit per-provider override (e.g., `allow_local_endpoints: true`) and still subject to DNS resolution checks.

### Response Safety Filter (No LLM Recursion)
- Validate cloud responses with a lightweight, local-only filter:
  - JSON/schema validation for action payloads
  - Tool allowlist checks
  - AutonomyController gating (unchanged)
- Do **not** use an LLM to review LLM outputs (prevents recursive calls and hidden cloud use).
- FearAgent heuristics can be reused locally if they do not require LLM inference.
 - Define a strict JSON schema (or dataclass contract) for tool actions and reject any unknown fields.

### Autonomy Preservation
- Existing tool gating and `AutonomyController` checks apply regardless of provider.
- Cloud providers cannot bypass `SafetyConstraints` or escalate `AutonomyLevel`.

## Energy and Cost Tracking

Energy and cost are **separate concerns** tracked independently:
- **Energy** (`LLMEnergyTracker`): Relative cognitive energy units used for internal agent decisions. Uses existing model multipliers (Haiku=0.5, Opus=2.0, etc.). Unchanged.
- **Cost** (`CostTracker`, new): Actual USD spend based on real API prices. Used for budget enforcement in `RoutingPolicy`. This is the layer that prevents runaway bills.

### Cost as an Energy Signal (Distinct)
Tie cost into the energy system as a distinct signal so the agent can learn cost sensitivity:
- Add `EnergyType.LLM_COST` (normalized units) and emit `EnergySignal` alongside `CostTracker` updates.
- Store actual USD in `EnergySignal.context` (e.g., `{"usd": 0.0123, "model": "...", "provider": "...", "lane": "infer_net"}`).
- EnergyRegistry + NAc can then learn “expensive action” associations the same way it learns latency or token cost.
- Normalization: map USD to a 0–100 energy scale per a configurable `cost_energy_scale` (e.g., $1.00 == 100 energy units).
- **NAc threshold co-tuning**: `EnergyBridgeConfig` thresholds (`high_energy_valence_threshold: 3.0`, `low_energy_valence_threshold: 0.5`) were tuned for LLM_TOKENS signals, not USD-scale cost signals. To avoid LLM_COST flooding the NAc with outsized values, add a separate `cost_bridge` config block:
  ```python
  cost_bridge:
      energy_type: LLM_COST
      cost_energy_scale: 100.0    # $1.00 = 100 energy units
      high_valence_threshold: 10.0  # >$0.10 per call = high cost signal
      low_valence_threshold: 1.0    # <$0.01 per call = negligible
      energy_weight: 0.3            # Same weight as other energy signals to NAc
  ```
  These thresholds are independent of the existing `EnergyBridgeConfig` so tuning cost sensitivity doesn't break LLM_TOKENS or MOTOR_COMMAND signals. The NAc learns cost associations at the same `energy_weight: 0.3` as other signals but with thresholds calibrated to typical API call costs ($0.001–$0.10 range).
- **Signal frequency note**: LLM_COST signals fire once per cloud API call (much less frequently than LLM_TOKENS, which fire for local calls too). This lower frequency is desirable — cost signals are high-value learning events, not noise. The NAc's temporal decay handles the sparse signal pattern naturally.
- **Config location**: `cost_bridge` lives under the energy config (e.g., `data/util/energy.json`) and is loaded alongside existing energy settings.

#### Strategy-Level Cost Learning (Long-Horizon NAc)
The NAc's 5-minute `temporal_window_seconds` is correct for per-action learning ("internet_search costs X") but too short for strategy-level patterns ("exploration strategy costs 10x more than assist strategy over a session"). Add a coarser-grain learning layer:

- **Session cost recording**: When a strategy session ends (mode change, sleep transition, or goal completion), record the total accumulated USD cost of that session as a single NAc outcome attributed to the `(strategy, goal_type)` event pair.
- **Normalize** per session: also record `usd_per_minute`, `usd_per_goal`, and `usd_per_action` so long sessions do not dominate learning.
- **Separate temporal window**: Use `strategy_temporal_window_seconds: 7200` (2 hours) for these coarser signals. This is independent of the per-action 5-minute window — it's a second learning channel, not a replacement.
- **What NAc learns**: "exploration + internet_search goals cost ~$2.00/session" vs "assist + conversation goals cost ~$0.10/session." Over time, the agent develops intuition about which strategies are expensive.
- **Integration**: `predict_energy()` already returns learned costs for action signatures. Extend it to also query strategy-level links when the current action is a strategy selection or goal decomposition. Weight: `0.3 * action_prediction + 0.7 * strategy_prediction` for planning decisions (inverted for individual tool calls).
- **Phase**: Phase 3+ (requires stable cost data from Phase 1-2 to learn from). Early phases use heuristic defaults.

This is analogous to how the brain distinguishes immediate costs ("this action costs 500 calories") from strategic costs ("this lifestyle costs 2000 cal/day"). Both inform decisions but at different horizons.

### Real Usage Data
- Cloud APIs return actual token counts in every response via `LLMResponse`.
- `LLMEnergyTracker.record()` uses real counts instead of `len(text) // 3` estimates.
- `CostTracker.record()` calculates actual USD cost from token counts and the price table.

### Token Usage Baselines (for PlanCostEstimate)
- Maintain rolling averages per `(tool_name, model)` for input/output tokens.
- Use these averages for plan-level cost estimates; fall back to heuristic defaults only if no history exists.
- **Heuristic defaults** (used when no rolling average is available yet):

| Action type | Est. input tokens | Est. output tokens | Notes |
|---|---|---|---|
| `llm_planning` (ExecAgent) | 2000 | 500 | Full context + system prompt |
| `llm_routing` (LLMWorker) | 800 | 300 | Smaller context, fast response |
| `internet_search` | 800 | 400 | Query + result parsing |
| `read_file` | 200 | 100 | Mostly local, minimal LLM |
| `respond` (TTS/speech) | 300 | 150 | Short conversational turn |
| `memory_recall` | 400 | 200 | Retrieval + association |

These defaults are conservative overestimates. They are replaced by rolling averages after ~10 observations per `(action, model)` pair.

### Price Table (separate from energy multipliers)
Energy multipliers are relative weights for internal agent decisions. The price table tracks actual API costs in USD per 1M tokens:

```python
@dataclass
class ModelPricing:
    """Actual API pricing for cost tracking. USD per 1M tokens."""
    input_price: float           # $/1M input tokens (uncached)
    output_price: float          # $/1M output tokens
    cached_input_price: float    # $/1M cached input tokens (provider cache hits)

MODEL_PRICES: dict[str, ModelPricing] = {
    # Anthropic (as of 2025) — cached input at 90% discount
    "claude-sonnet-4-5-20250514":  ModelPricing(3.00, 15.00, 0.30),
    "claude-haiku-4-5-20251001":   ModelPricing(0.80, 4.00, 0.08),
    "claude-opus-4-5-20250514":    ModelPricing(15.00, 75.00, 1.50),
    # OpenAI (as of 2025) — cached input at 50% discount
    "gpt-4o":                      ModelPricing(2.50, 10.00, 1.25),
    "gpt-4o-mini":                 ModelPricing(0.15, 0.60, 0.075),
    # Local models — zero cost
    "local":                       ModelPricing(0.0, 0.0, 0.0),
}
```

`CostTracker.record()` uses the split from `LLMResponse`:
```python
cost = (
    response.uncached_input_tokens * pricing.input_price / 1_000_000
    + response.cached_input_tokens * pricing.cached_input_price / 1_000_000
    + response.output_tokens * pricing.output_price / 1_000_000
)
```
If `cached_input_tokens` and `uncached_input_tokens` are both zero (local backends, or providers that don't report cache stats), fall back to `response.input_tokens * pricing.input_price / 1_000_000`.

Prices are configurable in `llm.json` under a `"pricing"` key to stay current without code changes. Prices are **per 1M tokens** and must match provider billing units.
If a cloud model is missing from the price table, **fail closed** (reject or require explicit pricing) rather than silently treating cost as zero. Local models default to zero cost.
Add `pricing_required` per provider (default true for cloud, false for local) to enforce this explicitly.

### Cost Persistence
Cost counters MUST survive process restarts. Without persistence, a monthly budget resets every time Maxim reboots.

- Persist cost tallies to `data/util/cost_state.json` (alongside existing `llm.json`).
- **Write buffering**: Instead of writing after every cloud API call, persist on a configurable interval (default `cost_persistence_interval_s: 10`) or after every N calls (`cost_persistence_interval_n: 5`), whichever comes first. Also flush on graceful shutdown (`atexit` handler). This avoids disk I/O on every inference call while keeping at most 10 seconds of data at risk.
- Atomic write with temp file + rename to prevent partial writes.
- **Corruption handling**: On startup, validate the JSON structure (required keys: `version`, `hourly`, `daily`, `monthly`). If the file is corrupt or missing required keys, rename it to `cost_state.json.corrupt.<timestamp>` for debugging, log a warning, and start from zero. Do not silently discard data — the corrupt file is preserved for manual inspection.
- **Versioning**: If `version` changes, run a small migration or refuse to load with a clear error; do not silently treat old data as valid.
- Load on startup; if missing, start from zero.
- Structure:
```json
{
  "version": 1,
  "hourly": {"window_start": 1740000000, "total_usd": 0.12},
  "daily":  {"window_start": 1740000000, "total_usd": 1.45},
  "monthly": {"window_start": 1738368000, "total_usd": 23.50},
  "lifetime": {"total_usd": 156.30, "total_requests": 4821},
  "spend_rates": {
    "ema_3h":  {"value": 0.42, "samples": 12, "last_update": 1740003600},
    "ema_24h": {"value": 0.38, "samples": 47, "last_update": 1740003600},
    "ema_7d":  {"value": 0.31, "samples": 203, "last_update": 1740003600}
  }
}
```
EMA values and sample counts survive restarts. Without persistence, the 7-day EMA would take a full week to mature after every reboot, leaving monthly projections invisible.

**Window semantics**:
- `hourly` and `daily` are rolling windows
- `monthly` is calendar-month

### Budget Enforcement
- **Pre-request**: Estimate cost from expected output tokens (rolling averages) + `thinking.budget_tokens`. Treat `max_tokens` as a hard cap, not the estimator.
- **Pre-batch (parallel actions)**: If batching multiple **LLM requests** is introduced, estimate aggregate **LLM cost only** across all N requests. The batch total is checked against `max_cost_per_request` (as a batch ceiling) and the remaining budget window. If the batch exceeds limits, trim lowest-priority LLM requests from the batch before dispatch. Tool execution cost modeling is out of scope for Phase 0–2.
 - **Per-provider spend**: Track per-provider cost tallies (hourly/daily/monthly) so audits can spot vendor spikes and budget caps can be enforced per provider if desired.
- **Per-window**: Check hourly/daily/monthly tallies against `RoutingPolicy` limits.
- **Graduated degradation** (not a hard cliff):
  - 80% of any budget: log warning, auto-downgrade model tier (Sonnet → Haiku).
  - 95% of any budget: restrict to cheapest cloud model only.
  - 100%: fall to local per `fallback_on_budget_exceeded`.
- **Startup guard**: If `cloud_enabled: true` and ALL cost limits are zero (unlimited), refuse to start and log an error. At least one limit must be set.

### Plan-Level Cost Projection (Long-Horizon Awareness)

Single-action cost checks are insufficient for multi-step plans. ExecAgent decomposes goals into `sub_goals` — a 5-step plan checked one step at a time will pass steps 1-3, then hit the budget threshold mid-plan, degrading steps 4-5 to Haiku or local. A plan designed for Sonnet-quality reasoning produces incoherent results when half executes on a weaker model.

**Solution**: Before executing a multi-step plan, estimate aggregate cost upfront:

```python
@dataclass
class StepEstimate:
    action: str                     # e.g., "internet_search", "llm_planning"
    estimated_input_tokens: int     # From NAc learned averages or heuristic
    estimated_output_tokens: int
    estimated_usd: float            # From CostTracker price table + model

@dataclass
class PlanCostEstimate:
    """Projected cost for a multi-step plan."""
    steps: list[StepEstimate]       # Per-step cost projection
    total_estimated_usd: float      # Sum of all steps
    model_assumed: str              # Model assumed for estimates
    fits_within_budget: bool        # total < remaining in tightest window
    tightest_window: str            # "hourly", "daily", or "monthly"
    remaining_in_window: float      # USD remaining in tightest window
    suggested_model: str | None     # Downgrade suggestion if over budget
```

**How it works:**
1. ExecAgent proposes a plan with N sub_goals, each mapping to a tool call (or LLM call for sub-planning).
2. For each step, estimate token usage from NAc learned averages (`predict_energy(action_signature)`) or a heuristic fallback (e.g., `internet_search` averages 800 input + 400 output tokens).
3. Sum all step estimates against the price table for the current model.
4. Check total against the **tightest** remaining budget window (min of remaining_hour, remaining_day, remaining_month).
5. If the plan doesn't fit:
   - **First**: Try downgrading the model for the *entire* plan (consistent quality). Re-estimate with Haiku instead of Sonnet.
   - **Second**: If still over, reduce the plan (drop lowest-priority sub_goals).
   - **Third**: If still over, reject the plan and return to ExecAgent with a budget-constrained re-planning signal.

**Plan model lock**:
- Add `plan_model_lock` (default true). If locked, do not silently degrade mid-plan; instead trigger a replan when budgets change.
- **Replan cooldown**: Max 1 replan per plan execution. If the replan itself pushes spend past a threshold, do NOT trigger a second replan — instead fall back to completing remaining steps on the degraded model. This prevents a cost spiral where replanning burns tokens that trigger further replanning. The replan call itself is exempt from triggering a `plan_model_lock` replan.
6. Inject the `PlanCostEstimate` summary into the prompt for the next planning cycle so the agent can learn to propose cheaper plans.

**Phase**: Implement in Phase 2 (ExecAgent migration), since that's when ExecAgent routes through LLMWorker and has access to CostTracker and NAc predictions.

**Key principle**: Downgrade the entire plan's model, not individual steps. A plan should execute at a consistent quality level to avoid incoherent partial results.

### Budget vs Energy (Clarification)
- **Budget** is hard enforcement (RoutingPolicy + CostTracker). The agent cannot override it.
- **Energy** is soft guidance (NAc learns to avoid expensive actions over time).
- `BudgetContext` is informative only; it never relaxes enforcement.

**Energy-cost divergence signal**: Energy (`EnergyBudget`) recharges passively at `recharge_rate` per second (default: 10/sec for LLM, capacity=1000). USD cost never recharges. After a burst of cloud calls, energy refills quickly but the hourly budget stays consumed. This means the NAc's energy-based intuitions ("plenty of energy available") can directly contradict budget reality ("95% of hourly budget spent").

To prevent this, add an explicit divergence signal to the prompt when the two disagree:
- `cost_energy_divergence`: `"none"` | `"moderate"` | `"high"`
  - `"none"`: Energy budget % and cost budget % are within 20 points of each other.
  - `"moderate"`: Energy says >50% available but cost says <30% remaining in any window.
  - `"high"`: Energy says >50% available but cost says <10% remaining (critical tier).
- When divergence is `"high"`, append a one-line note to BudgetContext: `"Energy has recharged but USD budget is nearly exhausted. Prefer local or defer."` This is informational — RoutingPolicy enforces the actual constraint — but helps the LLM align its planning intuition with real budget pressure.

**Phase**: Include in Phase 1 alongside BudgetContext injection. The signal is cheap to compute (compare two percentages) and prevents a common failure mode where the agent "feels" energetic but is actually broke.

### Model Multipliers (Energy — unchanged)
Existing energy multipliers in `LLMEnergyConfig` remain as-is for internal agent decisions:
- Claude: Haiku (0.5), Sonnet (1.0/1.2), Opus (2.0/2.5)
- OpenAI: GPT-4o (1.5), GPT-4o-mini (0.4), GPT-4-turbo (1.8)
- Local: llama.cpp (0.2), Ollama (0.3)

## Performance and Responsiveness

- All cloud calls remain off the control loop (LLMWorker runs in `WorkerPool` lane or background thread).
- Request timeouts at provider level (configurable per-provider `timeout_s`).
- Exponential backoff on rate limits with provider-level state (not per-request).
- Context window routing prevents wasted calls to providers that can't handle the prompt size.
- Streaming (Phase 3) reduces perceived latency for cloud calls.

## WorkerPool Lanes and GPU Assumptions
- Cloud providers do not require local GPU availability.
- The existing GPU-availability gate for agentic runtime remains unchanged; cloud usage does not override that policy.
- Add a new `infer_net` lane for network LLM calls (GPU-agnostic). Register lanes via a small lane registry so providers can declare preferred lanes without hardcoding.
- Keep `infer` lane for local GPU-bound inference.
- **Concurrency limits per provider**: The `infer_net` lane's `max_workers` should respect cloud API rate limits. Configure per-provider in the `providers` config block:
```json
"anthropic": {
    "type": "anthropic",
    "max_concurrent_requests": 2,
    ...
}
```
  The `infer_net` lane uses `max_workers = max(provider.max_concurrent_requests)` across configured cloud providers. Individual provider concurrency is enforced via per-provider semaphores within the lane.
- Track usage per lane in `LLMEnergyTracker` by adding a `lane` field to each `EnergySignal` context.

## Testing Plan

### Unit Tests
- `MockLLMProvider` for deterministic JSON responses (existing pattern in `test_llm_worker_pool.py`).
- **Contract tests**: `complete_with_usage()` returns valid `LLMResponse` with all fields populated; `complete()` remains `str` for compatibility.
- **Prompt template bypass tests**: Verify cloud backends receive raw messages, not `<|im_start|>` formatted strings.
- **Routing policy tests**: Simulate rate limits, timeouts, budget exceeded — verify fallback behavior.
- **Redaction filter tests**: Verify sensitive data is stripped before cloud dispatch.

### Integration Tests
- Gated by environment variables (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY` present).
- Validate real API calls return expected `LLMResponse` shape.
- **Latency characterization**: Compare local vs cloud response times under load.
- **Energy tracking accuracy**: Compare estimated vs actual token counts from cloud APIs.
- Offline default: tests run without network unless explicitly enabled via `pytest -m integration`.

### Simulation-Backed Tests
- Prefer the existing robot simulation harness if one already exists.
- If no simulator is available, build a `StructuredContext` fixture factory (in `tests/conftest.py`) that produces realistic context snapshots.
- Use simulated contexts in unit tests to validate routing, redaction, and prompt budgeting without hitting real APIs or requiring hardware.
- Include contexts of varying sizes (small/medium/large) to test context window routing logic.
- Keep all simulation tests offline by default.

### Cost and Budget Tests
- Verify `CostTracker` persistence: write state, restart (reload from file), confirm tallies survive.
- Verify graduated degradation: mock cost at 80% → confirm model downgrade; at 95% → cheapest only; at 100% → local fallback.
- Verify per-request cost pre-check: mock a large prompt + thinking budget → confirm rejection before API call.
- Verify startup guard: `cloud_enabled: true` with all cost limits at zero → confirm startup failure with actionable error.
- Verify `BudgetContext` injection: prompt includes remaining budget and current spend for ExecAgent + LLMWorker.
- Verify `EnergyType.LLM_COST` signal is emitted with correct USD in context and lane attribution.
- Verify cached token pricing: mock a response with `cached_input_tokens=400, uncached_input_tokens=100` → confirm cost uses split pricing (not flat `input_tokens * input_price`).
- Verify `PlanCostEstimate`: mock a 5-step plan → confirm aggregate estimate checks against tightest budget window; confirm whole-plan model downgrade when over budget.
- Verify spend rate EMAs: mock a sequence of cloud calls over 3 hours → confirm `spend_rate_3h`, `spend_rate_24h`, `spend_rate_7d` reflect trends at their respective windows.
- Verify EMA maturity fallback: with only 3 calls (below `min_spend_samples=5`), confirm `spend_rate_7d` falls back to `spend_rate_24h` or `spend_rate_3h` for monthly projection.
- Verify EMA persistence: write EMAs to `cost_state.json`, restart, confirm values and sample counts survive.
- Verify `hours_until_daily_limit` calculation: mock `remaining_day=$5.00` and `spend_rate_24h=$0.50` → confirm projection of 10 hours.
- Verify energy-cost divergence signal: mock energy at 80% available but cost at 95% of hourly budget → confirm `cost_energy_divergence: "high"` in BudgetContext.
- Verify per-step budget reservation: mock a 4-step plan with $2.00 remaining → confirm `plan_per_step_budget` = $0.50; execute step 1 at $0.80 → confirm step 2 shows `plan_per_step_budget` = $0.40.

### Fallback Behavior Tests
- Simulate provider failures (mock 429, 500, timeout) and verify graceful degradation.
- Verify `_generate_llm_fallback()` activates when all providers fail.
- Verify budget exceeded triggers correct fallback path.

### Audit and Redaction Tests
- Verify `CloudAuditEntry` is produced for every cloud API call with all required fields.
- Verify `CloudRedactionFilter` strips hardcoded PRIVATE fields (conversation transcripts, memory contents, file paths).
- Verify untagged data passes through as INTERNAL.
- Verify local blocklist rejects injection slot contents containing override phrases (no LLM review).

## Documentation and Decisions

- Update README.md to reflect optional cloud LLM support (Claude + OpenAI).
- Add DECISIONS.md entry: "Cloud LLM providers extend LLMRouter backends, not a parallel gateway."
- Add DECISIONS.md entry: "Cloud usage requires explicit opt-in (`cloud_enabled: true`)."
- Add DECISIONS.md entry: "Per-agent profiles — ExecAgent uses planning profile, LLMWorker uses routing profile."
- Provide example `llm.json` configurations for: Claude-only, OpenAI-only, hybrid (Claude planning + local routing), local-only (backward compat).

## Resolved Questions (from original plan)

| Question | Decision | Rationale |
|---|---|---|
| Preferred model default for planning? | `claude-sonnet-4-5-20250514` | Best at structured output and tool use. Sonnet balances quality/cost for planning. |
| Require explicit opt-in even if API key set? | **Yes.** `cloud_enabled: false` default. | A stray env var should never send data to cloud APIs. Aligns with "cloud usage opt-in" goal. |
| Per-agent or shared profile? | **Per-agent.** | ExecAgent needs deep reasoning (Sonnet/Opus). LLMWorker needs fast routing (Haiku/local). Different latency/quality tradeoffs. |
| Provider-specific safety filter? | **Yes.** Local-only filter (JSON schema validation, tool allowlist, AutonomyController). No LLM recursion. | Cloud models can be jailbroken. Same scrutiny as local responses, but without recursive cloud calls. |

## Remaining Open Questions

- Should extended thinking be available for all agents or restricted to ExecAgent planning?
- Do we want a response cache (hash-based with TTL) to reduce cloud API costs for repetitive states? If so, what TTL is appropriate given that StructuredContext changes frequently?
- Should `warmup()` for cloud backends validate model access (not just API key validity)? An API key might be valid but restricted from Opus. Validating model access during warmup avoids runtime failures but adds a billable API call at startup.
- What is the right cost-state file rotation strategy? Should `cost_state.json` roll over monthly, or accumulate indefinitely with a `lifetime` counter?

## Implementation Milestones

### Phase 0: Foundation (LLMRouter extension)
1. Define `LLMResponse` dataclass with usage metadata.
2. Implement `_AnthropicBackend` with `complete()` and `complete_with_usage()`.
3. Implement `_OpenAIBackend` with `complete()` and `complete_with_usage()`.
4. Extend `_get_backend()` dispatch and `generate_json()` prompt-formatting bypass.
5. Add a compatibility layer so `complete()` remains `str` for existing call sites.
6. Add config loading for `cloud_enabled`, `providers`, `routing`, `agent_profiles`, and `prompt_profiles`.
7. Add `pyproject.toml` optional deps (`llm-anthropic`, `llm-openai` with `tiktoken` bundled).
8. Wire `LLMResponse` usage data into `LLMEnergyTracker`.
9. **Startup guard**: Refuse to start with `cloud_enabled: true` if all cost limits are zero. Log actionable error message.

### Phase 1: LLMWorker with routing
10. Implement `RoutingPolicy` with fallback logic and graduated degradation.
11. Add provider-level rate limit state and exponential backoff.
12. Context window routing in `PromptBudgeter`.
13. Implement `CostTracker` with price table, persistence to `data/util/cost_state.json`, and budget enforcement (per-request, hourly, daily, monthly).
14. Implement `CloudRedactionFilter` with hardcoded PRIVATE defaults for known-sensitive fields.
15. Implement `CloudAuditEntry` logging for every cloud API call.
16. Add `infer_net` lane via the lane registry with per-provider concurrency semaphores.
17. Track usage per lane in `LLMEnergyTracker`.
18. Add `BudgetContext` injection to prompt building (ExecAgent + LLMWorker), including spend rate (EMA), horizon projections (`hours_until_*_limit`), and energy-cost divergence signal.
19. Add `EnergyType.LLM_COST` signal emission with configurable `cost_energy_scale`.

### Phase 1.5: ExecAgent early path (after Phase 1)
20. Introduce `PromptPrompt` base class and `ExecutivePrompt`.
21. Route ExecAgent planning through `ExecutivePrompt` and `LLMRouter`.
22. FearAgent review of prompt injection slot contents at config load time.

### Phase 2: ExecAgent migration
23. Migrate ExecAgent from `ChatLLMAgent` to `LLMWorker` submission.
24. Wire per-agent profile selection.
25. Implement `PlanCostEstimate` — aggregate cost projection before executing multi-step sub_goals. Whole-plan model downgrade if over budget.
26. Deprecate direct `ChatLLMAgent` inference path.

### Phase 3: Cloud-native features
27. Claude native tool use for ExecAgent's `ProposedGoal`.
28. Extended thinking for complex planning (default `budget_tokens: 5000`, hard ceiling: 20000, included in pre-request cost check).
29. Prompt caching for foundational context.
30. Streaming support in cloud backends.
31. Strategy-level cost learning in NAc (`strategy_temporal_window_seconds: 7200`) — session-end cost recording for long-horizon learning.

## Why Claude is a Particularly Good Fit for Maxim

| Feature | Benefit |
|---|---|
| **Native tool use** | ExecAgent proposes `ProposedGoal` with `tool_name` + `params`. Claude's tool_use returns structured JSON guaranteed to match a schema — eliminates fragile `_extract_json_object()` text parsing. |
| **Extended thinking** | ExecAgent's planning ("break complex tasks into sub_goals") benefits from reasoning before committing to a plan. |
| **System prompt (dedicated parameter)** | CONSTITUTION.md + AGENTS.md principles map naturally to Claude's system parameter — persists across turns without eating message context. |
| **200K context window** | `PromptBudgeter` currently truncates aggressively for ~2-4K local windows. Claude's window holds full `StructuredContext` without truncation. |
| **Prompt caching** | Foundational context is identical across calls. Caching reduces cost for repeated prefixes. |
| **Haiku/Sonnet/Opus tiering** | Maps to per-agent profiles: Haiku for LLMWorker fast routing, Sonnet for ExecAgent planning, Opus for complex multi-step research goals. |
| **Existing energy multipliers** | `LLMEnergyTracker` already has Claude model multipliers — integration was anticipated. |
