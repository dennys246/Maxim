# Communication Upgrade Plan

**Goal:** Make Maxim reachable and conversational across every practical medium — SMS, voice calls, Slack, Discord, email, WebSocket, and local audio — all deeply integrated into the agent pipeline so that conversations feel native regardless of channel.

**Current state:** Only Twilio SMS + voice exists. The `Channel` ABC is minimal (just `send` + optional `call`). Audio/TTS infrastructure is solid (Piper TTS, Whisper STT, Reachy speaker) but disconnected from the comms stack. The `CommunicationBridge` already wires outbound→NAc learning, but many values are hard-coded and the bridge can't adapt to channel-specific timing (email vs SMS).

---

## Phase 0 — Harden the Foundation (~200 LOC)

Before adding channels, fix the abstractions so new channels don't inherit current limitations.

### 0a. Upgrade Channel ABC

**File:** `src/maxim/comms/channels/base.py`

Expand the abstract interface:

```python
class Channel(ABC):
    name: str                          # "twilio", "slack", "discord", ...
    supports_voice: bool = False
    supports_media: bool = False
    supports_threading: bool = False
    supports_reactions: bool = False
    supports_realtime: bool = False    # WebSocket / audio stream

    @abstractmethod
    def send(self, recipient: str, body: str, *, thread_id: str | None = None,
             media: list[MediaAttachment] | None = None) -> SendResult: ...

    def call(self, recipient: str, message: str) -> bool:
        return False

    @abstractmethod
    def receive(self, raw: dict) -> InboundMessage: ...

    def validate_credentials(self) -> bool:
        return True

    def shutdown(self) -> None:
        pass
```

New dataclasses:

```python
@dataclass
class SendResult:
    success: bool
    message_id: str | None = None
    error: str | None = None

@dataclass
class InboundMessage:
    sender: str
    content: str
    channel: str
    thread_id: str | None = None
    media: list[MediaAttachment] | None = None
    raw: dict | None = None             # preserve original payload

@dataclass
class MediaAttachment:
    url: str | None = None
    data: bytes | None = None
    mime_type: str = "application/octet-stream"
    filename: str | None = None
```

**Why:** Every new channel currently needs ad-hoc `receive_sms`, `receive_voice` methods. A standard `receive()` → `InboundMessage` path lets the gateway handle all channels uniformly. `SendResult` replaces the bare `bool` so we can track delivery and surface errors.

### 0b. Remove Hard-coded Values

| Current | Location | Change |
|---------|----------|--------|
| `"twilio"` in `gateway.call()` | gateway.py:52 | Accept `channel` param, fall back to first voice-capable channel |
| `salience=0.9` for all inbound | gateway.py:76 | Per-channel default salience in Channel class, overridable |
| 1800s conversation expiry | conversation.py | Per-channel config dict, default 1800 |
| 300s response match window | communication_bridge.py | Per-channel config, default 300 |
| 600s unanswered timeout | communication_bridge.py | Per-channel config, default 600 |
| Keyword valence classifier | communication_bridge.py | Extract to pluggable classifier; LLM-based upgrade in Phase 4 |

### 0c. Standardize Inbound Flow

Refactor `gateway.receive_inbound()` to accept `InboundMessage` instead of raw args. All channels call `channel.receive(raw_payload)` → `InboundMessage` → `gateway.receive_inbound(msg)` → Percept on bus. One path, all channels.

### 0d. Migrate TwilioChannel

Update `TwilioChannel` to implement the new `receive()` method and return `SendResult` from `send()`. Preserve backward compat in the webhook handlers (they call `channel.receive(form_data)` now).

**Tests:** Unit tests for Channel ABC contract, SendResult, InboundMessage. Update existing Twilio tests.

---

## Phase 1 — Slack Channel (~250 LOC)

**File:** `src/maxim/comms/channels/slack_channel.py`

### Implementation

```python
class SlackChannel(Channel):
    name = "slack"
    supports_threading = True
    supports_media = True
    supports_reactions = True

    def __init__(self, bot_token: str, signing_secret: str,
                 default_channel: str | None = None) -> None: ...

    def send(self, recipient: str, body: str, *,
             thread_id: str | None = None, ...) -> SendResult: ...

    def receive(self, raw: dict) -> InboundMessage: ...

    def add_reaction(self, channel: str, timestamp: str, emoji: str) -> bool: ...
```

- **Dependency:** `slack-sdk>=3.27.0` (new optional extra: `comms-slack`)
- **Auth:** Bot token + signing secret via `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET`
- **Threading:** Map Slack `thread_ts` to ConversationManager's conversation model
- **Webhook:** Add `POST /api/slack/events` endpoint in `api.py` with Slack signature verification
- **URL verification:** Handle Slack's `url_verification` challenge automatically
- **User mapping:** Resolve Slack user IDs to display names via `users.info` (cache with TTL)

### Env Vars

```
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...
SLACK_DEFAULT_CHANNEL=#maxim       # Where Maxim posts unsolicited messages
```

### Bootstrap

In `build_comms_stack()`: if `SLACK_BOT_TOKEN` is set, instantiate `SlackChannel`, register as `"slack"`.

### Scopes Required

`chat:write`, `channels:read`, `users:read`, `reactions:write`, `files:write` (for media).

---

## Phase 2 — Discord Channel (~250 LOC)

**File:** `src/maxim/comms/channels/discord_channel.py`

### Implementation

```python
class DiscordChannel(Channel):
    name = "discord"
    supports_threading = True
    supports_media = True
    supports_reactions = True
    supports_realtime = True

    def __init__(self, bot_token: str, guild_id: str | None = None) -> None: ...
```

- **Dependency:** `discord.py>=2.3.0` (new optional extra: `comms-discord`)
- **Auth:** Bot token via `DISCORD_BOT_TOKEN`
- **Threading:** Discord threads map naturally to conversations
- **Gateway integration:** Discord.py uses its own event loop; bridge via `asyncio.run_coroutine_threadsafe()`
- **Webhook:** Discord interactions endpoint OR gateway bot events (gateway preferred for bidirectional)
- **Rich content:** Embed support for structured responses (status reports, diagnostics)

### Env Vars

```
DISCORD_BOT_TOKEN=...
DISCORD_GUILD_ID=...               # Optional: restrict to one server
DISCORD_CHANNEL_ID=...             # Default channel
```

---

## Phase 3 — Email Channel (~300 LOC)

**File:** `src/maxim/comms/channels/email_channel.py`

### Implementation

```python
class EmailChannel(Channel):
    name = "email"
    supports_media = True
    supports_threading = True

    def __init__(self, smtp_host: str, smtp_port: int, imap_host: str,
                 username: str, password: str, from_address: str) -> None: ...
```

- **Outbound:** SMTP (with STARTTLS)
- **Inbound:** IMAP IDLE for push notifications (long-poll fallback)
- **Threading:** RFC 2822 `Message-ID` / `In-Reply-To` / `References` headers
- **Conversation timing:** Override expiry to 7 days (email cadence)
- **Response match window:** 24 hours (vs 5 min for SMS)
- **Attachments:** Standard MIME multipart
- **Dependency:** `aiosmtplib>=3.0.0` (optional extra: `comms-email`), `aioimaplib>=1.0.0`
- **HTML rendering:** Plain text primary, optional HTML for formatted responses

### Env Vars

```
MAXIM_EMAIL_SMTP_HOST=smtp.gmail.com
MAXIM_EMAIL_SMTP_PORT=587
MAXIM_EMAIL_IMAP_HOST=imap.gmail.com
MAXIM_EMAIL_USERNAME=...
MAXIM_EMAIL_PASSWORD=...              # App password for Gmail
MAXIM_EMAIL_FROM=maxim@yourdomain.com
```

### Special Considerations

- Email is slow. `CommunicationBridge` response attribution needs the per-channel timeout config from Phase 0b.
- Inbound email polling runs in a daemon thread (like the current FastAPI server).
- Subject line extraction: first inbound email subject becomes conversation topic.

---

## Phase 4 — WebSocket Channel (~200 LOC)

**File:** `src/maxim/comms/channels/websocket_channel.py`

### Implementation

```python
class WebSocketChannel(Channel):
    name = "websocket"
    supports_realtime = True
    supports_media = True

    def __init__(self, host: str = "127.0.0.1", port: int = 5001) -> None: ...
```

- **Purpose:** Browser-based chat UI, programmatic clients, inter-agent communication
- **Protocol:** JSON messages over WebSocket (`{"type": "message", "content": "...", "sender": "..."}`)
- **Session management:** Connection ID = conversation ID. Reconnect with `session_id` to resume.
- **Heartbeat:** Ping/pong every 30s, disconnect after 90s silence
- **Server:** Reuse the existing FastAPI app with `WebSocket` endpoint at `/ws`
- **Dependency:** Already satisfied by `fastapi` + `uvicorn` (WebSocket support built-in)
- **Multi-client:** Support multiple concurrent WebSocket connections, each as separate conversation

### API Endpoint

```
ws://localhost:5001/ws?session_id=<optional>
```

### Use Cases

- Web dashboard for Maxim (chat interface)
- Claude Code or other agents talking to Maxim programmatically
- Mobile app integration
- Real-time status streaming (subscribe to bus events)

---

## Phase 5 — Audio Channel (Local + Remote) (~350 LOC)

**File:** `src/maxim/comms/channels/audio_channel.py`

This is the most architecturally interesting phase. Maxim already has Piper TTS, Whisper STT, and the Reachy audio stream — but none of it is wired into the comms stack. This phase unifies voice interaction as a first-class channel.

### Implementation

```python
class AudioChannel(Channel):
    name = "audio"
    supports_voice = True
    supports_realtime = True

    def __init__(self, tts_engine: TTSEngine | None = None,
                 transcriber: WhisperTranscriber | None = None,
                 audio_stream: AudioStream | None = None,
                 speaker_fn: Callable | None = None) -> None: ...

    def send(self, recipient: str, body: str, **kw) -> SendResult:
        """Synthesize text and play through speaker."""
        samples = self.tts_engine.synthesize(body)
        self.speaker_fn(samples, sample_rate=16000)
        return SendResult(success=True)

    def receive(self, raw: dict) -> InboundMessage:
        """Called when Whisper produces a transcript chunk."""
        return InboundMessage(
            sender="local_user",
            content=raw["transcript"],
            channel="audio",
        )
```

### Architecture

```
Microphone → AudioStream.get_sample()
  → Whisper STT → transcript chunk
    → AudioChannel.receive() → InboundMessage
      → Gateway.receive_inbound() → Percept on bus
        → Agent pipeline processes
          → Agent calls send_message(channel="audio", ...)
            → AudioChannel.send() → TTS → Speaker
```

### What Changes

- **`media_loop.py` integration:** The existing audio capture loop in `conscience/media_loop.py` currently writes transcript chunks directly. Instead, it should also push them through `AudioChannel.receive()` so conversations are tracked.
- **`ResponseOutput` bridging:** When audio is enabled AND the audio channel is registered, `respond` and `speak` tools should route through the channel (so CommunicationBridge can track them for NAc learning).
- **ConversationManager:** Audio conversations use voice-activity-based boundaries instead of time-based expiry. A new `AudioConversationPolicy` handles this: conversation ends after 60s of silence, not 30 min.

### Modes

1. **Local audio** (default): Microphone + speaker on the machine running Maxim
2. **Reachy audio**: Hardware microphone + speaker via Reachy SDK (already works)
3. **Remote audio**: WebSocket-based audio streaming (Phase 4 + Phase 5 combined — browser mic → WebSocket → Whisper → agent → TTS → WebSocket → browser speaker)

### Env Vars

```
MAXIM_AUDIO_CHANNEL_ENABLED=1      # Register audio as a comms channel
MAXIM_AUDIO_VAD_SILENCE_MS=1500    # Silence threshold for turn detection
```

---

## Phase 6 — Intelligent Channel Router (~200 LOC)

**File:** `src/maxim/comms/router.py`

Once multiple channels exist, the agent shouldn't need to specify `channel="twilio"` every time. The router picks the best channel based on context.

### Implementation

```python
class ChannelRouter:
    def __init__(self, gateway: CommunicationGateway) -> None: ...

    def resolve(self, recipient: str, *, prefer: str | None = None,
                urgency: str = "normal") -> tuple[str, str]:
        """Return (channel_name, resolved_recipient)."""
```

### Routing Logic

1. **Explicit preference:** If `prefer="slack"`, use Slack if available
2. **Recipient format detection:**
   - `+1234567890` → Twilio SMS
   - `@username` or `U12345` → Slack
   - `user@domain.com` → Email
   - `discord:user#1234` → Discord
   - `local` or `audio` → Audio channel
3. **Urgency escalation:**
   - `"critical"` → Voice call (if available) → SMS → Slack → Email
   - `"normal"` → Last-used channel for this recipient → SMS → Slack
   - `"low"` → Email → Slack → SMS
4. **Contact registry:** Maintain a `contacts.json` mapping names → channel+address pairs. Agent can use `add_contact` tool to build this over time.

### Updated Tool

Replace `SendMessageTool`'s required `channel` param with optional:

```python
class SendMessageTool(Tool):
    input_schema = {
        "recipient": str,        # Required
        "body": str,             # Required
        "channel": str | None,   # Optional — router picks if omitted
        "urgency": str,          # Optional — "normal", "critical", "low"
    }
```

### Contact Tool

```python
class ManageContactTool(Tool):
    name = "manage_contact"
    # add_contact(name, channel, address)
    # list_contacts()
    # Persisted in data/util/contacts.json via atomic_write_json
```

---

## Phase 7 — Conversation Context in Agent Prompt (~150 LOC)

**File:** `src/maxim/comms/conversation.py` + agent prompt building

Currently the agent doesn't see conversation history in its system prompt. It gets the raw Percept but no threading context. Fix this so multi-turn conversations feel natural.

### Implementation

- `ConversationManager.get_context_for_prompt()` already exists but isn't called during prompt assembly
- Wire it into `MemoryAgent` or `ExecAgent` context building: when the current percept is `comms:*`, inject the last N messages from that conversation into the structured context
- Include: participant name/number, channel, message count, conversation age
- Cap at 10 messages or 2000 tokens (whichever is smaller) to avoid prompt bloat

### Conversation-Aware Goal Setting

When a comms percept arrives, the `ExecAgent` should see:
```
[Active conversation with +1234567890 via SMS, 4 messages, started 3m ago]
User: What's your current goal?
User: And how's memory consolidation going?
Maxim: I'm currently exploring the room and mapping spatial landmarks.
User: Can you check if the door is locked?
```

This lets the agent maintain coherent multi-turn dialogue instead of treating each message as isolated.

---

## Phase 8 — Communication in Default Network (~100 LOC)

**File:** `src/maxim/default_network/behaviors/communication.py`

Add a reactive communication behavior to the Default Network so Maxim can initiate contact without explicit goals.

### Behaviors

1. **ProactiveStatusUpdate:** If a user has asked for status in the last hour and significant state change occurs (mode switch, goal completion, error), proactively send an update via the channel they last used.
2. **ConversationFollowUp:** If a conversation was left mid-thread (user asked a question, Maxim responded, user went silent), after 10 minutes generate a brief follow-up ("Let me know if you need anything else").
3. **UrgentEscalation:** If the harm system or pain detector fires above threshold, notify registered emergency contacts via the highest-urgency channel.

These are reactive (DN handles them without LLM calls) and gated by the ThalamicGate's escalation thresholds.

---

## Phase 9 — Notification Preferences & Quiet Hours (~100 LOC)

**File:** `src/maxim/comms/preferences.py`

### Implementation

```python
@dataclass
class ContactPreferences:
    quiet_hours: tuple[int, int] | None = None    # (22, 7) = 10pm-7am
    preferred_channel: str | None = None
    max_messages_per_hour: int = 10
    allow_voice_calls: bool = False
    allow_proactive: bool = True                   # DN-initiated messages
```

- Persisted in `data/util/contact_preferences.json`
- SCN (temporal rhythm) integration: align quiet hours with the circadian model
- Agent can update via `manage_contact` tool
- All outbound messages check preferences before sending

---

## Dependency Summary

| Phase | New Optional Extras | Libraries |
|-------|-------------------|-----------|
| 0 | — | — (refactor only) |
| 1 | `comms-slack` | `slack-sdk>=3.27.0` |
| 2 | `comms-discord` | `discord.py>=2.3.0` |
| 3 | `comms-email` | `aiosmtplib>=3.0.0`, `aioimaplib>=1.0.0` |
| 4 | — | Already have `fastapi`, `uvicorn` |
| 5 | — | Already have `piper-tts`, `faster-whisper` |
| 6-9 | — | No new deps |

## Phase Sequence

```
Phase 0 (foundation) ──→ Phase 1 (Slack)
                    ├──→ Phase 2 (Discord)     } can parallelize
                    ├──→ Phase 3 (Email)       }
                    └──→ Phase 4 (WebSocket) ──→ Phase 5 (Audio)
                                                     ↓
                                              Phase 6 (Router)
                                                     ↓
                                              Phase 7 (Context)
                                                     ↓
                                              Phase 8 (DN behaviors)
                                                     ↓
                                              Phase 9 (Preferences)
```

Phases 1-4 are independent after Phase 0 and can be done in any order. Phase 5 benefits from Phase 4 (WebSocket audio streaming). Phases 6-9 build on having multiple channels available.

## Estimated Scope

~1900 LOC total across all phases. Each phase is independently shippable and testable. Phase 0 is the only prerequisite for everything else.

## Files Touched

| File | Phase | Change |
|------|-------|--------|
| `comms/channels/base.py` | 0 | New ABC methods, dataclasses |
| `comms/gateway.py` | 0 | Accept InboundMessage, remove hard-coded values |
| `comms/conversation.py` | 0, 7 | Configurable expiry, prompt context wiring |
| `comms/api.py` | 1, 2, 4 | New webhook endpoints |
| `comms/channels/twilio_channel.py` | 0 | Implement new ABC |
| `comms/channels/slack_channel.py` | 1 | New file |
| `comms/channels/discord_channel.py` | 2 | New file |
| `comms/channels/email_channel.py` | 3 | New file |
| `comms/channels/websocket_channel.py` | 4 | New file |
| `comms/channels/audio_channel.py` | 5 | New file |
| `comms/router.py` | 6 | New file |
| `comms/preferences.py` | 9 | New file |
| `tools/comms.py` | 6 | Optional channel param, ManageContactTool |
| `bridges/communication_bridge.py` | 0 | Per-channel timing config |
| `runtime/bootstrap.py` | 1-5 | Register new channels |
| `conscience/media_loop.py` | 5 | Route transcripts through AudioChannel |
| `default_network/behaviors/communication.py` | 8 | New file |
| `pyproject.toml` | 1-3 | New optional extras |

## Testing Strategy

- Each channel gets a mock test (no real API calls) verifying `send()` → `SendResult` and `receive(raw)` → `InboundMessage`
- Integration test: register channel → gateway.receive_inbound → verify Percept on bus with correct fields
- Router test: verify recipient format detection and urgency escalation logic
- Conversation context test: verify prompt injection with multi-turn history
- CommunicationBridge test: verify per-channel timeout config is respected
- Audio channel: mock AudioStream + TTS, verify send→synthesize→speak flow
