"""Inbound Command API — minimal HTTP server for external control.

Security is built in from the start:
- Bearer token auth for control endpoints
- Twilio HMAC signature verification for webhooks
- Rate limiting on every endpoint
- Bound to 127.0.0.1 only (use ngrok/Cloudflare Tunnel for HTTPS)
"""

from __future__ import annotations

import hmac
import logging
import os
import threading
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from maxim.agents.bus import AgentBus
    from maxim.comms.gateway import CommunicationGateway


def _load_env() -> None:
    """Load .env file if python-dotenv is available."""
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]

        load_dotenv()
    except ImportError:
        pass


def create_api_app(
    bus: "AgentBus",
    gateway: "CommunicationGateway",
    goal_agent: Any = None,
    autonomy_controller: Any = None,
    mode_controller: Any = None,
) -> Any:
    """Create the FastAPI application with all endpoints.

    Returns the app instance. Requires ``fastapi`` and ``uvicorn`` packages.
    """
    try:
        from fastapi import Depends, FastAPI, HTTPException, Request
    except ImportError as exc:
        raise ImportError(
            "fastapi package required for API server. "
            "Install with: pip install fastapi uvicorn"
        ) from exc

    _load_env()

    app = FastAPI(title="Maxim API", docs_url=None, redoc_url=None)

    # ── Rate limiting (optional) ──────────────────────────────────────────
    try:
        from slowapi import Limiter  # type: ignore[import-untyped]
        from slowapi.util import get_remote_address  # type: ignore[import-untyped]

        limiter = Limiter(key_func=get_remote_address)
        app.state.limiter = limiter
    except ImportError:
        limiter = None
        logger.warning("slowapi not installed — rate limiting disabled")

    # ── Authentication ────────────────────────────────────────────────────

    API_TOKEN = os.environ.get("MAXIM_API_TOKEN", "")

    async def require_api_token(request: Request) -> None:
        """Bearer token auth for all non-Twilio endpoints."""
        if not API_TOKEN:
            raise HTTPException(status_code=503, detail="API token not configured")
        token = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if not token or not hmac.compare_digest(token, API_TOKEN):
            raise HTTPException(status_code=401, detail="Invalid API token")

    async def require_twilio_signature(request: Request) -> None:
        """Twilio HMAC signature verification — prevents webhook spoofing."""
        twilio_auth = os.environ.get("TWILIO_AUTH_TOKEN", "")
        if not twilio_auth:
            raise HTTPException(status_code=503, detail="Twilio auth not configured")
        try:
            from twilio.request_validator import RequestValidator  # type: ignore[import-untyped]
        except ImportError:
            raise HTTPException(status_code=503, detail="twilio package not installed")

        validator = RequestValidator(twilio_auth)
        form = await request.form()
        signature = request.headers.get("X-Twilio-Signature", "")
        if not validator.validate(str(request.url), dict(form), signature):
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    # ── Twilio Webhooks ───────────────────────────────────────────────────

    @app.post("/api/twilio/sms", dependencies=[Depends(require_twilio_signature)])
    async def twilio_sms(request: Request) -> dict:
        form = await request.form()
        from_number = str(form.get("From", ""))
        body = str(form.get("Body", ""))
        twilio_ch = gateway.channels.get("twilio")
        if twilio_ch and hasattr(twilio_ch, "receive_sms"):
            twilio_ch.receive_sms(from_number, body)
        return {"status": "ok"}

    @app.post("/api/twilio/voice", dependencies=[Depends(require_twilio_signature)])
    async def twilio_voice(request: Request) -> dict:
        form = await request.form()
        from_number = str(form.get("From", ""))
        transcript = str(form.get("SpeechResult", ""))
        twilio_ch = gateway.channels.get("twilio")
        if twilio_ch and hasattr(twilio_ch, "receive_voice"):
            twilio_ch.receive_voice(from_number, transcript)
        return {"status": "ok"}

    # ── Control Endpoints ─────────────────────────────────────────────────

    @app.post("/api/command", dependencies=[Depends(require_api_token)])
    async def inject_command(request: Request) -> dict:
        data = await request.json()
        command = data.get("command", "")
        gateway.receive_inbound(channel="api", sender="api_user", text=command)
        return {"status": "accepted"}

    @app.get("/api/status", dependencies=[Depends(require_api_token)])
    async def get_status() -> dict:
        result: dict[str, Any] = {}
        if mode_controller:
            result["mode"] = getattr(mode_controller, "current_mode", "unknown")
            result["strategy"] = getattr(mode_controller, "current_strategy", "unknown")
        if goal_agent:
            current = getattr(goal_agent, "_current_goal", None)
            result["active_goal"] = current.description if current else None
        if autonomy_controller:
            level = getattr(autonomy_controller, "current_level", None)
            result["autonomy_level"] = level.value if level else "unknown"
        return result

    @app.get("/api/goals", dependencies=[Depends(require_api_token)])
    async def get_goals() -> dict:
        if not goal_agent:
            return {"goals": []}
        current = getattr(goal_agent, "_current_goal", None)
        return {
            "active": current.description if current else None,
        }

    return app


def start_api_server(
    bus: "AgentBus",
    gateway: "CommunicationGateway",
    goal_agent: Any = None,
    autonomy_controller: Any = None,
    mode_controller: Any = None,
    host: str = "127.0.0.1",
    port: int = 5000,
) -> threading.Thread:
    """Start the API server in a daemon thread.

    CRITICAL: Binds to localhost only. Use ngrok/Cloudflare Tunnel for
    HTTPS termination. NEVER expose this port directly to the internet.
    """
    try:
        import uvicorn  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "uvicorn package required for API server. "
            "Install with: pip install uvicorn"
        ) from exc

    app = create_api_app(
        bus=bus,
        gateway=gateway,
        goal_agent=goal_agent,
        autonomy_controller=autonomy_controller,
        mode_controller=mode_controller,
    )

    t = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": host, "port": port, "log_level": "warning"},
        daemon=True,
        name="maxim-api-server",
    )
    t.start()
    logger.info("API server started on %s:%d", host, port)
    return t
