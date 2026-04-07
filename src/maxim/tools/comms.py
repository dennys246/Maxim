"""Communication tools — send_message and call_user.

These tools let the agent send messages and make calls through the
CommunicationGateway. The agent decides when/what/to whom; the gateway
handles transport.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from maxim.tools.base import Tool, ToolOutput

if TYPE_CHECKING:
    from maxim.comms.gateway import CommunicationGateway


class SendMessageTool(Tool):
    """Send a message to a user via a communication channel.

    Use this tool to proactively notify the user about completed goals,
    important events, errors, or to reply to their messages.
    """

    name = "send_message"
    description = (
        "Send a text message to a user via SMS, webhook, or other channel. "
        "Use to notify about events, answer questions, or reply to conversations."
    )
    input_schema = {
        "channel": str,  # Required: "twilio", "webhook", etc.
        "recipient": str,  # Required: phone number, user ID, etc.
        "body": str,  # Required: message text
    }

    def __init__(self, gateway: "CommunicationGateway") -> None:
        super().__init__()
        self._gateway = gateway

    def execute(
        self,
        channel: str,
        recipient: str,
        body: str,
        **kwargs: Any,
    ) -> ToolOutput:
        if not body or not body.strip():
            return ToolOutput(success=False, error="Message body cannot be empty")
        if not channel:
            return ToolOutput(success=False, error="Channel is required")
        if not recipient:
            return ToolOutput(success=False, error="Recipient is required")

        success = self._gateway.send(channel, recipient, body)
        if success:
            return ToolOutput(
                success=True,
                output={
                    "delivered": True,
                    "channel": channel,
                    "recipient": recipient,
                    "body_length": len(body),
                },
            )
        return ToolOutput(
            success=False,
            error=f"Failed to send message via {channel} to {recipient}",
        )


class CallUserTool(Tool):
    """Initiate an outbound voice call to the user.

    Use this tool for urgent notifications that require immediate
    attention, or when the user has requested a voice call.
    """

    name = "call_user"
    description = (
        "Initiate an outbound voice call with text-to-speech. "
        "Use for urgent notifications that require immediate attention."
    )
    input_schema = {
        "recipient": str,  # Required: phone number
        "message": str,  # Required: text to speak
    }

    def __init__(self, gateway: "CommunicationGateway") -> None:
        super().__init__()
        self._gateway = gateway

    def execute(
        self,
        recipient: str,
        message: str,
        **kwargs: Any,
    ) -> ToolOutput:
        if not message or not message.strip():
            return ToolOutput(success=False, error="Call message cannot be empty")
        if not recipient:
            return ToolOutput(success=False, error="Recipient is required")

        success = self._gateway.call(recipient, message)
        if success:
            return ToolOutput(
                success=True,
                output={"called": True, "recipient": recipient},
            )
        return ToolOutput(
            success=False,
            error=f"Failed to initiate call to {recipient}",
        )
