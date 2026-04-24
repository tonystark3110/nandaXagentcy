"""
SLIM Client for Exchange Agent
Calls agents via A2A SSE streaming using synchronous requests in a thread pool.

The asyncio event loop is never blocked by SSE streaming — each call runs in
a dedicated thread via asyncio.to_thread(), using the synchronous `requests`
library. This prevents all connection-pool and event-loop-stall issues that
arise when breaking out of async SSE streams early.
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, Optional
from uuid import uuid4

import requests as req_lib

logger = logging.getLogger(__name__)

# All endpoints come from environment variables — no hardcoded IPs in source.
# Set in supervisor/docker env:
#   AGENT_HOST          = IP of the agents server  (alerts/planner/stopfinder)
#   SLIM_ALERTS_PORT    = A2A port for alerts       (default 50051)
#   SLIM_PLANNER_PORT   = A2A port for planner      (default 50052)
#   SLIM_STOPFINDER_PORT= A2A port for stopfinder   (default 50053)
# Fares endpoints are resolved dynamically via ANS — no static entry needed.
_AGENT_HOST          = os.getenv("AGENT_HOST", "")
_ALERTS_PORT         = os.getenv("SLIM_ALERTS_PORT",     "50051")
_PLANNER_PORT        = os.getenv("SLIM_PLANNER_PORT",    "50052")
_STOPFINDER_PORT     = os.getenv("SLIM_STOPFINDER_PORT", "50053")

def _ep(host: str, port: str) -> str:
    return f"http://{host}:{port}" if host else ""

AGENT_ENDPOINTS: Dict[str, str] = {
    "alerts":     _ep(_AGENT_HOST, _ALERTS_PORT),
    "planner":    _ep(_AGENT_HOST, _PLANNER_PORT),
    "stopfinder": _ep(_AGENT_HOST, _STOPFINDER_PORT),
    "fares":      "",  # always resolved via ANS at call time
}

# Map slim name → registry agent_id (used by synthesize_node)
AGENT_ID_MAP: Dict[str, str] = {
    "alerts":     "mbta-alerts",
    "planner":    "mbta-planner",
    "stopfinder": "mbta-stopfinder",
    "fares":      "mbta-fares",
}


def _build_a2a_request(message: str) -> dict:
    """Build a JSON-RPC A2A streaming request using the a2a SDK types for
    correct serialisation (handles field names, enums, etc.)."""
    try:
        from a2a.types import (
            SendStreamingMessageRequest,
            MessageSendParams,
            Message,
            TextPart,
        )
        request = SendStreamingMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(
                message=Message(
                    messageId=str(uuid4()),
                    role="user",
                    parts=[TextPart(text=message)],
                )
            ),
        )
        return request.model_dump(mode="json", exclude_none=True)
    except Exception:
        # Fallback: hand-build the JSON-RPC body
        return {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": "message/stream",
            "params": {
                "message": {
                    "messageId": str(uuid4()),
                    "role": "user",
                    "parts": [{"kind": "text", "text": message}],
                }
            },
        }


def _extract_text_from_event(result: Any) -> str:
    """Extract plain text from an A2A result payload (dict or SDK object)."""
    if isinstance(result, str):
        return result

    if not isinstance(result, dict):
        # SDK object — try attribute access
        for attr in ("parts", "text"):
            val = getattr(result, attr, None)
            if val and isinstance(val, str):
                return val
        return ""

    kind = result.get("kind", "")

    # Message
    if kind == "message" or ("parts" in result and "role" in result):
        for part in result.get("parts", []):
            text = _text_from_part(part)
            if text:
                return text

    # Task
    if kind == "task":
        status = result.get("status", {})
        msg = status.get("message", {})
        if isinstance(msg, dict):
            for part in msg.get("parts", []):
                text = _text_from_part(part)
                if text:
                    return text
        for artifact in result.get("artifacts", []):
            if isinstance(artifact, dict):
                for part in artifact.get("parts", []):
                    text = _text_from_part(part)
                    if text:
                        return text

    # TaskStatusUpdateEvent
    if kind == "status-update":
        status = result.get("status", {})
        msg = status.get("message", {})
        if isinstance(msg, dict):
            for part in msg.get("parts", []):
                text = _text_from_part(part)
                if text:
                    return text

    return ""


def _text_from_part(part: Any) -> str:
    """Extract text from a Part dict or SDK Part object."""
    if isinstance(part, dict):
        # Direct text field or nested root
        text = part.get("text", "")
        if text:
            return text
        root = part.get("root", {})
        if isinstance(root, dict):
            return root.get("text", "")
    else:
        # SDK Part object
        inner = getattr(part, "root", part)
        return getattr(inner, "text", "") or ""
    return ""


def _sync_call_agent(base_url: str, request_body: dict) -> str:
    """Synchronous SSE call — runs in a thread pool, never blocks the event loop.

    Sends the JSON-RPC A2A request, reads SSE lines until text is found,
    then immediately closes the connection (os-level RST via resp.close()).
    """
    try:
        with req_lib.post(
            base_url,
            json=request_body,
            stream=True,
            timeout=(10, 60),          # (connect_timeout, read_timeout)
            headers={
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
            },
        ) as resp:
            resp.raise_for_status()

            for raw_line in resp.iter_lines(chunk_size=256):
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue

                data_str = line[6:].strip()
                if not data_str or data_str == "[DONE]":
                    continue

                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # JSON-RPC envelope: {"jsonrpc": "2.0", "id": ..., "result": {...}}
                result = event.get("result", event)
                text = _extract_text_from_event(result)
                if text:
                    # Hard-close: don't drain remaining SSE events
                    resp.close()
                    return text

    except req_lib.exceptions.RequestException as e:
        raise RuntimeError(f"A2A HTTP request failed: {e}") from e

    return ""


class SlimAgentClient:
    """Client for calling A2A/SLIM agents via SSE streaming.

    All SSE I/O runs synchronously in a thread pool via asyncio.to_thread().
    The asyncio event loop is never touched by SSE reads or connection cleanup,
    eliminating event-loop stalls that occur when async SSE streams are
    interrupted.
    """

    def __init__(self):
        self._initialized = True

    async def initialize(self):
        """No-op — no persistent resources needed."""
        pass

    async def call_agent(
        self,
        agent_name: str,
        message: str,
        base_url: Optional[str] = None,   # ANS-resolved URL overrides static map
    ) -> Dict[str, Any]:
        """Call an agent via A2A SSE streaming (thread-pool isolated).

        If `base_url` is provided (e.g. from ANS resolution) it is used directly,
        bypassing the static AGENT_ENDPOINTS map.  This lets the ANS system
        dynamically pick the best endpoint (Boston vs Frankfurt) on every call.
        """
        if not base_url:
            base_url = AGENT_ENDPOINTS.get(agent_name)
        if not base_url:
            raise ValueError(f"Unknown agent: {agent_name}")

        logger.info(f"📤 Calling {agent_name} via A2A SSE thread ({base_url})...")

        request_body = _build_a2a_request(message)

        try:
            # Run synchronous blocking SSE call in the thread pool.
            # asyncio.to_thread() returns when the thread finishes — the event
            # loop is free to handle other requests the entire time.
            response_text: str = await asyncio.to_thread(
                _sync_call_agent, base_url, request_body
            )
        except Exception as e:
            logger.error(f"❌ A2A call to {agent_name} failed: {e}", exc_info=True)
            raise

        if not response_text:
            response_text = f"Agent {agent_name} returned no text response"

        logger.info(f"✅ A2A SSE thread SUCCESS for {agent_name}: {len(response_text)} chars")

        return {
            "response": response_text,
            "agent_used": AGENT_ID_MAP.get(agent_name, agent_name),
            "metadata": {
                "transport": "slim",
                "agent": agent_name,
            },
        }

    async def cleanup(self):
        """No-op — no persistent resources to release."""
        logger.info("✅ SlimAgentClient cleanup (stateless)")
