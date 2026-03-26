"""WebSocket routes for real-time pipeline progress and system events."""

import asyncio
import hmac
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import API_KEY
from app.ws.manager import manager

logger = logging.getLogger(__name__)
ws_router = APIRouter(tags=["websocket"])


def _check_ws_api_key(api_key: str | None) -> bool:
    """Validate WebSocket API key."""
    if api_key is None:
        return False
    return hmac.compare_digest(api_key, API_KEY)


async def _authenticate_ws(ws: WebSocket) -> bool:
    """Accept the WebSocket and wait for an auth message with the API key.

    The client must send ``{"type": "auth", "api_key": "<key>"}`` as its
    first message.  Returns True on success, closes with 4003 on failure.
    """
    await ws.accept()
    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
        msg = json.loads(raw)
        if msg.get("type") == "auth" and _check_ws_api_key(msg.get("api_key")):
            return True
    except asyncio.TimeoutError:
        await ws.close(code=4003, reason="Auth timeout")
        return False
    except WebSocketDisconnect:
        return False
    except Exception:
        try:
            await ws.close(code=4003, reason="Auth failed")
        except Exception:
            pass
        return False
    await ws.close(code=4003, reason="Invalid or missing API key")
    return False


@ws_router.websocket("/ws/pipeline/{batch_id}")
async def ws_pipeline(ws: WebSocket, batch_id: str):
    """Subscribe to real-time progress for a specific pipeline run."""
    if not await _authenticate_ws(ws):
        return
    manager.add(ws, batch_id=batch_id)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(ws, batch_id=batch_id)
        logger.info("WS disconnected from batch %s", batch_id)


@ws_router.websocket("/ws/global")
async def ws_global(ws: WebSocket):
    """Subscribe to system-wide events (watchdog detections, alerts)."""
    if not await _authenticate_ws(ws):
        return
    manager.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(ws)
        logger.info("Global WS disconnected")
