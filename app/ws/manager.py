"""WebSocket ConnectionManager — tracks connections and broadcasts messages."""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections: global subscribers and per-batch_id subscribers."""

    def __init__(self):
        self._global: list[WebSocket] = []
        self._batch: dict[str, list[WebSocket]] = {}

    def add(self, ws: WebSocket, batch_id: Optional[str] = None):
        """Register an already-accepted WebSocket connection."""
        if batch_id:
            self._batch.setdefault(batch_id, []).append(ws)
            logger.info("WS connected to batch %s (total=%d)", batch_id, len(self._batch[batch_id]))
        else:
            self._global.append(ws)
            logger.info("WS connected globally (total=%d)", len(self._global))

    def disconnect(self, ws: WebSocket, batch_id: Optional[str] = None):
        """Remove a WebSocket connection."""
        if batch_id and batch_id in self._batch:
            self._batch[batch_id] = [c for c in self._batch[batch_id] if c is not ws]
            if not self._batch[batch_id]:
                del self._batch[batch_id]
        else:
            self._global = [c for c in self._global if c is not ws]

    async def broadcast_to_batch(self, batch_id: str, message: dict):
        """Send a message to all connections subscribed to a specific batch."""
        message.setdefault("batch_id", batch_id)
        message.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        data = json.dumps(message)
        dead: list[WebSocket] = []
        for ws in self._batch.get(batch_id, []):
            try:
                await ws.send_text(data)
            except Exception as exc:
                logger.warning("Failed to send WS message to batch %s: %s", batch_id, exc)
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws, batch_id)

    async def broadcast_global(self, message: dict):
        """Send a message to all global subscribers."""
        message.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        data = json.dumps(message)
        dead: list[WebSocket] = []
        for ws in self._global:
            try:
                await ws.send_text(data)
            except Exception as exc:
                logger.warning("Failed to send global WS message: %s", exc)
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def build_message(
        self,
        msg_type: str,
        batch_id: str = "",
        stage: str = "",
        status: str = "",
        progress_pct: int = 0,
        data: Optional[dict[str, Any]] = None,
    ) -> dict:
        """Build a standardized WS message."""
        return {
            "type": msg_type,
            "batch_id": batch_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "stage": stage,
                "status": status,
                "progress_pct": progress_pct,
                "data": data or {},
            },
        }


# Module-level singleton
manager = ConnectionManager()
