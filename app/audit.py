"""Audit logging for ChainInsight.

Provides structured audit trail for all API operations, including
user identity, action, resource, and outcome. Supports both in-memory
and persistent (database) backends.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audit event model
# ---------------------------------------------------------------------------

@dataclass
class AuditEvent:
    """A single audit log entry."""

    event_id: str
    timestamp: str
    user_id: str
    username: str
    role: str
    action: str  # HTTP method
    resource: str  # Request path
    status_code: int
    ip_address: str
    user_agent: str
    duration_ms: float
    request_body: str | None = None
    response_summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


# ---------------------------------------------------------------------------
# Audit logger
# ---------------------------------------------------------------------------

class AuditLogger:
    """Centralized audit logger with pluggable storage backends.

    Default backend: in-memory ring buffer (last 10,000 events).
    Production: override with database or external logging service.
    """

    def __init__(self, max_memory_events: int = 10_000) -> None:
        self._events: deque[AuditEvent] = deque(maxlen=max_memory_events)
        self._logger = logging.getLogger("chaininsight.audit")

    def log(self, event: AuditEvent) -> None:
        """Record an audit event."""
        self._events.append(event)

        # Also emit to structured logger for external ingestion
        self._logger.info(
            "AUDIT action=%s resource=%s user=%s status=%d duration_ms=%.1f",
            event.action,
            event.resource,
            event.username,
            event.status_code,
            event.duration_ms,
        )

    def query(
        self,
        user_id: str | None = None,
        action: str | None = None,
        resource_prefix: str | None = None,
        min_status: int | None = None,
        max_status: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query audit events with filters.

        Args:
            user_id: Filter by user ID.
            action: Filter by HTTP method.
            resource_prefix: Filter by resource path prefix.
            min_status: Minimum HTTP status code.
            max_status: Maximum HTTP status code.
            limit: Maximum results to return.
            offset: Number of results to skip.

        Returns:
            List of matching audit events as dictionaries.
        """
        results = []

        for event in reversed(self._events):
            if user_id and event.user_id != user_id:
                continue
            if action and event.action != action:
                continue
            if resource_prefix and not event.resource.startswith(resource_prefix):
                continue
            if min_status and event.status_code < min_status:
                continue
            if max_status and event.status_code > max_status:
                continue
            results.append(event.to_dict())

        return results[offset : offset + limit]

    def get_recent(self, count: int = 50) -> list[dict[str, Any]]:
        """Get the most recent audit events."""
        return [e.to_dict() for e in list(self._events)[-count:]]

    def get_stats(self) -> dict[str, Any]:
        """Get aggregate audit statistics."""
        if not self._events:
            return {
                "total_events": 0,
                "unique_users": 0,
                "error_count": 0,
                "avg_duration_ms": 0,
            }

        events_list = list(self._events)
        error_count = sum(1 for e in events_list if e.status_code >= 400)
        unique_users = len({e.user_id for e in events_list})
        avg_duration = sum(e.duration_ms for e in events_list) / len(events_list)

        return {
            "total_events": len(events_list),
            "unique_users": unique_users,
            "error_count": error_count,
            "avg_duration_ms": round(avg_duration, 2),
            "earliest": events_list[0].timestamp if events_list else None,
            "latest": events_list[-1].timestamp if events_list else None,
        }

    @property
    def event_count(self) -> int:
        return len(self._events)


# Singleton instance
audit_logger = AuditLogger()


# ---------------------------------------------------------------------------
# Audit middleware
# ---------------------------------------------------------------------------

# Paths to skip auditing (high-frequency, low-value)
SKIP_AUDIT_PATHS = {
    "/api/health",
    "/api/metrics",
    "/docs",
    "/openapi.json",
    "/redoc",
}

# Sensitive paths where request body should not be logged
SENSITIVE_PATHS = {
    "/api/auth",
    "/api/users",
}


class AuditMiddleware(BaseHTTPMiddleware):
    """Middleware that creates audit log entries for all API requests.

    Captures user identity (from request.state.user set by RBACMiddleware),
    request details, response status, and timing.
    """

    def __init__(self, app: Any, audit: AuditLogger | None = None) -> None:
        super().__init__(app)
        self._audit = audit or audit_logger

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        path = request.url.path

        # Skip non-API and high-frequency paths
        if path in SKIP_AUDIT_PATHS or not path.startswith("/api"):
            return await call_next(request)

        start_time = time.perf_counter()

        # Extract user info (set by RBACMiddleware)
        user = getattr(request.state, "user", None)
        user_id = user.id if user else "anonymous"
        username = user.username if user else "anonymous"
        role = user.role.value if user else "none"

        # Read request body for non-sensitive paths
        request_body = None
        if request.method in ("POST", "PUT", "PATCH") and path not in SENSITIVE_PATHS:
            try:
                body_bytes = await request.body()
                if len(body_bytes) < 4096:  # Only log small bodies
                    request_body = body_bytes.decode("utf-8", errors="replace")
            except Exception:
                pass

        # Process request
        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Build audit event
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            username=username,
            role=role,
            action=request.method,
            resource=path,
            status_code=response.status_code,
            ip_address=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("User-Agent", "unknown"),
            duration_ms=round(duration_ms, 2),
            request_body=request_body,
            metadata={
                "query_params": dict(request.query_params),
            },
        )

        self._audit.log(event)

        return response
