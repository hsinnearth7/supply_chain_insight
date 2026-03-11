"""Role-Based Access Control (RBAC) for ChainInsight.

Implements a role/permission system with middleware enforcement for
enterprise multi-tenant deployments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------

class Permission(str, Enum):
    """Granular permissions for ChainInsight operations."""
    READ_DASHBOARD = "read:dashboard"
    READ_ANALYSIS = "read:analysis"
    TRIGGER_PIPELINE = "trigger:pipeline"
    UPLOAD_DATA = "upload:data"
    MANAGE_USERS = "manage:users"
    READ_MODELS = "read:models"
    DEPLOY_MODELS = "deploy:models"
    READ_AUDIT = "read:audit"
    MANAGE_CONFIG = "manage:config"


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

class Role(str, Enum):
    """User roles with hierarchical permission sets."""
    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"


# Role -> Permissions mapping
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.VIEWER: {
        Permission.READ_DASHBOARD,
        Permission.READ_ANALYSIS,
        Permission.READ_MODELS,
    },
    Role.OPERATOR: {
        Permission.READ_DASHBOARD,
        Permission.READ_ANALYSIS,
        Permission.READ_MODELS,
        Permission.TRIGGER_PIPELINE,
        Permission.UPLOAD_DATA,
        Permission.DEPLOY_MODELS,
    },
    Role.ADMIN: set(Permission),  # All permissions
}


# ---------------------------------------------------------------------------
# User model
# ---------------------------------------------------------------------------

@dataclass
class User:
    """Authenticated user with role-based permissions."""
    id: str
    username: str
    role: Role
    email: str = ""
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def permissions(self) -> set[Permission]:
        """Get the set of permissions for this user's role."""
        return ROLE_PERMISSIONS.get(self.role, set())

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions

    def has_any_permission(self, *permissions: Permission) -> bool:
        """Check if user has any of the given permissions."""
        return bool(self.permissions & set(permissions))

    def has_all_permissions(self, *permissions: Permission) -> bool:
        """Check if user has all of the given permissions."""
        return set(permissions).issubset(self.permissions)


# ---------------------------------------------------------------------------
# User store (in-memory for now; swap with DB in production)
# ---------------------------------------------------------------------------

# Default users for development
_user_store: dict[str, User] = {
    "dev-key-change-me": User(
        id="user-001",
        username="admin",
        role=Role.ADMIN,
        email="admin@chaininsight.dev",
    ),
}


def get_user_by_api_key(api_key: str) -> User | None:
    """Look up a user by their API key."""
    return _user_store.get(api_key)


def register_api_key(api_key: str, user: User) -> None:
    """Register a new API key -> user mapping."""
    _user_store[api_key] = user


# ---------------------------------------------------------------------------
# Permission checking helpers
# ---------------------------------------------------------------------------

def require_permission(*required: Permission):
    """FastAPI dependency factory that checks permissions.

    Usage:
        @router.post("/pipeline/run", dependencies=[Depends(require_permission(Permission.TRIGGER_PIPELINE))])
    """
    async def _checker(request: Request):
        user: User | None = getattr(request.state, "user", None)
        if user is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        if not user.is_active:
            raise HTTPException(status_code=403, detail="User account is disabled")
        if not user.has_all_permissions(*required):
            missing = set(required) - user.permissions
            raise HTTPException(
                status_code=403,
                detail=f"Missing permissions: {', '.join(p.value for p in missing)}",
            )
        return user

    return _checker


# ---------------------------------------------------------------------------
# RBAC Middleware
# ---------------------------------------------------------------------------

# Paths that do not require authentication
PUBLIC_PATHS = {
    "/api/health",
    "/docs",
    "/openapi.json",
    "/redoc",
}


class RBACMiddleware(BaseHTTPMiddleware):
    """Middleware that resolves the user from the API key and attaches to request state.

    This middleware runs before route handlers, extracting the X-API-Key header
    and resolving it to a User object with role-based permissions.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        path = request.url.path

        # Skip auth for public paths
        if path in PUBLIC_PATHS or not path.startswith("/api"):
            return await call_next(request)

        # Extract API key
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return Response(
                content='{"detail":"Missing API key"}',
                status_code=401,
                media_type="application/json",
            )

        # Resolve user
        user = get_user_by_api_key(api_key)
        if user is None:
            return Response(
                content='{"detail":"Invalid API key"}',
                status_code=403,
                media_type="application/json",
            )

        if not user.is_active:
            return Response(
                content='{"detail":"User account is disabled"}',
                status_code=403,
                media_type="application/json",
            )

        # Attach user to request state for downstream handlers
        request.state.user = user

        logger.debug(
            "RBAC: user=%s role=%s path=%s method=%s",
            user.username,
            user.role.value,
            path,
            request.method,
        )

        return await call_next(request)
