"""
FastAPI Dependencies
Common dependencies for dependency injection
"""

from typing import Optional
from fastapi import Depends, HTTPException, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config.settings import get_settings
from services.auth_service import AuthService

# Initialize services
settings = get_settings()
security = HTTPBearer(auto_error=False)

async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[dict]:
    """Get current user if authenticated (optional)"""
    if not settings.enable_auth:
        return None

    if not credentials:
        return None

    try:
        auth_service = AuthService()
        user = await auth_service.verify_token(credentials.credentials)
        return user
    except Exception:
        return None

async def get_current_user_required(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """Get current user (required)"""
    if not settings.enable_auth:
        return {"uid": "anonymous", "email": "anonymous@example.com"}

    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        auth_service = AuthService()
        user = await auth_service.verify_token(credentials.credentials)
        return user
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid authentication: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_api_key(
    x_api_key: Optional[str] = Header(None)
) -> Optional[str]:
    """Get API key from header"""
    return x_api_key

async def verify_api_key(
    api_key: Optional[str] = Depends(get_api_key)
) -> bool:
    """Verify API key"""
    if not settings.api_key_required:
        return True

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required"
        )

    # In production, check against database or external service
    valid_keys = settings.valid_api_keys  # List from settings
    if api_key not in valid_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    return True
