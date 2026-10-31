import hmac
import logging
from typing import Optional

from fastapi import Header, HTTPException, Request, status

logger = logging.getLogger("churnguard.api")


def require_api_key(request: Request, x_api_key: Optional[str] = Header(default=None)):
    expected = request.app.state.settings.api_key
    supplied = x_api_key or ""
    if not supplied or not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        logger.warning(
            "authentication failed",
            extra={"event": {"path": request.url.path, "reason": "missing or invalid API key"}},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Send it in the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
