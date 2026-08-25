"""
Falcon API Client for MCP Server

This module provides the Falcon API client and authentication utilities for the Falcon MCP server.
"""

import functools
import os
import platform
import sys
import threading
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from urllib.parse import urlparse

import anyio
import requests

# Import the APIHarnessV2 from FalconPy
from falconpy import APIHarnessV2, BaseURL  # type: ignore[import-untyped]

from falcon_mcp.common.logging import get_logger

# The MCP SDK sets this contextvar around each tool-call dispatch, carrying the
# HTTP request that triggered it. Multi-tenant credential resolution reads the
# tenant headers from there. Imported defensively because it is an SDK internal:
# if it ever moves, multi-tenant mode must fail loudly rather than silently serve
# the wrong tenant (see _request_headers).
try:
    from mcp.server.lowlevel.server import request_ctx
except ImportError:  # pragma: no cover - SDK layout changed
    request_ctx = None  # type: ignore[assignment]

logger = get_logger(__name__)


class FalconClient:
    """Client for interacting with the CrowdStrike Falcon API."""

    def __init__(
        self,
        base_url: str | None = None,
        debug: bool = False,
        user_agent_comment: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        member_cid: str | None = None,
        proxy: str | None = None,
        access_token: str | None = None,
        session: "requests.Session | None" = None,
    ):
        """Initialize the Falcon client.

        Args:
            base_url: Falcon API base URL (defaults to FALCON_BASE_URL env var)
            debug: Enable debug logging
            user_agent_comment: Additional information to include in the User-Agent comment section
            client_id: Falcon API Client ID (defaults to FALCON_CLIENT_ID env var)
            client_secret: Falcon API Client Secret (defaults to FALCON_CLIENT_SECRET env var)
            member_cid: Child CID for Flight Control (MSSP) support (defaults to FALCON_MEMBER_CID env var)
            proxy: HTTP/HTTPS proxy URL for outbound Falcon API connections (defaults to FALCON_PROXY_URL env var).
                   Example: "http://proxy.corp.example.com:8080"
            access_token: Pre-minted Falcon bearer token. Supplying this selects FalconPy's
                   TOKEN auth style: no /oauth2/token call is made and the client is
                   non-refreshable, so the caller owns token freshness. Used by
                   multi-tenant mode, where each request brings its own token.
            session: Existing requests.Session to reuse for connection pooling. FalconPy
                   never closes a session it is handed, so the owner controls its lifecycle.
        """
        # Get credentials from parameters or environment variables (parameters take precedence)
        self.client_id = client_id or os.environ.get("FALCON_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("FALCON_CLIENT_SECRET")
        self.base_url = base_url or os.environ.get(
            "FALCON_BASE_URL", "https://api.crowdstrike.com"
        )
        self.debug = debug
        self.user_agent_comment = user_agent_comment or os.environ.get(
            "FALCON_MCP_USER_AGENT_COMMENT"
        )
        self.member_cid = member_cid or os.environ.get("FALCON_MEMBER_CID")
        self.proxy = proxy or os.environ.get("FALCON_PROXY_URL")
        self.access_token = access_token

        if not self.access_token and (not self.client_id or not self.client_secret):
            raise ValueError(
                "Falcon API credentials not provided. Either pass client_id and client_secret "
                "parameters or set FALCON_CLIENT_ID and FALCON_CLIENT_SECRET environment variables."
            )

        # Build APIHarnessV2 initialization parameters.
        #
        # An injected access_token wins, and the credential pair is then deliberately
        # omitted rather than passed alongside it. FalconPy only honours access_token
        # when `cred_format_valid` is False, so handing it both would silently ignore
        # the token and authenticate as the credential owner instead — in multi-tenant
        # mode that means serving every tenant with one set of credentials.
        api_params: dict[str, Any] = {
            "base_url": self.base_url,
            "debug": debug,
            "user_agent": self.get_user_agent(),
        }
        if self.access_token:
            api_params["access_token"] = self.access_token
        else:
            api_params["client_id"] = self.client_id
            api_params["client_secret"] = self.client_secret

        # Only include session if provided; requires falconpy >= 1.6.5.
        if session is not None:
            api_params["session"] = session

        # Only include member_cid if it's provided
        if self.member_cid:
            api_params["member_cid"] = self.member_cid

        # Only include proxy if configured; APIHarnessV2 expects {"https": url}
        if self.proxy:
            api_params["proxy"] = {"https": self.proxy}

        # Initialize the Falcon API client using APIHarnessV2
        self.client = APIHarnessV2(**api_params)

        # Serializes the stale-token refresh path. Concurrent tool calls run their
        # blocking FalconPy work on separate threads (see command_async); without this
        # lock, several threads could observe a stale token at once and each fire its
        # own POST /oauth2/token. The lock guards only the refresh, never the API call.
        self._token_lock = threading.Lock()

        logger.debug("Initialized Falcon client with base URL: %s", self.base_url)
        if self.member_cid:
            logger.debug("Flight Control member_cid: %s", self.member_cid)

    @property
    def token_status(self) -> int | None:
        """HTTP status code from the last authentication attempt."""
        result: int | None = self.client.token_status
        return result

    @property
    def token_fail_reason(self) -> str | None:
        """Error message from the API when authentication failed."""
        result: str | None = self.client.token_fail_reason
        return result

    def auth_failure_message(self) -> str:
        """Build a diagnostic message after a failed authentication attempt."""
        parts = ["Failed to authenticate with the Falcon API"]
        if self.token_status:
            parts[0] += f" (HTTP {self.token_status})"
        if self.token_fail_reason:
            parts.append(self.token_fail_reason)

        if self.token_status == 401:
            parts.append(
                "Hint: Verify FALCON_CLIENT_ID and FALCON_CLIENT_SECRET are correct"
                " and the API key has not been revoked."
            )
        elif self.token_status == 403 and self.member_cid:
            parts.append(
                f"Hint: A member_cid is configured ({self.member_cid})."
                " Verify this is a valid child CID managed by your parent tenant,"
                " not the parent CID itself."
            )
        elif self.token_status == 403:
            parts.append(
                "Hint: Verify the API client has the required scopes"
                " and has not been disabled."
            )
        else:
            parts.append(
                f"Hint: Check network connectivity to {self.base_url}"
                " and verify FALCON_BASE_URL is correct for your CrowdStrike region."
            )

        return ". ".join(parts)

    def authenticate(self) -> bool:
        """Authenticate with the Falcon API.

        Returns:
            bool: True if authentication was successful
        """
        result: bool = self.client.login()
        return result

    def is_authenticated(self) -> bool:
        """Check if the client is authenticated.

        Returns:
            bool: True if the client is authenticated
        """
        result: bool = self.client.token_valid
        return result

    def _ensure_token_fresh(self) -> None:
        """Collapse concurrent stale-token refreshes into a single login.

        FalconPy refreshes lazily inside `command` (it reads `auth_headers`,
        which calls `login()` when the token is stale). Under concurrency several
        offloaded threads can see a stale token simultaneously and each POST
        `/oauth2/token`. This serializes the refresh with a double-checked lock:
        the fast path takes no lock when the token is valid, and only the first
        thread through the lock logs in while the rest observe the fresh token and
        skip.

        The collapse depends on `login()` clearing `token_stale`, so it holds only
        when the refresh succeeds. If login fails (revoked or wrong credentials,
        network failure) the token stays stale and each waiting thread retries in
        turn — N serial attempts rather than N parallel ones. That trades added
        latency for not hammering the token endpoint; the credentials are already
        broken in that state, and `command` still returns the API's 401 to the
        caller.
        """
        client = self.client
        # Fast path: a valid token needs no lock. `refreshable` guards clients that
        # cannot self-refresh (e.g. token supplied directly).
        if not getattr(client, "token_stale", False) or not getattr(
            client, "refreshable", False
        ):
            return

        with self._token_lock:
            # Re-check under the lock: another thread may have refreshed already.
            if getattr(client, "token_stale", False):
                client.login()

    def command(self, operation: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a Falcon API command.

        Args:
            operation: The API operation to execute
            **kwargs: Additional arguments to pass to the API

        Returns:
            dict[str, Any]: The API response
        """
        self._ensure_token_fresh()
        result: dict[str, Any] = self.client.command(operation, **kwargs)
        return result

    async def command_async(self, operation: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a Falcon API command off the event loop.

        Runs the blocking FalconPy call on a worker thread so the asyncio event
        loop stays free to service other in-flight requests. Async handlers (e.g.
        ngsiem) should await this instead of calling the sync `command` directly;
        sync handlers are offloaded automatically by the tool wrapper in
        `BaseModule._add_tool`. The thread-pool cap and cancellation semantics
        described on `offload_to_thread` apply here too — both share the same
        default anyio limiter.

        Args:
            operation: The API operation to execute
            **kwargs: Additional arguments to pass to the API

        Returns:
            dict[str, Any]: The API response
        """
        return await anyio.to_thread.run_sync(
            functools.partial(self.command, operation, **kwargs)
        )

    def get_user_agent(self) -> str:
        """Get RFC-compliant user agent string for API requests.

        Returns:
            str: User agent string in RFC format "falcon-mcp/VERSION (comment; falconpy/VERSION; Python/VERSION; Platform/VERSION)"
        """
        # Get falcon-mcp version
        falcon_mcp_version = get_version()

        # Get Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        # Get platform information
        platform_info = f"{platform.system()}/{platform.release()}"

        # Get FalconPy version
        try:
            falconpy_version = version("crowdstrike-falconpy")
        except PackageNotFoundError:
            falconpy_version = "unknown"
            logger.debug("crowdstrike-falconpy package version not found")

        # Build comment section components (RFC-compliant format)
        comment_parts = []
        if self.user_agent_comment:
            comment_parts.append(self.user_agent_comment.strip())
        comment_parts.extend(
            [f"falconpy/{falconpy_version}", f"Python/{python_version}", platform_info]
        )

        return f"falcon-mcp/{falcon_mcp_version} ({'; '.join(comment_parts)})"

    def get_headers(self) -> dict[str, str]:
        """Get authentication headers for API requests.

        This method returns the authentication headers from the underlying Falcon API client,
        which can be used for custom HTTP requests or advanced integration scenarios.

        Returns:
            dict[str, str]: Authentication headers including the bearer token
        """
        headers: dict[str, str] = self.client.auth_headers
        return headers


def get_version() -> str:
    """Get falcon-mcp version with multiple fallback methods.

    This function tries multiple methods to determine the version:
    1. importlib.metadata (works when package is properly installed)
    2. pyproject.toml (works in development/Docker environments)
    3. Hardcoded fallback

    Returns:
        str: The version string
    """
    # Try importlib.metadata first (works when properly installed)
    try:
        return version("falcon-mcp")
    except PackageNotFoundError:
        logger.debug(
            "falcon-mcp package not found via importlib.metadata, trying pyproject.toml"
        )

    # Try reading from pyproject.toml (works in development/Docker)
    try:
        import pathlib
        import tomllib  # Python 3.11+

        # Look for pyproject.toml in current directory and parent directories
        current_path = pathlib.Path(__file__).parent
        for _ in range(3):  # Check up to 3 levels up
            pyproject_path = current_path / "pyproject.toml"
            if pyproject_path.exists():
                with open(pyproject_path, "rb") as f:
                    data = tomllib.load(f)
                    version_str: str = data["project"]["version"]
                    logger.debug(
                        "Found version %s in pyproject.toml at %s",
                        version_str,
                        pyproject_path,
                    )
                    return version_str
            current_path = current_path.parent

        logger.debug("pyproject.toml not found in current or parent directories")
    except (KeyError, ImportError, OSError, TypeError) as e:
        logger.debug("Failed to read version from pyproject.toml: %s", e)

    # Final fallback
    fallback_version = "0.1.0"
    logger.debug("Using fallback version: %s", fallback_version)
    return fallback_version


# ---------------------------------------------------------------------------
# Multi-tenant request-scoped credential resolution
# ---------------------------------------------------------------------------
#
# In multi-tenant mode the process holds no Falcon credentials at all. Every
# request supplies its own bearer token and region, which the MCP SDK exposes
# through RequestContext.request (a Starlette Request) for the duration of the
# tool call. anyio copies that contextvar into the worker thread, so resolution
# works from inside offload_to_thread — where the blocking FalconPy call runs.
#
# A fresh FalconClient is built per request rather than caching one per tenant:
# mutating a shared client's bearer would let two concurrent requests for the
# same tenant observe each other's token at a rotation boundary, and holding a
# lock across the API call would serialize that tenant. Construction is cheap
# (FalconPy's endpoint table is a module-level reference, and a token-auth
# client performs no login), and connection reuse is preserved by pooling the
# requests.Session per region instead.

MULTI_TENANT_ENV = "FALCON_MCP_MULTI_TENANT"

TOKEN_HEADER = "authorization"
BASE_URL_HEADER = "x-falcon-base-url"

# There is deliberately no per-request member_cid header. FalconPy discards
# member_cid under TOKEN auth — the credential review leaves `creds` empty and
# sends only `Authorization` — so a child-CID request would be answered with the
# token's own CID data. Accepting the header would be worse than not offering it:
# the caller would believe it was scoped when it was not. Per-tenant Flight
# Control needs a token minted for the child CID instead.

# base_url arrives in a request header and FalconPy will send the bearer token
# to whatever host it names, with no validation of its own. An unvalidated value
# is therefore a token-exfiltration vector.
#
# The allowlist is FalconPy's own region table rather than a domain suffix: a
# suffix match would admit any *.crowdstrike.com host (not just the API), and
# would exclude GovCloud-2, which lives on crowdstrike.mil. Taking the vendor's
# enum means new regions arrive with a FalconPy upgrade instead of silently
# failing here.
ALLOWED_API_HOSTS = frozenset(region.value.lower() for region in BaseURL)

# Transport-level settings the server was started with. Per-request clients are
# built from headers, which carry credentials but not operational config, so an
# egress proxy or debug flag passed on the command line would otherwise be lost
# on every multi-tenant call.
_tenant_client_defaults: dict[str, Any] = {
    "debug": False,
    "user_agent_comment": None,
    "proxy": None,
}


def set_tenant_client_defaults(
    *, debug: bool = False, user_agent_comment: str | None = None, proxy: str | None = None
) -> None:
    """Record the server's transport settings for per-request clients."""
    _tenant_client_defaults.update(
        debug=debug, user_agent_comment=user_agent_comment, proxy=proxy
    )


class TenantContextError(Exception):
    """A multi-tenant request carried no usable tenant credentials."""


def multi_tenant_enabled() -> bool:
    """Return True when the environment asks for multi-tenant mode.

    Only used as the default for the --multi-tenant flag. Runtime behaviour is
    driven by whether the server built a process client, not by this — see
    request_tenant_client.
    """
    return os.environ.get(MULTI_TENANT_ENV, "").strip().lower() in ("1", "true", "yes")


def validate_base_url(base_url: str) -> str:
    """Return base_url if it names a Falcon API host, else raise.

    Guards against a request redirecting a live customer bearer token to an
    arbitrary host. Matches the exact hostname against FalconPy's region table,
    so neither a lookalike domain nor an unrelated crowdstrike.com host passes.
    """
    parsed = urlparse(base_url)
    if parsed.scheme != "https":
        raise TenantContextError(
            f"{BASE_URL_HEADER} must use https, got {parsed.scheme or 'no scheme'!r}"
        )
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_API_HOSTS:
        raise TenantContextError(
            f"{BASE_URL_HEADER} host {host!r} is not a CrowdStrike API host "
            f"(expected one of: {', '.join(sorted(ALLOWED_API_HOSTS))})"
        )
    return base_url


def _request_headers() -> Any | None:
    """Return the headers of the request being served, or None if there is none."""
    if request_ctx is None:  # pragma: no cover - SDK layout changed
        logger.warning(
            "mcp.server.lowlevel.server.request_ctx is unavailable; "
            "per-request tenant credentials cannot be resolved"
        )
        return None
    try:
        ctx = request_ctx.get()
    except LookupError:
        return None
    return getattr(getattr(ctx, "request", None), "headers", None)


def request_tenant_client() -> "FalconClient":
    """Build a FalconClient from the calling request's tenant headers.

    Always resolves from the request, or raises. Deliberately takes no view on
    whether the server is in multi-tenant mode: the caller decides that by
    whether it has a process client to prefer (see BaseModule.client). Keeping
    the mode in one place stops the flag and the environment variable from
    disagreeing and silently falling back to the wrong credentials.

    Raises:
        TenantContextError: if there is no request context, or its tenant headers
            are missing or unusable.
    """
    headers = _request_headers()
    if headers is None:
        raise TenantContextError(
            "multi-tenant mode: no HTTP request context available for this tool call"
        )

    scheme, _, token = (headers.get(TOKEN_HEADER) or "").partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        raise TenantContextError(
            "multi-tenant mode: missing or malformed 'Authorization: Bearer <token>' header"
        )

    base_url = headers.get(BASE_URL_HEADER)
    if not base_url:
        raise TenantContextError(
            f"multi-tenant mode: missing {BASE_URL_HEADER} header "
            "(the Falcon API is regional, so the token alone is not enough)"
        )
    base_url = validate_base_url(base_url)

    defaults = _tenant_client_defaults
    return FalconClient(
        access_token=token,
        base_url=base_url,
        # No shared requests.Session: one session per region would put every
        # tenant on a common cookie jar, and requests.Session is not documented
        # thread-safe under the 40-thread offload pool. Connection reuse is not
        # worth either risk on an auth boundary.
        debug=defaults["debug"],
        user_agent_comment=defaults["user_agent_comment"],
        proxy=defaults["proxy"],
    )
