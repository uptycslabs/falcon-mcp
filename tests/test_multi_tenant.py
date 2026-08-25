"""
Tests for multi-tenant mode.

Multi-tenant mode lets one HTTP process serve many Falcon tenants by taking the
bearer token and region from each request instead of the environment. The tests
here concentrate on the ways that can fail *silently* — serving one tenant's data
to another, or leaking a bearer token off-platform — because those produce wrong
answers rather than errors.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from starlette.requests import Request

from falcon_mcp.client import (
    MULTI_TENANT_ENV,
    FalconClient,
    TenantContextError,
    multi_tenant_enabled,
    request_tenant_client,
    set_tenant_client_defaults,
    validate_base_url,
)
from falcon_mcp.modules.base import BaseModule

REAL_BASE_URL = "https://api.us-2.crowdstrike.com"


def request_context(headers: dict[str, str]) -> MagicMock:
    """Build a RequestContext-alike carrying a Starlette request with headers."""
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    request = Request({"type": "http", "headers": raw, "method": "POST", "path": "/mcp"})
    ctx = MagicMock()
    ctx.request = request
    return ctx


class _Module(BaseModule):
    """Concrete BaseModule so the client property can be exercised."""

    def register_tools(self, server):  # pragma: no cover - not used
        pass


class TestMultiTenantEnabled(unittest.TestCase):
    """The mode flag is read from the environment."""

    def test_truthy_values_enable(self):
        for value in ("1", "true", "TRUE", "yes", " true "):
            with patch.dict(os.environ, {MULTI_TENANT_ENV: value}):
                self.assertTrue(multi_tenant_enabled(), f"{value!r} should enable")

    def test_other_values_disable(self):
        for value in ("", "0", "false", "off", "no"):
            with patch.dict(os.environ, {MULTI_TENANT_ENV: value}):
                self.assertFalse(multi_tenant_enabled(), f"{value!r} should disable")


class TestValidateBaseURL(unittest.TestCase):
    """base_url comes from a request header, so it is attacker-influenced.

    FalconPy sends the bearer token to whatever host base_url names and performs
    no validation of its own, so an unchecked value exfiltrates a live customer
    token. These cases are the reason the allowlist exists.
    """

    def test_accepts_every_falconpy_region(self):
        # Sourced from FalconPy's own table so a new region needs no change here.
        from falconpy import BaseURL

        for region in BaseURL:
            url = f"https://{region.value}"
            self.assertEqual(validate_base_url(url), url, f"{region.name} must be allowed")

    def test_accepts_govcloud_mil_host(self):
        # GovCloud-2 is on crowdstrike.mil; a .crowdstrike.com suffix check
        # would have excluded it entirely.
        url = "https://api.us-gov-2.crowdstrike.mil"
        self.assertEqual(validate_base_url(url), url)

    def test_rejects_non_api_crowdstrike_host(self):
        # A suffix check would have admitted this; only API hosts are valid
        # destinations for a bearer token.
        with self.assertRaises(TenantContextError):
            validate_base_url("https://falcon.crowdstrike.com")

    def test_rejects_off_platform_host(self):
        with self.assertRaises(TenantContextError):
            validate_base_url("https://evil.example.com")

    def test_rejects_suffix_lookalike(self):
        # The check runs on the parsed hostname, so appending a domain does not
        # sneak past a naive "endswith" on the raw string.
        with self.assertRaises(TenantContextError):
            validate_base_url("https://api.crowdstrike.com.evil.example.com")

    def test_rejects_plaintext_http(self):
        with self.assertRaises(TenantContextError):
            validate_base_url("http://api.crowdstrike.com")

    def test_rejects_credentials_in_userinfo(self):
        # urlparse puts "api.crowdstrike.com" in the userinfo here; hostname is
        # the real target, so this must be rejected.
        with self.assertRaises(TenantContextError):
            validate_base_url("https://api.crowdstrike.com@evil.example.com")


class TestTenantClientDefaults(unittest.TestCase):
    """Transport settings travel with the server, not the request."""

    @patch("falcon_mcp.client.APIHarnessV2")
    def test_server_proxy_and_debug_reach_per_request_clients(self, mock_harness):
        # Credentials arrive per request; an egress proxy or debug flag does not,
        # so the server's own settings must be carried forward or they are lost
        # on every multi-tenant call.
        set_tenant_client_defaults(
            debug=True, user_agent_comment="Juno/1.0", proxy="http://egress:8080"
        )
        try:
            headers = {
                "Authorization": "Bearer tok",
                "X-Falcon-Base-Url": REAL_BASE_URL,
            }
            with patch("falcon_mcp.client.request_ctx") as ctx:
                ctx.get.return_value = request_context(headers)
                client = request_tenant_client()
            self.assertEqual(client.proxy, "http://egress:8080")
            params = mock_harness.call_args[1]
            self.assertEqual(params["proxy"], {"https": "http://egress:8080"})
            self.assertTrue(params["debug"])
            self.assertIn("Juno/1.0", params["user_agent"])
            # No shared session: one per region would pool cookies across tenants.
            self.assertNotIn("session", params)
        finally:
            set_tenant_client_defaults()


@patch("falcon_mcp.client.APIHarnessV2")
class TestResolveRequestClient(unittest.TestCase):
    """Per-request credential resolution."""

    def test_resolves_regardless_of_env_flag(self, _harness):
        """Resolution must not consult the env var.

        Regression test: the mode used to be read from FALCON_MCP_MULTI_TENANT
        here while the server read it from the --multi-tenant flag. Starting with
        the flag alone left the two disagreeing, so resolution silently declined
        and every request failed as though it had no tenant context.
        """
        headers = {
            "Authorization": "Bearer tenant-a-token",
            "X-Falcon-Base-Url": REAL_BASE_URL,
        }
        with patch.dict(os.environ, {MULTI_TENANT_ENV: "false"}, clear=True):
            with patch("falcon_mcp.client.request_ctx") as ctx:
                ctx.get.return_value = request_context(headers)
                client = request_tenant_client()
        self.assertEqual(client.access_token, "tenant-a-token")

    def test_resolves_token_and_region(self, mock_harness):
        headers = {
            "Authorization": "Bearer tenant-a-token",
            "X-Falcon-Base-Url": REAL_BASE_URL,
            "X-Falcon-Member-Cid": "child-cid",  # must be ignored, see below
        }
        with patch.dict(os.environ, {}, clear=True):
            with patch("falcon_mcp.client.request_ctx") as ctx:
                ctx.get.return_value = request_context(headers)
                client = request_tenant_client()

        self.assertIsNotNone(client)
        self.assertEqual(client.access_token, "tenant-a-token")
        self.assertEqual(client.base_url, REAL_BASE_URL)
        # FalconPy drops member_cid under token auth, so honouring the header
        # would answer a child-CID request with the parent's data.
        self.assertIsNone(client.member_cid)
        self.assertNotIn("member_cid", mock_harness.call_args[1])

        # An injected token must be passed alone: FalconPy only honours
        # access_token when no credential pair is present, so shipping both would
        # silently authenticate as the credential owner instead of the tenant.
        params = mock_harness.call_args[1]
        self.assertEqual(params["access_token"], "tenant-a-token")
        self.assertNotIn("client_id", params)
        self.assertNotIn("client_secret", params)

    def test_missing_authorization_fails_closed(self, _harness):
        with patch.dict(os.environ, {}, clear=True):
            with patch("falcon_mcp.client.request_ctx") as ctx:
                ctx.get.return_value = request_context(
                    {"X-Falcon-Base-Url": REAL_BASE_URL}
                )
                with self.assertRaises(TenantContextError):
                    request_tenant_client()

    def test_non_bearer_scheme_fails_closed(self, _harness):
        with patch.dict(os.environ, {}, clear=True):
            with patch("falcon_mcp.client.request_ctx") as ctx:
                ctx.get.return_value = request_context(
                    {"Authorization": "Basic abc", "X-Falcon-Base-Url": REAL_BASE_URL}
                )
                with self.assertRaises(TenantContextError):
                    request_tenant_client()

    def test_missing_base_url_fails_closed(self, _harness):
        # The Falcon API is regional, so a token on its own is not enough.
        with patch.dict(os.environ, {}, clear=True):
            with patch("falcon_mcp.client.request_ctx") as ctx:
                ctx.get.return_value = request_context(
                    {"Authorization": "Bearer tenant-a-token"}
                )
                with self.assertRaises(TenantContextError):
                    request_tenant_client()

    def test_off_platform_base_url_fails_closed(self, _harness):
        with patch.dict(os.environ, {}, clear=True):
            with patch("falcon_mcp.client.request_ctx") as ctx:
                ctx.get.return_value = request_context(
                    {
                        "Authorization": "Bearer tenant-a-token",
                        "X-Falcon-Base-Url": "https://evil.example.com",
                    }
                )
                with self.assertRaises(TenantContextError):
                    request_tenant_client()

    def test_no_request_context_fails_closed(self, _harness):
        # Outside a request there is no tenant, and guessing one would mean
        # answering with someone else's credentials.
        with patch.dict(os.environ, {}, clear=True):
            with patch("falcon_mcp.client.request_ctx") as ctx:
                ctx.get.side_effect = LookupError
                with self.assertRaises(TenantContextError):
                    request_tenant_client()


@patch("falcon_mcp.client.APIHarnessV2")
class TestBaseModuleClientProperty(unittest.TestCase):
    """BaseModule.client is the single seam every tool call resolves through."""

    def test_process_client_present_means_single_tenant(self, _harness):
        process_client = MagicMock(spec=FalconClient)
        module = _Module(process_client)
        self.assertIs(module.client, process_client)

    def test_process_client_wins_even_inside_a_request(self, _harness):
        """A single-tenant server ignores tenant headers entirely.

        Otherwise any caller could redirect a single-tenant deployment at another
        region by adding a header.
        """
        process_client = MagicMock(spec=FalconClient)
        module = _Module(process_client)
        headers = {
            "Authorization": "Bearer tenant-b-token",
            "X-Falcon-Base-Url": REAL_BASE_URL,
        }
        with patch("falcon_mcp.client.request_ctx") as ctx:
            ctx.get.return_value = request_context(headers)
            self.assertIs(module.client, process_client)

    def test_no_process_client_resolves_from_request(self, _harness):
        module = _Module(None)
        headers = {
            "Authorization": "Bearer tenant-b-token",
            "X-Falcon-Base-Url": REAL_BASE_URL,
        }
        with patch("falcon_mcp.client.request_ctx") as ctx:
            ctx.get.return_value = request_context(headers)
            resolved = module.client
        self.assertEqual(resolved.access_token, "tenant-b-token")

    def test_no_process_client_and_no_context_raises(self, _harness):
        # This is the configuration a multi-tenant server runs in, so a request
        # without tenant headers must error rather than fall back to anything.
        module = _Module(None)
        with patch("falcon_mcp.client.request_ctx") as ctx:
            ctx.get.side_effect = LookupError
            with self.assertRaises(TenantContextError):
                _ = module.client

    def test_concurrent_requests_do_not_share_credentials(self, _harness):
        """Two tenants resolved in sequence must not bleed into one another.

        A design that cached one client per tenant and mutated its bearer would
        pass a single-request test and fail this one.
        """
        module = _Module(None)
        seen = []
        with patch.dict(os.environ, {MULTI_TENANT_ENV: "true"}, clear=True):
            for token, base in (
                ("token-a", "https://api.us-2.crowdstrike.com"),
                ("token-b", "https://api.eu-1.crowdstrike.com"),
            ):
                with patch("falcon_mcp.client.request_ctx") as ctx:
                    ctx.get.return_value = request_context(
                        {"Authorization": f"Bearer {token}", "X-Falcon-Base-Url": base}
                    )
                    client = module.client
                    seen.append((client.access_token, client.base_url))

        self.assertEqual(
            seen,
            [
                ("token-a", "https://api.us-2.crowdstrike.com"),
                ("token-b", "https://api.eu-1.crowdstrike.com"),
            ],
        )


class TestClientTokenInjection(unittest.TestCase):
    """FalconClient accepts a pre-minted token instead of a credential pair."""

    @patch("falcon_mcp.client.APIHarnessV2")
    def test_token_only_client_omits_credentials(self, mock_harness):
        with patch.dict(os.environ, {}, clear=True):
            FalconClient(access_token="tok", base_url=REAL_BASE_URL)
        params = mock_harness.call_args[1]
        self.assertEqual(params["access_token"], "tok")
        self.assertNotIn("client_id", params)
        self.assertNotIn("client_secret", params)

    @patch("falcon_mcp.client.APIHarnessV2")
    def test_token_wins_over_environment_credentials(self, mock_harness):
        # A stray credential pair in the environment must not quietly displace
        # the caller's token — that is how every tenant ends up sharing one identity.
        env = {"FALCON_CLIENT_ID": "env-id", "FALCON_CLIENT_SECRET": "env-secret"}
        with patch.dict(os.environ, env, clear=True):
            FalconClient(access_token="tok", base_url=REAL_BASE_URL)
        params = mock_harness.call_args[1]
        self.assertEqual(params["access_token"], "tok")
        self.assertNotIn("client_id", params)

    @patch("falcon_mcp.client.APIHarnessV2")
    def test_credentials_still_work_without_token(self, mock_harness):
        env = {"FALCON_CLIENT_ID": "env-id", "FALCON_CLIENT_SECRET": "env-secret"}
        with patch.dict(os.environ, env, clear=True):
            FalconClient(base_url=REAL_BASE_URL)
        params = mock_harness.call_args[1]
        self.assertEqual(params["client_id"], "env-id")
        self.assertNotIn("access_token", params)

    @patch("falcon_mcp.client.APIHarnessV2")
    def test_no_token_and_no_credentials_raises(self, _harness):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                FalconClient(base_url=REAL_BASE_URL)


if __name__ == "__main__":
    unittest.main()
