"""Live MCP probe: verify pagination + chunked hydration.

Loads creds from ~/.config/falcon-mcp/.env, calls search_detections twice
(offset=0, offset=PAGE), confirms IDs are disjoint and no 413s. Run with:

    cd /Users/gdandu/gitlocal/falcon-mcp && uv run python scripts/probe_pagination.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ENV_FILE = Path.home() / ".config" / "falcon-mcp" / ".env"
PAGE = 1000  # same value as the falcon-mcp internal hydration chunk
WINDOW_DAYS = 30  # rolling window so the script never bitrots
STRESS_LIMIT = 2500  # forces ≥3 hydration batches inside _base_get_by_ids

if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

for required in ("FALCON_CLIENT_ID", "FALCON_CLIENT_SECRET", "FALCON_BASE_URL"):
    if not os.environ.get(required):
        print(f"missing {required}", file=sys.stderr)
        sys.exit(1)

from falcon_mcp.client import FalconClient  # noqa: E402
from falcon_mcp.modules.base import BaseModule  # noqa: E402

client = FalconClient(
    base_url=os.environ["FALCON_BASE_URL"],
    debug=False,
    user_agent_comment="uptycs-pagination-probe",
)
if not client.authenticate():
    print("auth failed", file=sys.stderr)
    sys.exit(2)


# Drop down to the API level so we sidestep the FQL-error wrapping that
# falcon_search_detections does on any internal failure, and can see exactly
# which call returns what.
class Probe(BaseModule):
    def register_tools(self, server):  # noqa: D401
        pass


probe = Probe(client)


def query_ids(filter_: str, limit: int, offset: int | None) -> list[str]:
    """Step 1: queries call → returns composite_ids."""
    # Use created_timestamp.desc — high-cardinality timestamps ⇒ stable paging.
    params = {"filter": filter_, "limit": limit, "sort": "created_timestamp.desc"}
    if offset is not None:
        params["offset"] = offset
    result = probe._base_search_api_call(
        operation="GetQueriesAlertsV2",
        search_params=params,
        error_message="queries failed",
    )
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"queries error: {result['error']!r}")
    return list(result)


def hydrate(ids: list[str]) -> list[dict]:
    """Step 2: hydration call(s) → exercises the patched chunking."""
    res = probe._base_get_by_ids(
        operation="PostEntitiesAlertsV2",
        ids=ids,
        id_key="composite_ids",
        include_hidden=True,
    )
    if isinstance(res, dict) and "error" in res:
        raise RuntimeError(f"hydrate error: {res['error']!r}")
    return res


since = (datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
FILTER = f"created_timestamp:>'{since}'"
print(f"base={os.environ['FALCON_BASE_URL']}, filter={FILTER}")

# A. Pagination at the queries layer.
page1_ids = query_ids(FILTER, limit=PAGE, offset=0)
page2_ids = query_ids(FILTER, limit=PAGE, offset=PAGE)
overlap = set(page1_ids) & set(page2_ids)
print(f"page1: {len(page1_ids)} ids   page2: {len(page2_ids)} ids   overlap: {len(overlap)} (expected 0)")
if overlap:
    print("FAIL: pages overlap")
    sys.exit(4)

# B. Chunked hydration: big ID list forces multi-batch path (>1000 IDs).
# Fail loud if the tenant doesn't have enough detections to exercise batching —
# otherwise the test silently degrades to single-chunk and stops being meaningful.
big_ids = query_ids(FILTER, limit=STRESS_LIMIT, offset=0)
print(f"big query: {len(big_ids)} ids (need >{PAGE} to exercise batching)")
if len(big_ids) <= PAGE:
    print(f"FAIL: tenant has only {len(big_ids)} detections in last {WINDOW_DAYS}d — "
          f"chunking path NOT exercised. Pick a tenant with denser detection volume "
          f"or widen WINDOW_DAYS.")
    sys.exit(5)

rows = hydrate(big_ids)
expected_batches = (len(big_ids) + PAGE - 1) // PAGE
print(f"hydrated {len(rows)} rows from {len(big_ids)} ids "
      f"({expected_batches} POST batches inside _base_get_by_ids)")

if len(rows) != len(big_ids):
    print(f"FAIL: row count {len(rows)} != id count {len(big_ids)} — chunks dropped data")
    sys.exit(6)

print("OK")
