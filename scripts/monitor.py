#!/usr/bin/env python3
"""Providence monitoring script — polls API health and logs metrics.

Usage:
    python scripts/monitor.py [--url URL] [--interval SECONDS] [--verbose]

Checks:
    1. API liveness (GET /api/v1/health/live)
    2. API readiness (GET /api/v1/health/ready)
    3. System health (GET /api/v1/health)
    4. Pipeline stats (GET /api/v1/pipeline/stats)
    5. Store sizes (GET /api/v1/stores/fragments/stats, beliefs/stats)

Outputs structured JSON lines for log aggregation (ELK, CloudWatch, etc.).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def _get(url: str, timeout: int = 10) -> tuple[int, dict | None]:
    """HTTP GET returning (status_code, json_body)."""
    try:
        req = Request(url, headers={"Accept": "application/json"})
        resp = urlopen(req, timeout=timeout)
        data = json.loads(resp.read().decode())
        return resp.status, data
    except HTTPError as e:
        try:
            body = json.loads(e.read().decode())
        except Exception:
            body = None
        return e.code, body
    except URLError as e:
        return 0, {"error": str(e.reason)}
    except Exception as e:
        return 0, {"error": str(e)}


def _log(level: str, check: str, status_code: int, detail: dict | None = None):
    """Emit a structured JSON log line."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "check": check,
        "status_code": status_code,
    }
    if detail:
        entry["detail"] = detail
    print(json.dumps(entry), flush=True)


def run_checks(base_url: str, verbose: bool = False) -> bool:
    """Run all health checks. Returns True if all pass."""
    all_ok = True

    # 1. Liveness
    code, body = _get(f"{base_url}/api/v1/health/live")
    if code == 200:
        _log("INFO", "liveness", code)
    else:
        _log("ERROR", "liveness", code, body)
        all_ok = False

    # 2. Readiness
    code, body = _get(f"{base_url}/api/v1/health/ready")
    if code == 200:
        _log("INFO", "readiness", code)
    else:
        _log("WARN", "readiness", code, body)
        all_ok = False

    # 3. System health
    code, body = _get(f"{base_url}/api/v1/health")
    if code == 200 and body:
        system_status = body.get("system_status", "UNKNOWN")
        level = "INFO" if system_status == "HEALTHY" else "WARN"
        if system_status in ("CRITICAL", "HALTED"):
            level = "ERROR"
            all_ok = False
        _log(level, "system_health", code, {
            "system_status": system_status,
            "agents": body.get("agents"),
            "pipeline": body.get("pipeline"),
        })
    else:
        _log("ERROR", "system_health", code, body)
        all_ok = False

    # 4. Pipeline stats
    code, body = _get(f"{base_url}/api/v1/pipeline/stats")
    if code == 200 and body and verbose:
        _log("INFO", "pipeline_stats", code, body)

    # 5. Store sizes
    code, body = _get(f"{base_url}/api/v1/stores/fragments/stats")
    if code == 200 and body and verbose:
        _log("INFO", "fragment_store", code, {"total": body.get("total_count")})

    code, body = _get(f"{base_url}/api/v1/stores/beliefs/stats")
    if code == 200 and body and verbose:
        _log("INFO", "belief_store", code, {"total": body.get("total_count")})

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Providence health monitor")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--interval", type=int, default=0, help="Poll interval (0=once)")
    parser.add_argument("--verbose", action="store_true", help="Include store/pipeline stats")
    args = parser.parse_args()

    if args.interval <= 0:
        ok = run_checks(args.url, verbose=args.verbose)
        sys.exit(0 if ok else 1)

    print(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": "INFO",
        "check": "monitor_start",
        "status_code": 0,
        "detail": {"url": args.url, "interval": args.interval},
    }), flush=True)

    while True:
        try:
            run_checks(args.url, verbose=args.verbose)
        except KeyboardInterrupt:
            break
        except Exception as e:
            _log("ERROR", "monitor_error", 0, {"error": str(e)})
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
