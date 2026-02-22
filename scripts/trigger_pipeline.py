#!/usr/bin/env python3
"""Trigger the embedding pipeline locally without a real Pub/Sub subscription.

Builds the Pub/Sub push envelope and POSTs it to the running pipeline service.

Usage:
    python scripts/trigger_pipeline.py --expedition-ids 1 2 3 4 5
    python scripts/trigger_pipeline.py --expedition-ids 1 --port 8001
"""

import argparse
import base64
import json
import urllib.error
import urllib.request


def trigger(expedition_ids: list[int], port: int = 8001) -> None:
    payload = json.dumps({"expedition_ids": expedition_ids}).encode()
    data = base64.b64encode(payload).decode()

    body = json.dumps({
        "message": {
            "data": data,
            "messageId": "local-trigger-001",
            "publishTime": "2026-02-22T10:00:00Z",
        },
        "subscription": "local-trigger",
    }).encode()

    req = urllib.request.Request(
        f"http://localhost:{port}/",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as response:
            print(f"Status: {response.status} — batch of {len(expedition_ids)} expedition(s) queued.")
    except urllib.error.HTTPError as exc:
        print(f"Error {exc.code}: {exc.read().decode()}")
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger the local embedding pipeline.")
    parser.add_argument("--expedition-ids", nargs="+", type=int, required=True)
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    trigger(args.expedition_ids, args.port)
