#!/usr/bin/env python3
"""Pre-tool-use hook to prevent reading irrelevant/sensitive files."""
import json
import sys

BLOCKED_PATTERNS = [
    ".env",
    ".venv",
    ".DS_Store",
    ".idea",
    "__pycache__",
    "app.yaml",
    "cred.json",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "wandb/",
    ".git/",
]


def main():
    tool_args = json.loads(sys.stdin.read())

    # Extract the file path Claude is trying to read
    tool_input = tool_args.get("tool_input", {})
    read_path = tool_input.get("file_path") or tool_input.get("path") or ""

    # Check if path contains any blocked pattern
    for pattern in BLOCKED_PATTERNS:
        if pattern in read_path:
            print(f"Blocked: cannot read files matching '{pattern}'", file=sys.stderr)
            sys.exit(2)


if __name__ == "__main__":
    main()
