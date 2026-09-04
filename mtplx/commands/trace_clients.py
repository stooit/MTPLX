"""Normalize local harness transcripts without changing their history."""

from __future__ import annotations

import datetime
import json
from pathlib import Path


def load_pi_session(path: Path) -> tuple[dict, list[dict]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    header = next((r for r in records if r.get("type") == "session"), {})
    # Pi is an append-only tree. Follow the current leaf's ancestors, so a
    # fork or rewind does not quietly include abandoned turns in the report.
    entries = {r["id"]: r for r in records if r.get("id") and r.get("type") != "session"}
    lineage, seen = [], set()
    current = next((r for r in reversed(records) if r.get("id") in entries), None)
    while current and current["id"] not in seen:
        seen.add(current["id"])
        lineage.append(current)
        current = entries.get(current.get("parentId"))
    messages = []
    for record in reversed(lineage):
        if record.get("type") != "message":
            continue
        message = dict(record.get("message") or {})
        if message.get("role") not in {"user", "assistant"}:
            continue
        stamp = message.get("timestamp")
        if stamp is None:
            stamp = datetime.datetime.fromisoformat(record["timestamp"]).timestamp() * 1000
        usage = message.get("usage") or {}
        content = message.get("content") or []
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        parts = []
        for part in content:
            kind = part.get("type")
            if kind == "thinking":
                parts.append({"type": "reasoning", "text": part.get("thinking", "")})
            elif kind == "toolCall":
                parts.append({"type": "tool", "tool": part.get("name"), "callID": part.get("id"),
                              "state": {"input": part.get("arguments")}})
            else:
                parts.append(part)
        messages.append({
            **message, "_id": record["id"], "_parts": parts,
            "_parent_entry_id": record.get("parentId"),
            "_time_created_s": float(stamp) / 1000,
            "time": {"created": stamp},
            "tokens": {"input": usage.get("input", 0), "output": usage.get("output", 0),
                       "cache": {"read": usage.get("cacheRead", 0)}},
            "finish": message.get("stopReason"),
        })
    return {"id": header.get("id"), "directory": header.get("cwd"),
            "title": path.stem, "client": "pi", "transcript_path": str(path)}, messages
