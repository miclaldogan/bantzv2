"""
Bantz v3 — Data Migration Utility

Validates existing v2 JSON data files and imports them into the unified
SQLite schema.  Runs automatically on first launch (via DataLayer) or
manually via CLI:

    python -m bantz.data.migration --validate
    python -m bantz.data.migration --migrate
    python -m bantz.data.migration --migrate --dry-run
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("bantz.data.migration")

DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "bantz"

_JSON_FILES = {
    "profile": "profile.json",
    "places": "places.json",
    "schedule": "schedule.json",
    "session": "session.json",
}


# ━━ Validation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def validate_json_files(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, Any]:
    """Check all known JSON files: exist? parseable? Return a report dict.

    >>> report = validate_json_files()
    >>> report["profile"]
    {'status': 'ok', 'path': '...', 'entries': 8}
    """
    report: dict[str, Any] = {}
    for name, filename in _JSON_FILES.items():
        path = data_dir / filename
        if not path.exists():
            report[name] = {"status": "missing", "path": str(path)}
            continue
        try:
            data = json.loads(path.read_text("utf-8"))
            report[name] = {
                "status": "ok",
                "path": str(path),
                "entries": len(data) if isinstance(data, (list, dict)) else 1,
            }
        except Exception as exc:
            report[name] = {"status": "error", "path": str(path), "error": str(exc)}
    return report


# ━━ SQLite Migration ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


_UNIFIED_SCHEMA = """\
CREATE TABLE IF NOT EXISTS user_profile (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS places (
    key    TEXT PRIMARY KEY,
    label  TEXT NOT NULL,
    lat    REAL NOT NULL DEFAULT 0.0,
    lon    REAL NOT NULL DEFAULT 0.0,
    radius REAL NOT NULL DEFAULT 100.0
);

CREATE TABLE IF NOT EXISTS schedule_entries (
    day      TEXT NOT NULL,
    idx      INTEGER NOT NULL,
    name     TEXT NOT NULL,
    time     TEXT NOT NULL,
    duration INTEGER NOT NULL DEFAULT 60,
    location TEXT DEFAULT '',
    type     TEXT DEFAULT '',
    PRIMARY KEY (day, idx)
);

CREATE TABLE IF NOT EXISTS session_state (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def migrate_to_sqlite(
    db_path: Path,
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Import JSON data into unified SQLite tables.

    Parameters:
        db_path  – target SQLite database (can be the same as bantz.db)
        data_dir – directory containing the JSON files
        dry_run  – if True, validate but don't write

    Returns:
        A results dict with per-file status and counts.
    """
    results: dict[str, Any] = {}
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")

    # Create unified tables
    conn.executescript(_UNIFIED_SCHEMA)

    # ── profile → user_profile ────────────────────────────────────────
    _migrate_kv(conn, data_dir / "profile.json", "user_profile", results, dry_run)

    # ── session → session_state ───────────────────────────────────────
    _migrate_kv(conn, data_dir / "session.json", "session_state", results, dry_run)

    # ── places → places table ─────────────────────────────────────────
    places_path = data_dir / "places.json"
    if places_path.exists():
        try:
            data = json.loads(places_path.read_text("utf-8"))
            count = 0
            if not dry_run:
                for key, place in data.items():
                    conn.execute(
                        """INSERT OR REPLACE INTO places(key, label, lat, lon, radius)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            key,
                            place.get("label", key),
                            place.get("lat", 0.0),
                            place.get("lon", 0.0),
                            place.get("radius", 100.0),
                        ),
                    )
                    count += 1
            else:
                count = len(data)
            results["places"] = {"migrated": count, "status": "ok"}
        except Exception as exc:
            results["places"] = {"status": "error", "error": str(exc)}
    else:
        results["places"] = {"status": "skipped", "reason": "file not found"}

    # ── schedule → schedule_entries table ─────────────────────────────
    schedule_path = data_dir / "schedule.json"
    if schedule_path.exists():
        try:
            data = json.loads(schedule_path.read_text("utf-8"))
            count = 0
            if not dry_run:
                for day, entries in data.items():
                    for idx, entry in enumerate(entries):
                        conn.execute(
                            """INSERT OR REPLACE INTO schedule_entries
                                   (day, idx, name, time, duration, location, type)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (
                                day,
                                idx,
                                entry.get("name", ""),
                                entry.get("time", ""),
                                entry.get("duration", 60),
                                entry.get("location", ""),
                                entry.get("type", ""),
                            ),
                        )
                        count += 1
            else:
                count = sum(len(v) for v in data.values() if isinstance(v, list))
            results["schedule"] = {"migrated": count, "status": "ok"}
        except Exception as exc:
            results["schedule"] = {"status": "error", "error": str(exc)}
    else:
        results["schedule"] = {"status": "skipped", "reason": "file not found"}

    if not dry_run:
        conn.commit()
    conn.close()

    log.info("Migration %s: %s", "dry-run" if dry_run else "complete", results)
    return results


def _migrate_kv(
    conn: sqlite3.Connection,
    json_path: Path,
    table_name: str,
    results: dict,
    dry_run: bool,
) -> None:
    """Migrate a flat JSON file into a key/value table."""
    # Derive a friendly name from the table name
    name = table_name.replace("user_", "").replace("_state", "")
    if not json_path.exists():
        results[name] = {"status": "skipped", "reason": "file not found"}
        return
    try:
        data = json.loads(json_path.read_text("utf-8"))
        now = datetime.now().isoformat()
        if not dry_run:
            for k, v in data.items():
                conn.execute(
                    f"""INSERT OR REPLACE INTO {table_name}(key, value, updated_at)
                       VALUES (?, ?, ?)""",
                    (k, json.dumps(v, ensure_ascii=False), now),
                )
        results[name] = {"migrated": len(data), "status": "ok"}
    except Exception as exc:
        results[name] = {"status": "error", "error": str(exc)}


# ━━ CLI entry point ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Bantz data migration utility")
    parser.add_argument("--validate", action="store_true", help="Validate JSON files")
    parser.add_argument("--migrate", action="store_true", help="Migrate JSON → SQLite")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, don't write")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--db", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.validate:
        report = validate_json_files(args.data_dir)
        for name, info in report.items():
            status = info["status"]
            icon = "✅" if status == "ok" else "❌" if status == "error" else "⚠️"
            print(f"  {icon} {name}: {info}")
        sys.exit(0)

    if args.migrate:
        db = args.db or (args.data_dir / "bantz.db")
        results = migrate_to_sqlite(db, args.data_dir, dry_run=args.dry_run)
        tag = " (dry-run)" if args.dry_run else ""
        for name, info in results.items():
            status = info["status"]
            icon = "✅" if status == "ok" else "❌" if status == "error" else "⏭️"
            print(f"  {icon} {name}{tag}: {info}")
        sys.exit(0)

    parser.print_help()
