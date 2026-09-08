"""Tool argument schemas: declaration, validation, and deterministic repair.

The router emits ``tool_args`` as free-form JSON and Brain splats it straight
into ``tool.execute(**args)``. Nothing checks it, so a malformed argument is
discovered only when the tool fails — if it fails at all. On the loop_eval
corpus the ``bad_args`` class recovers at 0-5%, the worst of every recoverable
class, because plain retry re-sends the same wrong value and re-deciding costs
a full routing call to maybe fix one field.

This module adds the cheap half of the fix: a declared parameter schema per
tool, a validator, and a repair pass that costs ZERO model calls.

Scope is deliberately narrow. Repair only handles faults where the correct
value is *derivable* from what the model emitted:

  - type coercion       "5" -> 5, "true" -> True
  - date normalisation  "July 6th" -> "2026-07-06"   (core.date_parser)
  - time normalisation  "8pm" -> "20:00"             (core.time_parser)
  - unknown keys        dropped rather than splatted into **kwargs

It deliberately does NOT guess at:

  - enum mismatches — picking the "nearest" action would paper over a genuine
    decision error and make the agent look better than it is
  - identifiers — no schema can invent ``message_id='m1'`` from "the final
    exam email"; that needs a lookup or the error fed back
  - semantic values — a bare path, a decorated city name, a vague query

That split is the point: it separates faults a validator can fix from faults
that genuinely need the failure observed, instead of collapsing both into one
"argument error" bucket.

Schema shape (a JSON-Schema subset, kept small on purpose)::

    parameters = {
        "action": {"type": "string", "enum": ["today", "create"],
                   "required": True},
        "date":   {"type": "string", "format": "date"},
        "limit":  {"type": "integer"},
    }

Supported ``type``: string, integer, number, boolean.
Supported ``format``: date (YYYY-MM-DD), time (HH:MM).
A tool with empty ``parameters`` is treated as undeclared: validation passes
and repair is a no-op, so tools that have not been annotated keep working
exactly as before.
"""
from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger("bantz.tools.arg_schema")

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_HH_MM = re.compile(r"^\d{2}:\d{2}$")

_TYPE_NAMES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
}

_TRUEY = {"true", "yes", "on", "1"}
_FALSEY = {"false", "no", "off", "0"}


class ArgError(str):
    """An actionable validation message. Subclasses str so it can be joined
    into an observation block without ceremony."""


def _type_ok(value: Any, want: str) -> bool:
    types = _TYPE_NAMES.get(want)
    if types is None:
        return True  # unknown type keyword: do not invent a failure
    if want in ("integer", "number") and isinstance(value, bool):
        return False  # bool is an int subclass; not what "integer" means
    return isinstance(value, types)


def validate(parameters: dict, args: dict) -> list[ArgError]:
    """Return actionable messages for everything wrong with *args*.

    An empty schema returns no errors: undeclared tools are not policed.
    Messages name the field and say what was expected, because they are fed
    back to the model verbatim when repair fails.
    """
    if not parameters:
        return []
    errors: list[ArgError] = []

    for key, spec in parameters.items():
        if not isinstance(spec, dict):
            continue
        present = key in args and args[key] not in ("", None)
        if spec.get("required") and not present:
            errors.append(ArgError(f"{key}: required argument is missing"))
            continue
        if not present:
            continue

        value = args[key]
        want = spec.get("type")
        if want and not _type_ok(value, want):
            errors.append(ArgError(
                f"{key}: expected {want}, got {type(value).__name__} "
                f"({value!r})"))
            continue

        enum = spec.get("enum")
        if enum and value not in enum:
            errors.append(ArgError(
                f"{key}: {value!r} is not one of {', '.join(map(str, enum))}"))
            continue

        fmt = spec.get("format")
        if fmt == "date" and isinstance(value, str) and not _ISO_DATE.match(value):
            errors.append(ArgError(
                f"{key}: {value!r} is not a date — use YYYY-MM-DD"))
        elif fmt == "time" and isinstance(value, str) and not _HH_MM.match(value):
            errors.append(ArgError(
                f"{key}: {value!r} is not a time — use 24h HH:MM"))

    unknown = [k for k in args if k not in parameters]
    if unknown:
        errors.append(ArgError(
            f"unknown argument(s): {', '.join(sorted(unknown))}"))
    return errors


def _coerce_scalar(value: Any, want: str) -> tuple[Any, bool]:
    """Best-effort scalar coercion. Returns (value, changed)."""
    if _type_ok(value, want):
        return value, False
    if want == "boolean" and isinstance(value, str):
        low = value.strip().lower()
        if low in _TRUEY:
            return True, True
        if low in _FALSEY:
            return False, True
        return value, False
    if want in ("integer", "number") and isinstance(value, str):
        text = value.strip()
        try:
            return (int(text), True) if want == "integer" else (float(text), True)
        except ValueError:
            return value, False
    if want == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value), True
    return value, False


def repair(parameters: dict, args: dict) -> tuple[dict, list[str]]:
    """Deterministically repair what is repairable. Zero model calls.

    Returns the repaired args and a list of human-readable notes describing
    each change, so a caller can log or report exactly what was touched.
    Values that cannot be derived are left untouched for ``validate`` to
    report — repair never guesses.
    """
    if not parameters:
        return dict(args), []

    out = dict(args)
    notes: list[str] = []

    for key in [k for k in out if k not in parameters]:
        notes.append(f"dropped unknown argument {key!r}")
        out.pop(key, None)

    for key, spec in parameters.items():
        if not isinstance(spec, dict) or key not in out:
            continue
        value = out[key]
        if value in ("", None):
            continue

        want = spec.get("type")
        if want:
            new, changed = _coerce_scalar(value, want)
            if changed:
                notes.append(f"coerced {key} {value!r} -> {new!r}")
                out[key] = new
                value = new

        fmt = spec.get("format")
        if fmt == "date" and isinstance(value, str) and not _ISO_DATE.match(value):
            fixed = _to_iso_date(value)
            if fixed:
                notes.append(f"normalised {key} {value!r} -> {fixed!r}")
                out[key] = fixed
        elif fmt == "time" and isinstance(value, str) and not _HH_MM.match(value):
            fixed = _to_hh_mm(value)
            if fixed:
                notes.append(f"normalised {key} {value!r} -> {fixed!r}")
                out[key] = fixed

    # Fill declared defaults only for absent keys — never overwrite the model.
    for key, spec in parameters.items():
        if isinstance(spec, dict) and "default" in spec and key not in out:
            out[key] = spec["default"]
            notes.append(f"filled default {key}={spec['default']!r}")

    return out, notes


_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
# "July 6th" / "July 6, 2026"  and  "6th of July" / "6 July 2026"
_MD = re.compile(
    r"\b(?P<mon>[a-z]+)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?"
    r"(?:[,\s]+(?P<year>\d{4}))?\b", re.IGNORECASE)
_DM = re.compile(
    r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<mon>[a-z]+)"
    r"(?:[,\s]+(?P<year>\d{4}))?\b", re.IGNORECASE)


def _absolute_date(text: str) -> str | None:
    """Month-name dates -> YYYY-MM-DD.

    ``core.date_parser`` resolves only relative references (tomorrow, thursday,
    next week), so "July 6th" — the single most common way a model writes a
    date it was told in prose — falls through it. Parsed here rather than in
    the production parser so routing and reminders keep their exact current
    behaviour.

    A bare month-day carries no year. The current year is assumed, matching
    the usual convention; that is a documented limitation, not an inference —
    a date in the recent past stays in the past rather than rolling forward.
    """
    from datetime import datetime
    for rx in (_MD, _DM):
        m = rx.search(text)
        if not m:
            continue
        month = _MONTHS.get(m.group("mon").lower())
        if month is None:
            continue
        day = int(m.group("day"))
        year = int(m.group("year")) if m.group("year") else datetime.now().year
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return None  # e.g. February 31st
    return None


def _to_iso_date(text: str) -> str | None:
    """Natural-language date -> YYYY-MM-DD.

    Relative references go through the production parser; absolute month-name
    dates are handled here because that parser does not cover them.
    """
    try:
        from bantz.core.date_parser import resolve_date
        dt = resolve_date(text)
        if dt:
            return dt.strftime("%Y-%m-%d")
        return _absolute_date(text)
    except Exception:  # noqa: BLE001 — repair must never break a tool call
        log.debug("date repair failed for %r", text, exc_info=True)
        return None


def _to_hh_mm(text: str) -> str | None:
    """Natural-language time -> HH:MM, reusing the production parser."""
    try:
        from bantz.core.time_parser import resolve_time
        out = resolve_time(text)
        return out if out and _HH_MM.match(out) else None
    except Exception:  # noqa: BLE001
        log.debug("time repair failed for %r", text, exc_info=True)
        return None
