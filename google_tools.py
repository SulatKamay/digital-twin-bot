"""
google_tools.py - what the twin can DO in Google Calendar and Google Tasks.

Each action is a small function. Gemini picks which one to call (via the tool
descriptions in TOOL_DECLARATIONS), brain.py runs it through GoogleTools.run(),
and the result goes back to Gemini so it can answer in the twin's voice.

Only the owner gets these tools (bot.py decides). Google's client library is
blocking, so every call runs in a worker thread to keep the bot responsive.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import date, datetime, timedelta
from datetime import time as dtime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from google.genai import types
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from google_auth import GoogleAuthError, get_credentials

log = logging.getLogger("twin.google")

MAX_EVENTS = 25        # most events returned for one lookup
MAX_RANGE_DAYS = 31    # longest date range allowed for one lookup
MAX_TASKS = 50


# ---------------------------------------------------------------------------
# 1. Tool descriptions: this is what Gemini reads to decide what to call
# ---------------------------------------------------------------------------
def _declare(name: str, description: str, properties: dict, required: list | None = None):
    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return types.FunctionDeclaration(
        name=name, description=description, parameters_json_schema=schema
    )


def _text(description: str) -> dict:
    return {"type": "string", "description": description}


_TIME_HELP = (
    "'YYYY-MM-DDTHH:MM' in 24-hour local time (e.g. 2026-11-03T15:00), "
    "or 'YYYY-MM-DD' for an all-day event"
)

TOOL_DECLARATIONS = [
    _declare(
        "list_events",
        "List calendar events between two dates (inclusive). Use it for questions like "
        "'what's on tomorrow?' or 'am I free Friday?'. Also use it to find an event's id "
        "before changing or deleting it.",
        {
            "start_date": _text("First day, YYYY-MM-DD"),
            "end_date": _text("Last day, YYYY-MM-DD. Leave empty for just start_date."),
        },
        ["start_date"],
    ),
    _declare(
        "create_event",
        "Add an event to the calendar.",
        {
            "title": _text("Event title"),
            "start": _text(f"Start: {_TIME_HELP}"),
            "end": _text("End, same format as start. Empty = 1 hour long (or one day if all-day)."),
            "location": _text("Optional location"),
            "description": _text("Optional notes"),
        },
        ["title", "start"],
    ),
    _declare(
        "update_event",
        "Change an existing event. Get the event_id from list_events first. Only send the "
        "fields that change. To reschedule, send BOTH start and end.",
        {
            "event_id": _text("The event's id from list_events"),
            "title": _text("New title"),
            "start": _text(f"New start: {_TIME_HELP}"),
            "end": _text("New end, same format as start"),
            "location": _text("New location"),
            "description": _text("New notes"),
        },
        ["event_id"],
    ),
    _declare(
        "delete_event",
        "Delete a calendar event. Get the event_id from list_events first.",
        {"event_id": _text("The event's id from list_events")},
        ["event_id"],
    ),
    _declare(
        "list_tasks",
        "List the user's to-do tasks (open ones by default).",
        {"include_completed": {"type": "boolean", "description": "Also show finished tasks"}},
    ),
    _declare(
        "add_task",
        "Add a to-do task.",
        {
            "title": _text("Task title"),
            "due_date": _text("Optional due date, YYYY-MM-DD"),
            "notes": _text("Optional notes"),
        },
        ["title"],
    ),
    _declare(
        "complete_task",
        "Mark a task as done. Get the task_id from list_tasks first.",
        {"task_id": _text("The task's id from list_tasks")},
        ["task_id"],
    ),
    _declare(
        "delete_task",
        "Delete a task. Get the task_id from list_tasks first.",
        {"task_id": _text("The task's id from list_tasks")},
        ["task_id"],
    ),
]


# ---------------------------------------------------------------------------
# 2. Small helpers for dates and for shaping Google's replies
# ---------------------------------------------------------------------------
def _parse_when(value: str, tz: str):
    """'2026-11-03' -> date (all-day). '2026-11-03T15:00' -> naive datetime in `tz`."""
    value = value.strip()
    if len(value) == 10:
        return date.fromisoformat(value)
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is not None:  # convert explicit offsets into the calendar's time zone
        dt = dt.astimezone(ZoneInfo(tz)).replace(tzinfo=None)
    return dt


def _event_times(start: str, end: str, tz: str) -> tuple[dict, dict]:
    """Turn the model's start/end text into Google's start/end objects."""
    first = _parse_when(start, tz)

    if isinstance(first, datetime):  # timed event
        last = _parse_when(end, tz) if end else first + timedelta(hours=1)
        if not isinstance(last, datetime):
            raise ValueError("start has a time but end is only a date")
        if last <= first:
            raise ValueError("end must be after start")
        return (
            {"dateTime": first.isoformat(), "timeZone": tz},
            {"dateTime": last.isoformat(), "timeZone": tz},
        )

    last = _parse_when(end, tz) if end else first  # all-day event (end date is inclusive)
    if isinstance(last, datetime):
        raise ValueError("start is only a date but end has a time")
    if last < first:
        raise ValueError("end must not be before start")
    return {"date": first.isoformat()}, {"date": (last + timedelta(days=1)).isoformat()}


def _event_summary(event: dict) -> dict:
    """The few fields Gemini needs. (Descriptions are left out on purpose.)"""
    start, end = event.get("start", {}), event.get("end", {})
    summary = {
        "id": event.get("id"),
        "title": event.get("summary", "(no title)"),
        "start": start.get("dateTime") or start.get("date"),
        "end": end.get("dateTime") or end.get("date"),
        "all_day": "date" in start,
        "location": event.get("location"),
    }
    return {k: v for k, v in summary.items() if v is not None}


def _task_summary(task: dict) -> dict:
    summary = {
        "id": task.get("id"),
        "title": task.get("title", "(no title)"),
        "due": (task.get("due") or "")[:10] or None,
        "notes": (task.get("notes") or "")[:200] or None,
        "done": task.get("status") == "completed",
    }
    return {k: v for k, v in summary.items() if v is not None}


# ---------------------------------------------------------------------------
# 3. The tools themselves
# ---------------------------------------------------------------------------
class GoogleTools:
    def __init__(self) -> None:
        self.tool = types.Tool(function_declarations=TOOL_DECLARATIONS)
        # Your calendar's time zone is read from Google; set TIMEZONE in .env to override.
        self._tz_name: str | None = os.getenv("TIMEZONE") or None
        self._actions = {
            "list_events": self._list_events,
            "create_event": self._create_event,
            "update_event": self._update_event,
            "delete_event": self._delete_event,
            "list_tasks": self._list_tasks,
            "add_task": self._add_task,
            "complete_task": self._complete_task,
            "delete_task": self._delete_task,
        }

    # ---- used by brain.py -------------------------------------------------
    async def now_text(self) -> str:
        """Current local date and time, e.g. 'Sunday, 2026-11-01 17:05 (Asia/Manila)'."""
        tz = await asyncio.to_thread(self._timezone)
        return f"{datetime.now(ZoneInfo(tz)):%A, %Y-%m-%d %H:%M} ({tz})"

    async def run(self, name: str, args: dict) -> dict:
        """Run one tool. Always returns a dict: {"result": ...} or {"error": "..."}."""
        action = self._actions.get(name)
        if action is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            return {"result": await asyncio.to_thread(action, **args)}
        except GoogleAuthError as exc:
            return {"error": str(exc)}
        except HttpError as exc:
            log.warning("Google API error in %s: %s", name, exc)
            return {"error": f"Google API error {exc.resp.status}: {exc.reason}"}
        except (ValueError, TypeError) as exc:  # bad dates, unexpected arguments
            return {"error": f"Bad input: {exc}"}

    # ---- plumbing ----------------------------------------------------------
    def _calendar(self):
        return build("calendar", "v3", credentials=get_credentials(), cache_discovery=False)

    def _tasks(self):
        return build("tasks", "v1", credentials=get_credentials(), cache_discovery=False)

    def _timezone(self) -> str:
        if not self._tz_name:
            try:
                reply = self._calendar().events().list(calendarId="primary", maxResults=1).execute()
                self._tz_name = reply.get("timeZone", "UTC")
            except Exception:  # noqa: BLE001 - never block a chat over the clock
                log.exception("Couldn't read the calendar's time zone; using UTC for now")
                return "UTC"
        try:
            ZoneInfo(self._tz_name)
        except ZoneInfoNotFoundError:
            log.warning("Unknown time zone %r; using UTC", self._tz_name)
            return "UTC"
        return self._tz_name

    # ---- calendar ------------------------------------------------------------
    def _list_events(self, start_date: str, end_date: str = "") -> dict:
        tz = self._timezone()
        zone = ZoneInfo(tz)
        first = date.fromisoformat(start_date.strip()[:10])
        last = date.fromisoformat((end_date or start_date).strip()[:10])
        if last < first:
            raise ValueError("end_date is before start_date")
        if (last - first).days > MAX_RANGE_DAYS:
            raise ValueError(f"date range too long (max {MAX_RANGE_DAYS} days)")

        reply = (
            self._calendar()
            .events()
            .list(
                calendarId="primary",
                timeMin=datetime.combine(first, dtime.min, tzinfo=zone).isoformat(),
                timeMax=datetime.combine(last + timedelta(days=1), dtime.min, tzinfo=zone).isoformat(),
                singleEvents=True,
                orderBy="startTime",
                maxResults=MAX_EVENTS,
            )
            .execute()
        )
        events = [_event_summary(e) for e in reply.get("items", [])]
        return {"timezone": tz, "count": len(events), "events": events}

    def _create_event(
        self, title: str, start: str, end: str = "", location: str = "", description: str = ""
    ) -> dict:
        tz = self._timezone()
        start_obj, end_obj = _event_times(start, end, tz)
        body = {"summary": title, "start": start_obj, "end": end_obj}
        if location:
            body["location"] = location
        if description:
            body["description"] = description
        event = self._calendar().events().insert(calendarId="primary", body=body).execute()
        return {"created": _event_summary(event)}

    def _update_event(
        self,
        event_id: str,
        title: str = "",
        start: str = "",
        end: str = "",
        location: str = "",
        description: str = "",
    ) -> dict:
        body: dict = {}
        if title:
            body["summary"] = title
        if location:
            body["location"] = location
        if description:
            body["description"] = description
        if start:
            body["start"], body["end"] = _event_times(start, end, self._timezone())
        elif end:
            raise ValueError("to change the end time, send both start and end")
        if not body:
            raise ValueError("nothing to change")
        event = (
            self._calendar()
            .events()
            .patch(calendarId="primary", eventId=event_id, body=body)
            .execute()
        )
        return {"updated": _event_summary(event)}

    def _delete_event(self, event_id: str) -> dict:
        self._calendar().events().delete(calendarId="primary", eventId=event_id).execute()
        return {"deleted_event_id": event_id}

    # ---- tasks ---------------------------------------------------------------
    def _list_tasks(self, include_completed: bool = False) -> dict:
        reply = (
            self._tasks()
            .tasks()
            .list(
                tasklist="@default",
                showCompleted=bool(include_completed),
                showHidden=bool(include_completed),
                maxResults=MAX_TASKS,
            )
            .execute()
        )
        tasks = [_task_summary(t) for t in reply.get("items", [])]
        return {"count": len(tasks), "tasks": tasks}

    def _add_task(self, title: str, due_date: str = "", notes: str = "") -> dict:
        body = {"title": title}
        if due_date:
            # Google Tasks only keeps the date part of "due".
            body["due"] = f"{date.fromisoformat(due_date.strip()[:10]).isoformat()}T00:00:00.000Z"
        if notes:
            body["notes"] = notes
        task = self._tasks().tasks().insert(tasklist="@default", body=body).execute()
        return {"created": _task_summary(task)}

    def _complete_task(self, task_id: str) -> dict:
        task = (
            self._tasks()
            .tasks()
            .patch(tasklist="@default", task=task_id, body={"status": "completed"})
            .execute()
        )
        return {"completed": _task_summary(task)}

    def _delete_task(self, task_id: str) -> dict:
        self._tasks().tasks().delete(tasklist="@default", task=task_id).execute()
        return {"deleted_task_id": task_id}
