"""Local macOS app integrations for Bad Apple.

All queries run against the user's local Mail, Calendar, and Reminders
applications via AppleScript. No cloud APIs are used.
"""

import subprocess
from pathlib import Path

CONTACTS_FILE = Path("/var/lib/bad_apple/allowed_contacts.json")


def _esc_applescript(value: str) -> str:
    """Escape a string for safe interpolation into an AppleScript string literal.

    AppleScript string literals are delimited by double quotes.  Backslash is
    the escape character.  Injecting a raw `"` or `\\` would let an attacker
    break out of the literal and run arbitrary AppleScript.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _run_applescript(script: str, timeout: int = 30) -> str:
    """Run an AppleScript and return the result or error."""
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        check=False)
        if result.returncode != 0:
            return f"AppleScript error: {result.stderr.strip() or result.stdout.strip()}"
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "AppleScript timed out."
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        return f"AppleScript failed: {e}"


def today_events() -> str:
    """Return today's events from the default Calendar app."""
    script = """
tell application "Calendar"
    set now to current date
    set startOfDay to now - (time of now)
    set endOfDay to startOfDay + 1 * days
    set out to ""
    repeat with cal in calendars
        repeat with ev in (every event of cal whose start date ≥ startOfDay and start date < endOfDay)
            set s to (start date of ev) as string
            set e to (end date of ev) as string
            set out to out & s & " — " & (summary of ev) & " (" & e & ")" & return
        end repeat
    end repeat
    if out is "" then
        return "No events found for today."
    end if
    return out
end tell
"""
    return _run_applescript(script)


def upcoming_events(days: int = 7, limit: int = 20) -> str:
    """Return upcoming calendar events for the next N days."""
    script = f"""
tell application "Calendar"
    set now to current date
    set startOfDay to now - (time of now)
    set endRange to startOfDay + {days} * days
    set out to ""
    set countEvents to 0
    repeat with cal in calendars
        repeat with ev in (every event of cal whose start date ≥ now and start date < endRange)
            set s to (start date of ev) as string
            set out to out & s & " — " & (summary of ev) & return
            set countEvents to countEvents + 1
            if countEvents ≥ {limit} then exit repeat
        end repeat
        if countEvents ≥ {limit} then exit repeat
    end repeat
    if out is "" then
        return "No upcoming events."
    end if
    return out
end tell
"""
    return _run_applescript(script)


def list_reminders(list_name: str = "", completed: bool = False, limit: int = 20) -> str:
    """Return reminders from the local Reminders app."""
    filter_expr = "whose completed is " + ("true" if completed else "false")
    if list_name:
        list_ref = f'list "{_esc_applescript(list_name)}"'
    else:
        list_ref = "default list"
    script = f"""
tell application "Reminders"
    set out to ""
    set countRem to 0
    repeat with r in (every reminder of {list_ref} {filter_expr})
        if name of r is not "" then
            set out to out & "- " & (name of r)
            if due date of r is not missing value then
                set out to out & " (due " & (due date of r as string) & ")"
            end if
            set out to out & return
            set countRem to countRem + 1
            if countRem ≥ {limit} then exit repeat
        end if
    end repeat
    if out is "" then
        return "No reminders found."
    end if
    return out
end tell
"""
    return _run_applescript(script)


def unread_emails(limit: int = 10) -> str:
    """Return sender/subject lines of unread Mail messages."""
    script = f"""
tell application "Mail"
    set unreadMessages to (every message of inbox whose read status is false)
    if (count of unreadMessages) is 0 then
        return "No unread mail."
    end if
    set out to ""
    set mCount to 0
    repeat with m in unreadMessages
        set out to out & "From: " & (sender of m as string) & return & "Subject: " & (subject of m as string) & return & return
        set mCount to mCount + 1
        if mCount ≥ {limit} then exit repeat
    end repeat
    return out
end tell
"""
    return _run_applescript(script)


def search_mail(query: str, limit: int = 10) -> str:
    """Search local Mail by subject or sender."""
    safe_query = _esc_applescript(query)
    script = f"""
tell application "Mail"
    set results to (every message of inbox whose subject contains "{safe_query}" or sender contains "{safe_query}")
    if (count of results) is 0 then
        return "No matching mail."
    end if
    set out to ""
    set mCount to 0
    repeat with m in results
        set out to out & "From: " & (sender of m as string) & return & "Subject: " & (subject of m as string) & return & "Date: " & (date received of m as string) & return & return
        set mCount to mCount + 1
        if mCount ≥ {limit} then exit repeat
    end repeat
    return out
end tell
"""
    return _run_applescript(script)


def add_reminder(name: str, list_name: str = "", due: str = "") -> str:
    """Add a reminder to the local Reminders app."""
    safe_name = _esc_applescript(name)
    list_ref = f'list "{_esc_applescript(list_name)}"' if list_name else "default list"
    due_attr = f' with due date (date "{_esc_applescript(due)}")' if due else ""
    script = f"""
tell application "Reminders"
    tell {list_ref}
        make new reminder with properties {{name:"{safe_name}"}}{due_attr}
    end tell
    return "Reminder added."
end tell
"""
    return _run_applescript(script)
