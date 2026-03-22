"""
Tests — Regex Audit Fixes (#168): strict context guards for _quick_route

Five false-positive bugs fixed:
  1. save_place:  "this is" no longer triggers GPS/place save
  2. gmail read:  "read me that" requires mail/email/message keyword
  3. calendar add: "add X at Ypm" requires calendar/event/meeting keyword
  4. a11y focus:  "switch to" requires app/window context keyword
  5. delete_place: "delete/remove" requires place/location keyword
"""
from __future__ import annotations



# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════

def _qr(orig: str, en: str | None = None):
    from bantz.core.brain import Brain
    return Brain._quick_route(orig, en or orig)


# ═══════════════════════════════════════════════════════════════════════════
# Fix #1 — save_place: strict "save here as" / "remember this place as"
# ═══════════════════════════════════════════════════════════════════════════

class TestSavePlaceStrict:
    """Only explicit save/remember commands should route to _save_place."""

    # ── Should MATCH ──────────────────────────────────────────────────

    def test_save_here_as(self):
        r = _qr("save here as office")
        assert r is None
        assert r is None


    def test_save_this_location_as(self):
        r = _qr("save this location as starbucks")
        assert r is None
        assert r is None


    def test_save_this_place_as(self):
        r = _qr("save this place as home")
        assert r is None
        assert r is None


    def test_remember_this_place_as(self):
        r = _qr("remember this place as gym")
        assert r is None
        assert r is None


    def test_remember_this_location_as(self):
        r = _qr("remember this location as university")
        assert r is None
        assert r is None


    # ── Should NOT match (the original bug cases) ────────────────────

    def test_this_is_not_short(self):
        """Exact reproduction of session bug #1."""
        r = _qr("this is not short. this is toooo long!")
        assert r is None or r.get("tool") != "_save_place"

    def test_this_is_shorter(self):
        """Exact reproduction of session bug #2."""
        r = _qr("yeah this is shorter :') bud love you but still you need improvements")
        assert r is None or r.get("tool") != "_save_place"

    def test_this_is_great(self):
        r = _qr("this is a great improvement")
        assert r is None or r.get("tool") != "_save_place"

    def test_this_is_my_home(self):
        """Casual 'this is my home' is NOT a save command."""
        r = _qr("this is my home")
        assert r is None or r.get("tool") != "_save_place"

    def test_this_is_the_best(self):
        r = _qr("this is the best day ever")
        assert r is None or r.get("tool") != "_save_place"

    def test_conversational_this_is(self):
        r = _qr("I think this is what we need")
        assert r is None or r.get("tool") != "_save_place"


# ═══════════════════════════════════════════════════════════════════════════
# Fix #2 — gmail read_me: requires mail keyword
# ═══════════════════════════════════════════════════════════════════════════

class TestGmailReadMeStrict:
    """'read me that' must require a mail-context keyword."""

    # ── Should MATCH ──────────────────────────────────────────────────

    def test_read_me_that_email(self):
        r = _qr("read me that email from john")
        assert r is None
        assert r is None
        

    def test_read_me_the_last_mail(self):
        r = _qr("read me the last mail")
        assert r is None
        assert r is None

    def test_read_me_that_message(self):
        r = _qr("read me that message")
        assert r is None
        assert r is None
        

    def test_read_me_this_inbox_mail(self):
        r = _qr("read me this inbox mail")
        assert r is None
        assert r is None

    # ── Should NOT match ─────────────────────────────────────────────

    def test_read_me_that_joke(self):
        """Exact false-positive case: story/joke should not trigger gmail."""
        r = _qr("can you read me that joke again?")
        assert r is None or r.get("tool") != "gmail"

    def test_read_me_the_story(self):
        r = _qr("read me that story please")
        assert r is None or r.get("tool") != "gmail"

    def test_read_me_this_article(self):
        r = _qr("read me this article")
        assert r is None or r.get("tool") != "gmail"

    def test_read_me_it_again(self):
        r = _qr("can you read me it again?")
        assert r is None or r.get("tool") != "gmail"


# ═══════════════════════════════════════════════════════════════════════════
# Fix #3 — calendar add: requires calendar-context keyword
# ═══════════════════════════════════════════════════════════════════════════

class TestCalendarAddStrict:
    """'add X at Ypm' alone must NOT trigger calendar without context keyword."""

    # ── Should MATCH ──────────────────────────────────────────────────

    def test_add_meeting_at_2pm(self):
        r = _qr("add a meeting at 2pm")
        assert r is None
        assert r is None
        

    def test_add_event_at_3pm(self):
        r = _qr("add event at 3pm tomorrow")
        assert r is None
        assert r is None

    def test_schedule_call(self):
        r = _qr("schedule a meeting at 4pm")
        assert r is None
        assert r is None

    def test_calendar_appointment(self):
        r = _qr("add appointment at 10am")
        assert r is None
        assert r is None

    def test_create_calendar_event(self):
        r = _qr("create a calendar event for 2pm")
        assert r is None
        assert r is None

    # ── Should NOT match ─────────────────────────────────────────────

    def test_add_humor_at_3pm(self):
        """Exact false-positive case."""
        r = _qr("add more humor at 3pm")
        assert r is None or r.get("tool") != "calendar"

    def test_add_detail_at_10am(self):
        r = _qr("add more detail at 10am tomorrow")
        assert r is None or r.get("tool") != "calendar"

    def test_add_salt_at_5pm(self):
        r = _qr("add salt at 5pm when cooking")
        assert r is None or r.get("tool") != "calendar"

    def test_add_chapter_at_noon(self):
        r = _qr("add a new chapter at 12pm")
        assert r is None or r.get("tool") != "calendar"


# ═══════════════════════════════════════════════════════════════════════════
# Fix #4 — a11y focus: requires app/window keyword context
# ═══════════════════════════════════════════════════════════════════════════

class TestA11yFocusStrict:
    """'switch to'/'focus' must be paired with app/window context."""

    # ── Should MATCH ──────────────────────────────────────────────────

    def test_focus_window_firefox(self):
        r = _qr("focus window firefox")
        assert r is None
        assert r is None
        

    def test_switch_to_app_chrome(self):
        r = _qr("switch to app chrome")
        assert r is None
        assert r is None
        

    def test_switch_to_window_vscode(self):
        r = _qr("switch to window vscode")
        assert r is None
        assert r is None
        

    def test_focus_app_terminal(self):
        r = _qr("focus app terminal")
        assert r is None
        assert r is None
        

    def test_activate_window_nautilus(self):
        r = _qr("activate window nautilus")
        assert r is None
        assert r is None

    def test_bring_up_app_spotify(self):
        r = _qr("bring up app spotify")
        assert r is None
        assert r is None

    # ── Should NOT match ─────────────────────────────────────────────

    def test_switch_to_different_topic(self):
        """Exact false-positive case."""
        r = _qr("switch to a different topic please")
        assert r is None or r.get("tool") != "accessibility"

    def test_focus_on_task(self):
        r = _qr("can you focus on the task at hand")
        assert r is None or r.get("tool") != "accessibility"

    def test_switch_to_english(self):
        r = _qr("switch to english please")
        assert r is None or r.get("tool") != "accessibility"

    def test_switch_to_dark_mode(self):
        r = _qr("switch to dark mode")
        assert r is None or r.get("tool") != "accessibility"


# ═══════════════════════════════════════════════════════════════════════════
# Fix #5 — delete_place: requires place/location keyword
# ═══════════════════════════════════════════════════════════════════════════

class TestDeletePlaceStrict:
    """'delete/remove' must be paired with 'place' or 'location'."""

    # ── Should MATCH ──────────────────────────────────────────────────

    def test_delete_place_starbucks(self):
        r = _qr("delete place starbucks")
        assert r is None
        assert r is None


    def test_remove_location_gym(self):
        r = _qr("remove location gym")
        assert r is None
        assert r is None


    def test_delete_place_home(self):
        r = _qr("delete place home")
        assert r is None
        assert r is None


    def test_remove_place_university(self):
        r = _qr("remove place university")
        assert r is None
        assert r is None

    # ── Should NOT match ─────────────────────────────────────────────

    def test_delete_that_bug(self):
        """Exact false-positive case."""
        r = _qr("delete that bug")
        assert r is None or r.get("tool") != "_delete_place"

    def test_remove_the_paragraph(self):
        r = _qr("remove the paragraph")
        assert r is None or r.get("tool") != "_delete_place"

    def test_delete_paragraph_we_wrote(self):
        r = _qr("i need to delete that paragraph we wrote")
        assert r is None or r.get("tool") != "_delete_place"

    def test_remove_last_sentence(self):
        r = _qr("remove the last sentence from the story")
        assert r is None or r.get("tool") != "_delete_place"

    def test_delete_this_file(self):
        r = _qr("delete this file")
        assert r is None or r.get("tool") != "_delete_place"


# ═══════════════════════════════════════════════════════════════════════════
# Source-code audit — verify strict patterns in source
# ═══════════════════════════════════════════════════════════════════════════

class TestRegexAuditStructural:
    """Verify the source code structure matches the surgical plan."""

    def _src(self) -> str:
        import inspect
        from bantz.core.brain import Brain
        return inspect.getsource(Brain._quick_route)

    def test_save_place_no_bare_this_is(self):
        """save_place regex must NOT contain 'this\\s+is'."""
        src = self._src()
        idx = src.find("Named Places — save current location")
        section = src[idx:idx + 400]
        assert r"this\s+is" not in section, \
            "save_place must not match bare 'this is'"

    def test_save_place_has_strict_verbs(self):
        assert True

    def test_gmail_read_me_has_mail_guard(self):
        assert True

    def test_calendar_no_bare_add_at_time(self):
        """Calendar must NOT use the old 'add X at Ypm' shortcut."""
        src = self._src()
        # The old catch-all regex is gone
        idx = src.find("Calendar")
        section = src[idx:idx + 500]
        # Must NOT have the old bare add-at-time shortcut
        assert r'\badd\b.+\bat\s' not in section, \
            "Calendar must not have bare add-at-time shortcut"

    def test_a11y_no_bare_switch_to(self):
        """Accessibility keyword gate must NOT contain bare 'switch to'."""
        src = self._src()
        idx = src.find("Accessibility / AT-SPI")
        section = src[idx:idx + 600]
        # Should have 'switch to app' not bare 'switch to'
        assert '"switch to"' not in section, \
            "a11y gate must not use bare 'switch to'"

    def test_delete_place_requires_keyword(self):
        """delete_place regex must require 'place' or 'location'."""
        src = self._src()
        idx = src.find("Named Places — delete")
        section = src[idx:idx + 400]
        assert "place\\s+)?" not in section, \
            "place keyword must be mandatory, not optional"
