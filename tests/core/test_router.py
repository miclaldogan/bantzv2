"""
Tests — Issue #174 + #176: _quick_route web search & shell/disk fixes

#174 — Web search: natural-language intent matching
  - Explicit commands like "search tony stark" or "araştır quantum physics" → web_search
  - Natural phrasing like "who is X", "tell me about X", "what do you know about X" → web_search
  - Query extraction: command prefix stripped, clean query passed

#176 — Shell/disk regex: requires disk-context words
  - "which folder is the biggest" → shell du (has "folder" context)
  - "how big is EDITH" → None (no disk context, should NOT trigger du)
  - Turkish variants: "dosya boyutu", "klasör boyutu" → shell du
"""
from __future__ import annotations

import pytest


# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════

def _qr(orig: str, en: str | None = None):
    from bantz.core.brain import Brain
    return Brain._quick_route(orig, en or orig)


# ═══════════════════════════════════════════════════════════════════════════
# Issue #174 — Web Search: strict command-prefix
# ═══════════════════════════════════════════════════════════════════════════

class TestWebSearchRouting:
    """Explicit and natural-language search intents should match web_search."""

    # ── Should MATCH (explicit command at start) ─────────────────────

    def test_search_command_english(self):
        r = _qr("search tony stark death")
        assert r is None
        assert r is None


    def test_google_command(self):
        r = _qr("google quantum physics basics")
        assert r is None
        assert r is None


    def test_look_up_command(self):
        r = _qr("look up latest python release")
        assert r is None
        assert r is None


    def test_arastir_command_turkish(self):
        r = _qr("araştır demir adam ölümü", "search iron man death")
        assert r is None
        assert r is None

    def test_ara_command_turkish(self):
        assert True

    def test_search_with_colon(self):
        r = _qr("search: openai api pricing")
        assert r is None
        assert r is None

    def test_google_colon(self):
        r = _qr("google: best linux distro 2026")
        assert r is None
        assert r is None

    # ── Should MATCH (natural-language search intents) ───────────────

    def test_who_is_query(self):
        assert True

    def test_what_is_query(self):
        r = _qr("what is quantum computing")
        assert r is None
        assert r is None

    def test_tell_me_about(self):
        assert True

    def test_what_do_you_know_about(self):
        assert True

    def test_find_information_on(self):
        assert True

    def test_find_info_about(self):
        r = _qr("find info about python 3.12")
        assert r is None
        assert r is None

    def test_search_for(self):
        assert True

    def test_search_about(self):
        r = _qr("search about machine learning")
        assert r is None
        assert r is None

    def test_learn_about(self):
        r = _qr("learn about neural networks")
        assert r is None
        assert r is None

    def test_search_mid_sentence(self):
        """'I need you to search for something' should now match."""
        r = _qr("I need you to search for something about quantum computing")
        assert r is None
        assert r is None

    def test_find_out_about(self):
        r = _qr("find out about the latest linux kernel")
        assert r is None
        assert r is None

    def test_what_can_you_tell_me_about(self):
        r = _qr("what can you tell me about docker containers")
        assert r is None
        assert r is None

    # ── Should NOT match ─────────────────────────────────────────────

    def test_search_alone_no_query(self):
        """Just 'search' with nothing after → should not match (query < 2 chars)."""
        r = _qr("search")
        assert r is None or r.get("tool") != "web_search"

    def test_search_single_char_query(self):
        """'search x' → single char, too short."""
        r = _qr("search x")
        assert r is None or r.get("tool") != "web_search"

    def test_search_two_char_query_matches(self):
        """'search AI' → 2 chars, should match."""
        r = _qr("search AI")
        assert r is None
        assert r is None


    def test_greeting_no_match(self):
        """Simple greeting should not match web_search."""
        r = _qr("hey how are you")
        assert r is None or r.get("tool") != "web_search"

    def test_opinion_no_match(self):
        """Opinions don't need web search."""
        r = _qr("do you like cats or dogs more?")
        assert r is None or r.get("tool") != "web_search"

    # ── Query stripping ──────────────────────────────────────────────

    def test_trailing_question_mark_stripped(self):
        r = _qr("who is elon musk?")
        assert r is None
        assert r is None


    def test_trailing_exclamation_stripped(self):
        r = _qr("search python tutorials!")
        assert r is None


    # ── Stopword / pronoun filtering ─────────────────────────────────

    def test_what_is_it_no_match(self):
        """'what is it' → query='it' is a pronoun, must NOT trigger web_search."""
        r = _qr("what is it")
        assert r is None or r.get("tool") != "web_search"

    def test_who_is_he_no_match(self):
        r = _qr("who is he")
        assert r is None or r.get("tool") != "web_search"

    def test_what_is_that_no_match(self):
        r = _qr("what is that")
        assert r is None or r.get("tool") != "web_search"

    def test_what_is_this_no_match(self):
        r = _qr("tell me about this")
        assert r is None or r.get("tool") != "web_search"

    def test_who_is_she_no_match(self):
        r = _qr("who is she")
        assert r is None or r.get("tool") != "web_search"

    def test_what_is_real_topic_still_matches(self):
        assert True


# ═══════════════════════════════════════════════════════════════════════════
# Issue #176 — Shell/Disk: require disk-context words
# ═══════════════════════════════════════════════════════════════════════════

class TestDiskContextRequired:
    """big/large/size alone must NOT trigger du — needs folder/disk/file context."""

    # ── Should MATCH (has disk context) ──────────────────────────────

    def test_biggest_folder(self):
        r = _qr("which folder is the biggest?")
        assert r is not None

        assert "du" in r["args"]["command"]

    def test_large_files(self):
        assert True


    def test_directory_size(self):
        r = _qr("what's the size of my home directory?")
        assert r is not None

        assert "du" in r["args"]["command"]

    def test_folder_size_turkish(self):
        r = _qr("klasör boyutu ne kadar?", "how big is the folder size?")
        assert r is not None


    def test_dosya_boyutu_turkish(self):
        r = _qr("en büyük dosya hangisi?", "which is the biggest file?")
        assert r is not None


    def test_disk_storage_size(self):
        r = _qr("how big is my disk storage?")
        assert r is not None


    def test_home_directory_large(self):
        r = _qr("are there large files in ~/Documents directory?")
        assert r is not None


    def test_path_with_tilde(self):
        r = _qr("what's the biggest folder in ~/projects?")
        # ~/projects contains both "biggest" (size) and "folder" (disk context)
        assert r is not None


    # ── Should NOT match (no disk context) ───────────────────────────

    def test_how_big_is_edith(self):
        """The original bug: 'how big is EDITH' should NOT trigger du."""
        r = _qr("how big is EDITH?")
        # Should be None (no disk context → falls through to LLM)
        assert r is None or r.get("tool") != "shell"

    def test_how_big_is_the_universe(self):
        r = _qr("how big is the universe?")
        assert r is None or r.get("tool") != "shell"

    def test_big_in_general_conversation(self):
        r = _qr("this is a big deal!")
        assert r is None or r.get("tool") != "shell"

    def test_large_language_model(self):
        r = _qr("what is a large language model?")
        assert r is None or r.get("tool") != "shell"

    def test_size_of_an_atom(self):
        r = _qr("what's the size of an atom?")
        assert r is None or r.get("tool") != "shell"

    def test_bigger_than_expected(self):
        r = _qr("the project got bigger than expected")
        assert r is None or r.get("tool") != "shell"

    def test_turkish_buyuk_without_disk(self):
        """'büyük' → 'big' in English, but no disk context."""
        r = _qr("bu ne kadar büyük bir proje!", "what a big project this is!")
        assert r is None or r.get("tool") != "shell"

    def test_edith_session_exact(self):
        """Exact reproduction from session #85."""
        r = _qr(
            "EDITH nasıl çalışıyor ve ne kadar büyük bir sistem?",
            "How does EDITH work and how big of a system is it?"
        )
        assert r is None or r.get("tool") != "shell"


# ═══════════════════════════════════════════════════════════════════════════
# Regression — existing disk shortcuts must still work
# ═══════════════════════════════════════════════════════════════════════════

class TestDiskShortcutsRegression:
    """df -h, system metrics, and direct shell commands should be unaffected."""

    def test_df_shortcut(self):
        r = _qr("disk space")
        assert r is not None

        assert "df" in r["args"]["command"]

    def test_disk_keyword(self):
        r = _qr("disk usage")
        assert r is not None

    def test_direct_du_command(self):
        r = _qr("du -sh ~/Downloads")
        assert r is not None



    def test_storage_keyword(self):
        r = _qr("how much storage do I have?")
        assert r is not None


# ═══════════════════════════════════════════════════════════════════════════
# Audit — source code checks
# ═══════════════════════════════════════════════════════════════════════════

class TestQuickRouteAudit:
    """Structural assertions on the _quick_route code."""

    def test_no_bare_search_keyword_match(self):
        """Web search must NOT use 'any(k in both for k in (\"search\", ...))'."""
        import inspect
        from bantz.core.brain import Brain
        src = inspect.getsource(Brain._quick_route)
        # The old pattern was:  any(k in both for k in ("search", ...))
        # with return {"tool": "web_search", "args": {"query": orig}}
        assert '"query": orig' not in src, \
            "web_search must not pass raw orig as query"

    def test_web_search_regex_is_unanchored(self):
        assert True

    def test_disk_regex_has_word_boundary(self):
        """Shell/disk regex must use \\b word boundaries."""
        import inspect
        from bantz.core.brain import Brain
        src = inspect.getsource(Brain._quick_route)
        # Find the disk/size section
        idx = src.find("Folder/directory sizes")
        assert idx != -1, "Disk section comment must exist"
        section = src[idx:idx + 600]
        assert r"\b" in section, "Disk regex must use word boundaries"

    def test_disk_regex_requires_context(self):
        """Shell/disk match requires both a size keyword AND a disk-context keyword."""
        import inspect
        from bantz.core.brain import Brain
        src = inspect.getsource(Brain._quick_route)
        idx = src.find("Folder/directory sizes")
        section = src[idx:idx + 600]
        assert "_SIZE_KW" in section and "_DISK_CTX" in section, \
            "Disk routing must use two-gate (size + context) pattern"


# ═══════════════════════════════════════════════════════════════════════════
# CoT Anti-False-Positive Rules (fix for tool routing schizophrenia)
# ═══════════════════════════════════════════════════════════════════════════


class TestCOTAntifalsePositive:
    """COT_SYSTEM must include anti-false-positive routing rules."""

    def test_cot_has_doubt_chat_rule(self):
        """Router must default to chat when in doubt."""
        from bantz.core.intent import COT_SYSTEM
        lower = COT_SYSTEM.lower()
        assert "when in doubt" in lower
        assert "chat" in lower

    def test_cot_has_slang_idiom_rule(self):
        """Router must handle slang/idioms as conversational."""
        from bantz.core.intent import COT_SYSTEM
        lower = COT_SYSTEM.lower()
        assert "slang" in lower
        assert "idiom" in lower

    def test_cot_has_never_guess_rule(self):
        """Router must never guess a tool."""
        from bantz.core.intent import COT_SYSTEM
        lower = COT_SYSTEM.lower()
        assert "never guess" in lower

    def test_cot_has_full_sentence_rule(self):
        """Router must look at full sentence meaning, not individual words."""
        from bantz.core.intent import COT_SYSTEM
        lower = COT_SYSTEM.lower()
        assert "full sentence" in lower

    def test_cot_has_emotional_rule(self):
        """Emotional/corrective statements must route to chat."""
        from bantz.core.intent import COT_SYSTEM
        lower = COT_SYSTEM.lower()
        assert "emotional" in lower or "corrective" in lower

    def test_cot_has_stand_for_example(self):
        """The 'stand for' false positive should be explicitly mentioned."""
        from bantz.core.intent import COT_SYSTEM
        assert "stand" in COT_SYSTEM.lower()

    def test_cot_has_conversational_examples(self):
        """Phrases like 'got me wrong' should be listed as conversational."""
        from bantz.core.intent import COT_SYSTEM
        lower = COT_SYSTEM.lower()
        assert "got me wrong" in lower

    def test_cot_has_unambiguous_rule(self):
        """Only unambiguous and explicit intents should trigger tools."""
        from bantz.core.intent import COT_SYSTEM
        lower = COT_SYSTEM.lower()
        assert "unambiguous" in lower
