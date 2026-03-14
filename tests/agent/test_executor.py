"""Tests for executor context-passing (#215) and thought-leakage scrubbing (#214).

Covers:
- strip_thinking() scrubbing (various formats, nested, edge cases)
- _inject_context() recursive resolution with rich context_store
- {step_N_params_KEY} dotted-param access (folder-path chaining scenario)
- {step_N_output} canonical replacement
- Hallucinated placeholder fallback
- read_url URL extraction from prose
- _replace_placeholders() edge cases
"""

from __future__ import annotations

import pytest

from bantz.core.intent import strip_thinking
from bantz.agent.executor import PlanExecutor, _replace_placeholders


# ── strip_thinking tests (#214) ──────────────────────────────────────────────


class TestStripThinking:
    """Verify <thinking> blocks are removed from LLM output."""

    def test_simple_block(self):
        raw = "<thinking>internal reasoning</thinking>Hello!"
        assert strip_thinking(raw) == "Hello!"

    def test_multiline_block(self):
        raw = (
            "<thinking>\nI need to plan.\nLet me think step by step.\n"
            "</thinking>\n{\"intent\": \"search\"}"
        )
        result = strip_thinking(raw)
        assert "<thinking>" not in result
        assert '{"intent": "search"}' in result

    def test_multiple_blocks(self):
        raw = (
            "<thinking>first</thinking>A"
            "<thinking>second</thinking>B"
        )
        assert strip_thinking(raw) == "AB"

    def test_no_thinking_unchanged(self):
        raw = '{"intent": "chat", "query": "hello"}'
        assert strip_thinking(raw) == raw

    def test_empty_thinking_block(self):
        raw = "<thinking></thinking>Result"
        assert strip_thinking(raw) == "Result"

    def test_trailing_whitespace_removed(self):
        raw = "<thinking>stuff</thinking>   \nClean"
        result = strip_thinking(raw)
        assert result == "Clean"

    def test_nested_angle_brackets(self):
        """<thinking> block containing angle-bracket-like content."""
        raw = "<thinking>if x < 5 and y > 3 then...</thinking>Output"
        assert strip_thinking(raw) == "Output"

    def test_preserves_other_xml_tags(self):
        raw = "<result>data</result>"
        assert strip_thinking(raw) == "<result>data</result>"

    def test_empty_string(self):
        assert strip_thinking("") == ""

    def test_only_thinking_block(self):
        raw = "<thinking>just thinking</thinking>"
        assert strip_thinking(raw).strip() == ""


# ── _inject_context tests (#215) ─────────────────────────────────────────────


class TestInjectContext:
    """Verify _inject_context resolves placeholders from context_store."""

    def test_canonical_output_replacement(self):
        """Standard {step_N_output} is replaced with step output."""
        params = {"content": "{step_1_output}"}
        ctx = {1: {"params": {}, "output": "Hello World"}}
        result = PlanExecutor._inject_context(params, ctx)
        assert result["content"] == "Hello World"

    def test_params_key_replacement(self):
        """Path chaining: {step_N_params_KEY} resolves to params[KEY]."""
        params = {"path": "{step_1_params_folder_path}/summary.txt"}
        ctx = {1: {"params": {"folder_path": "~/research"}, "output": "Success"}}
        result = PlanExecutor._inject_context(params, ctx)
        assert result["path"] == "~/research/summary.txt"

    def test_hallucinated_placeholder_fallback(self):
        """Unknown keys fall back to step output (LLM hallucination)."""
        params = {"url": "{step_1_best_url}"}
        ctx = {1: {"params": {}, "output": "https://example.com"}}
        result = PlanExecutor._inject_context(params, ctx)
        assert result["url"] == "https://example.com"

    def test_multiple_steps(self):
        """Placeholders referencing different steps resolve independently."""
        params = {
            "content": "{step_1_output}",
            "path": "{step_2_params_file_path}",
        }
        ctx = {
            1: {"params": {}, "output": "Chapter 1 content"},
            2: {"params": {"file_path": "~/docs/ch1.txt"}, "output": "Done"},
        }
        result = PlanExecutor._inject_context(params, ctx)
        assert result["content"] == "Chapter 1 content"
        assert result["path"] == "~/docs/ch1.txt"

    def test_recursive_nested_dict(self):
        """Placeholders inside nested dicts are resolved."""
        params = {
            "outer": {
                "inner": "{step_1_output}",
                "deep": {"leaf": "{step_2_output}"},
            }
        }
        ctx = {
            1: {"params": {}, "output": "alpha"},
            2: {"params": {}, "output": "beta"},
        }
        result = PlanExecutor._inject_context(params, ctx)
        assert result["outer"]["inner"] == "alpha"
        assert result["outer"]["deep"]["leaf"] == "beta"

    def test_recursive_list(self):
        """Placeholders inside lists are resolved."""
        params = {"items": ["{step_1_output}", "literal", "{step_2_output}"]}
        ctx = {
            1: {"params": {}, "output": "first"},
            2: {"params": {}, "output": "second"},
        }
        result = PlanExecutor._inject_context(params, ctx)
        assert result["items"] == ["first", "literal", "second"]

    def test_non_string_values_unchanged(self):
        """Numeric/boolean values pass through without modification."""
        params = {"count": 5, "active": True, "label": "{step_1_output}"}
        ctx = {1: {"params": {}, "output": "resolved"}}
        result = PlanExecutor._inject_context(params, ctx)
        assert result["count"] == 5
        assert result["active"] is True
        assert result["label"] == "resolved"

    def test_no_placeholders_unchanged(self):
        """Params without placeholders are returned as-is."""
        params = {"content": "plain text", "path": "~/file.txt"}
        ctx = {1: {"params": {}, "output": "unused"}}
        result = PlanExecutor._inject_context(params, ctx)
        assert result == {"content": "plain text", "path": "~/file.txt"}

    def test_empty_context_store(self):
        """Placeholders remain unresolved when context_store is empty."""
        params = {"content": "{step_1_output}"}
        result = PlanExecutor._inject_context(params, {})
        assert result["content"] == "{step_1_output}"

    def test_mixed_canonical_and_params(self):
        """Both canonical output and params-key in same params dict."""
        params = {
            "title": "{step_1_output}",
            "path": "{step_1_params_folder_path}/report.md",
        }
        ctx = {1: {"params": {"folder_path": "~/work"}, "output": "Project Report"}}
        result = PlanExecutor._inject_context(params, ctx)
        assert result["title"] == "Project Report"
        assert result["path"] == "~/work/report.md"

    def test_folder_chaining_scenario(self):
        """The exact folder-creation → file-save chaining bug from #215."""
        # Step 1: create_folder with folder_path
        # Step 2: save_file using {step_1_params_folder_path}/filename
        ctx = {
            1: {
                "params": {"folder_path": "~/research"},
                "output": "Success: Folder created.",
            }
        }
        params = {
            "content": "{step_2_output}",  # won't resolve (step 2 doesn't exist yet)
            "file_path": "{step_1_params_folder_path}/quantum_summary.txt",
        }
        result = PlanExecutor._inject_context(params, ctx)
        assert result["file_path"] == "~/research/quantum_summary.txt"
        # Unresolvable placeholder stays as-is
        assert result["content"] == "{step_2_output}"


# ── _replace_placeholders tests ──────────────────────────────────────────────


class TestReplacePlaceholders:
    """Verify placeholder resolution with read_url URL extraction and fallback."""

    def test_read_url_extracts_url_from_prose(self):
        """For read_url tool, extract first HTTP URL from prose output."""
        replacements = {"step_1_output": "Check out https://arxiv.org/paper123 for details."}
        ctx = {1: {"params": {}, "output": "Check out https://arxiv.org/paper123 for details."}}
        result = _replace_placeholders(
            "{step_1_output}", replacements, ctx, tool_name="read_url"
        )
        # read_url should still return the full output for canonical {step_N_output}
        assert "https://arxiv.org/paper123" in result

    def test_read_url_hallucinated_key_extracts_url(self):
        """read_url + hallucinated key → extract URL from prose."""
        ctx = {1: {"params": {}, "output": "Found it at https://example.com/page today."}}
        replacements = {"step_1_output": ctx[1]["output"]}
        result = _replace_placeholders(
            "{step_1_best_url}", replacements, ctx, tool_name="read_url"
        )
        assert result == "https://example.com/page"

    def test_non_read_url_no_url_extraction(self):
        """Non-read_url tools don't extract URLs, return full output."""
        ctx = {1: {"params": {}, "output": "Visit https://example.com for more."}}
        replacements = {"step_1_output": ctx[1]["output"]}
        result = _replace_placeholders(
            "{step_1_output}", replacements, ctx, tool_name="save_file"
        )
        assert result == "Visit https://example.com for more."

    def test_multiple_placeholders_in_one_string(self):
        """A single string with multiple placeholders all get resolved."""
        replacements = {
            "step_1_params_folder_path": "/home/user",
            "step_2_output": "report.md",
        }
        ctx = {
            1: {"params": {"folder_path": "/home/user"}, "output": "ok"},
            2: {"params": {}, "output": "report.md"},
        }
        result = _replace_placeholders(
            "{step_1_params_folder_path}/{step_2_output}",
            replacements, ctx, tool_name="",
        )
        assert result == "/home/user/report.md"

    def test_output_truncated_to_2000(self):
        """Replacement output is capped at 2000 chars."""
        long_output = "x" * 5000
        replacements = {"step_1_output": long_output}
        ctx = {1: {"params": {}, "output": long_output}}
        result = _replace_placeholders(
            "{step_1_output}", replacements, ctx, tool_name=""
        )
        assert len(result) == 2000

    def test_nonexistent_step_left_unresolved(self):
        """Reference to a step that doesn't exist stays as placeholder."""
        result = _replace_placeholders(
            "{step_99_output}", {}, {}, tool_name=""
        )
        assert result == "{step_99_output}"
