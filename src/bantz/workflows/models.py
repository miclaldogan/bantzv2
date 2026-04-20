"""
Pydantic models for the YAML workflow schema.

Example YAML:
    name: morning-briefing
    description: "Collect weather + news + calendar for a morning summary"
    version: "1.0"
    triggers:
      - schedule: "08:00"
      - manual: true
    inputs:
      city:
        type: string
        default: ""
    steps:
      - name: get_weather
        action: tool
        tool: weather
        args:
          city: "{{ inputs.city }}"
      - name: get_news
        action: tool
        tool: news
        args:
          query: "top headlines"
      - name: get_events
        action: tool
        tool: calendar
        args:
          action: today
      - name: summarize
        action: ask_llm
        prompt: |
          Create a morning briefing from:
          Weather: {{ steps.get_weather.output }}
          News: {{ steps.get_news.output }}
          Events: {{ steps.get_events.output }}
    on_failure: ask_llm
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# ── Step retry policy ────────────────────────────────────────────────────────

class RetryPolicy(BaseModel):
    max_retries: int = Field(0, ge=0, le=10)
    delay_seconds: float = Field(1.0, ge=0)


# ── Individual step definition ────────────────────────────────────────────────

class StepDef(BaseModel):
    """A single step in a workflow.

    Supported actions:
      - ``tool``           : invoke a registered Bantz tool
      - ``shell_command``  : run a bash command (wraps shell tool)
      - ``http_request``   : make an HTTP request (GET/POST)
      - ``ask_llm``        : generate text via the LLM
      - ``conditional``    : branch based on a simple expression
      - ``set_variable``   : assign a value to a context variable
    """

    name: str = Field(..., min_length=1, max_length=64)
    action: Literal[
        "tool",
        "shell_command",
        "http_request",
        "ask_llm",
        "conditional",
        "set_variable",
    ]

    # ── tool action ───────────────────────────────────────────────────────
    tool: str = ""
    args: dict[str, Any] = Field(default_factory=dict)

    # ── shell_command action ──────────────────────────────────────────────
    command: str = ""

    # ── http_request action ───────────────────────────────────────────────
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET"
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict[str, Any] = Field(default_factory=dict)

    # ── ask_llm action ───────────────────────────────────────────────────
    prompt: str = ""

    # ── conditional action ────────────────────────────────────────────────
    condition: str = ""           # e.g. "steps.get_weather.success == true"
    then_step: str = ""           # step name to jump to if true
    else_step: str = ""           # step name to jump to if false

    # ── set_variable action ───────────────────────────────────────────────
    variable: str = ""
    value: str = ""

    # ── common fields ─────────────────────────────────────────────────────
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    timeout_seconds: float = Field(30.0, ge=1, le=600)
    depends_on: list[str] = Field(default_factory=list)
    description: str = ""

    @model_validator(mode="after")
    def _validate_action_fields(self) -> "StepDef":
        a = self.action
        if a == "tool" and not self.tool:
            raise ValueError(f"Step '{self.name}': action=tool requires 'tool' field")
        if a == "shell_command" and not self.command:
            raise ValueError(f"Step '{self.name}': action=shell_command requires 'command' field")
        if a == "http_request" and not self.url:
            raise ValueError(f"Step '{self.name}': action=http_request requires 'url' field")
        if a == "ask_llm" and not self.prompt:
            raise ValueError(f"Step '{self.name}': action=ask_llm requires 'prompt' field")
        if a == "conditional" and not self.condition:
            raise ValueError(f"Step '{self.name}': action=conditional requires 'condition' field")
        if a == "set_variable" and not self.variable:
            raise ValueError(f"Step '{self.name}': action=set_variable requires 'variable' field")
        return self


# ── Input definition ──────────────────────────────────────────────────────────

class InputDef(BaseModel):
    type: Literal["string", "int", "float", "bool"] = "string"
    default: Any = ""
    description: str = ""
    required: bool = False


# ── Trigger definition ────────────────────────────────────────────────────────

class TriggerDef(BaseModel):
    schedule: str = ""        # cron-like or "HH:MM"
    manual: bool = False      # can be manually triggered
    event: str = ""           # event bus event name


# ── Top-level workflow definition ─────────────────────────────────────────────

class WorkflowDef(BaseModel):
    """Complete workflow definition as loaded from YAML."""

    name: str = Field(..., min_length=1, max_length=128)
    description: str = ""
    version: str = "1.0"
    triggers: list[TriggerDef] = Field(default_factory=list)
    inputs: dict[str, InputDef] = Field(default_factory=dict)
    steps: list[StepDef] = Field(..., min_length=1)
    on_failure: Literal["abort", "ask_llm", "continue"] = "abort"
    timeout_seconds: float = Field(300.0, ge=1, le=3600)

    @model_validator(mode="after")
    def _validate_step_names_unique(self) -> "WorkflowDef":
        names = [s.name for s in self.steps]
        dupes = [n for n in names if names.count(n) > 1]
        if dupes:
            raise ValueError(f"Duplicate step names: {set(dupes)}")
        return self

    @model_validator(mode="after")
    def _validate_depends_on_exist(self) -> "WorkflowDef":
        known = {s.name for s in self.steps}
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in known:
                    raise ValueError(
                        f"Step '{step.name}' depends_on unknown step '{dep}'"
                    )
        return self


# ── Runtime result types ──────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Result of executing a single workflow step."""
    step_name: str
    success: bool
    output: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    duration_ms: float = 0.0


@dataclass
class WorkflowResult:
    """Result of executing an entire workflow."""
    workflow_name: str
    success: bool
    steps: list[StepResult] = field(default_factory=list)
    final_output: str = ""
    error: str = ""
    total_duration_ms: float = 0.0
    variables: dict[str, Any] = field(default_factory=dict)
