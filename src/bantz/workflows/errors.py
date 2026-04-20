"""
Workflow error hierarchy.
"""
from __future__ import annotations


class WorkflowError(Exception):
    """Base class for all workflow errors."""


class WorkflowNotFoundError(WorkflowError):
    """Raised when a workflow YAML file cannot be located."""


class WorkflowValidationError(WorkflowError):
    """Raised when a workflow YAML fails schema validation."""


class StepExecutionError(WorkflowError):
    """Raised when a step fails after all retries are exhausted."""

    def __init__(self, step_name: str, message: str) -> None:
        self.step_name = step_name
        super().__init__(f"Step '{step_name}': {message}")


class StepTimeoutError(StepExecutionError):
    """Raised when a step exceeds its timeout."""

    def __init__(self, step_name: str, timeout: float) -> None:
        super().__init__(step_name, f"timed out after {timeout}s")
        self.timeout = timeout
