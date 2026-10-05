"""Validated plans, completion messages, and execution budgets."""

from dataclasses import dataclass
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def _strip_schema_defaults(schema: dict) -> None:
    """Keep local convenience defaults out of strict API JSON schemas."""
    def visit(value):
        if isinstance(value, dict):
            value.pop("default", None)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(schema)


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    description: str = Field(min_length=1, max_length=2000)
    tool: Literal["web_search", "fetch_page", "create_file", "read_file", "reason"]


class ArtifactExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, json_schema_extra=_strip_schema_defaults)
    path: str = Field(min_length=1, max_length=240)
    required_sections: list[str] = Field(default_factory=list, max_length=30)
    min_sources: int = Field(default=0, ge=0, le=20)


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra=_strip_schema_defaults)
    steps: list[Step] = Field(min_length=1, max_length=20)
    supported: bool = True
    unsupported_reason: str = Field(default="", max_length=2000)
    expected_artifacts: list[ArtifactExpectation] = Field(default_factory=list, max_length=20)
    minimum_sources: int = Field(default=0, ge=0, le=20)


class StepCompletion(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    status: Literal["completed", "failed"]
    summary: str = Field(min_length=1, max_length=2000)


@dataclass(frozen=True)
class Limits:
    max_steps: int = 8
    max_turns_per_step: int = 6
    max_tool_calls: int = 24
    max_tokens: int = 100000
    max_seconds: float = 600
    max_replans: int = 1
    max_cost_usd: float | None = None

    def __post_init__(self) -> None:
        for name, ceiling in (("max_steps", 20), ("max_turns_per_step", 30),
                              ("max_tool_calls", 100), ("max_tokens", 10000000)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be between 1 and {ceiling}.")
        if type(self.max_replans) is not int or not 0 <= self.max_replans <= 2:
            raise ValueError("max_replans must be between 0 and 2.")
        if not math.isfinite(self.max_seconds) or not 0 < self.max_seconds <= 86400:
            raise ValueError("max_seconds must be between 0 and 86400.")
        if self.max_cost_usd is not None and (
            not math.isfinite(self.max_cost_usd) or self.max_cost_usd <= 0
        ):
            raise ValueError("max_cost_usd must be positive and finite.")


@dataclass(frozen=True)
class Pricing:
    input_per_million: float | None = None
    output_per_million: float | None = None

    def __post_init__(self) -> None:
        values = (self.input_per_million, self.output_per_million)
        if (values[0] is None) != (values[1] is None):
            raise ValueError("Set both input and output token prices, or neither.")
        if any(v is not None and (not math.isfinite(v) or v < 0) for v in values):
            raise ValueError("Token prices must be finite, nonnegative numbers.")

    def estimate(self, input_tokens: int, output_tokens: int) -> float | None:
        if self.input_per_million is None:
            return None
        return (input_tokens * self.input_per_million
                + output_tokens * self.output_per_million) / 1_000_000


class AgentError(RuntimeError):
    """An expected failure with a message safe to print and log."""


class BudgetExceeded(AgentError):
    pass


class Cancelled(AgentError):
    pass
