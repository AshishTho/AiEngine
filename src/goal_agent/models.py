"""Validated plans, completion messages, and execution budgets."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    description: str = Field(min_length=1, max_length=2000)
    tool: Literal["web_search", "create_file", "reason"]


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: list[Step] = Field(min_length=1, max_length=20)


class StepCompletion(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    status: Literal["completed", "failed"]
    summary: str = Field(min_length=1, max_length=2000)


@dataclass(frozen=True)
class Limits:
    max_steps: int = 8
    max_turns_per_step: int = 6
    max_tool_calls: int = 24

    def __post_init__(self) -> None:
        for name, ceiling in (("max_steps", 20), ("max_turns_per_step", 30),
                              ("max_tool_calls", 100)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be between 1 and {ceiling}.")


class AgentError(RuntimeError):
    """An expected failure with a message safe to print and log."""
