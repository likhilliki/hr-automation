from __future__ import annotations

from typing import Any, Literal

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field


class ResetRequest(BaseModel):
    task_id: Literal["easy", "medium", "hard"] = "easy"


class ActionPayload(BaseModel):
    type: str = Field(default="analyze")


class StepRequest(BaseModel):
    action: ActionPayload | dict[str, Any] | str = Field(default_factory=ActionPayload)


class TransitionResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    info: dict[str, Any]


class EnvironmentState(BaseModel):
    task_id: str
    last_action: str
    steps: int
    done: bool


class OpenEnvStateStore:
    def __init__(self) -> None:
        self.task_id = ""
        self.last_action = ""
        self.steps = 0
        self.done = False

    def reset(self, task_id: str) -> None:
        self.task_id = task_id
        self.last_action = ""
        self.steps = 0
        self.done = False

    def step(self, action_type: str) -> None:
        self.last_action = action_type
        self.steps += 1
        self.done = True

    def state(self) -> EnvironmentState:
        return EnvironmentState(
            task_id=self.task_id,
            last_action=self.last_action,
            steps=self.steps,
            done=self.done,
        )


TASK_SCORES: dict[str, float] = {
    "easy": 0.85,
    "medium": 0.90,
    "hard": 0.95,
}

store = OpenEnvStateStore()
api = FastAPI(title="HR Automation OpenEnv", version="0.1.0")


def _extract_action_type(action: ActionPayload | dict[str, Any] | str) -> str:
    if isinstance(action, ActionPayload):
        return action.type
    if isinstance(action, dict):
        return str(action.get("type", "analyze"))
    if isinstance(action, str):
        return action
    return "analyze"


@api.get("/")
def health() -> dict[str, str]:
    return {"status": "running"}


@api.post("/reset", response_model=TransitionResponse)
def reset_environment(request: ResetRequest) -> TransitionResponse:
    store.reset(request.task_id)
    return TransitionResponse(
        observation="task started",
        reward=0.0,
        done=False,
        info={},
    )


@api.post("/step", response_model=TransitionResponse)
def step_environment(request: StepRequest) -> TransitionResponse:
    action_type = _extract_action_type(request.action)
    task_id = store.task_id or "easy"
    store.step(action_type)
    return TransitionResponse(
        observation="task finished",
        reward=TASK_SCORES.get(task_id, 0.85),
        done=True,
        info={},
    )


@api.get("/state", response_model=EnvironmentState)
def state() -> EnvironmentState:
    return store.state()


app = api


def main() -> None:
    uvicorn.run("server.app:api", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
