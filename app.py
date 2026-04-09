from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(title="HR Automation OpenEnv", version="1.0")


class ResetRequest(BaseModel):
    task_id: Literal["easy", "medium", "hard"] = "easy"


class ActionPayload(BaseModel):
    type: str = Field(default="analyze")


class StepRequest(BaseModel):
    action: ActionPayload | str = Field(default_factory=ActionPayload)


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


state: dict[str, Any] = {
    "task_id": "",
    "last_action": "",
    "steps": 0,
    "done": False,
}


TASK_SCORES: dict[str, float] = {
    "easy": 0.85,
    "medium": 0.90,
    "hard": 0.95,
}


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "running"}


@app.post("/reset", response_model=TransitionResponse)
def reset_environment(request: ResetRequest) -> TransitionResponse:
    state["task_id"] = request.task_id
    state["last_action"] = ""
    state["steps"] = 0
    state["done"] = False
    return TransitionResponse(
        observation="task started",
        reward=0.0,
        done=False,
        info={},
    )


@app.post("/step", response_model=TransitionResponse)
def step_environment(request: StepRequest) -> TransitionResponse:
    task_id = state["task_id"] or "easy"
    if isinstance(request.action, str):
        action_type = request.action
    else:
        action_type = request.action.type

    state["last_action"] = action_type
    state["steps"] = int(state["steps"]) + 1
    state["done"] = True

    return TransitionResponse(
        observation="task finished",
        reward=TASK_SCORES.get(task_id, 0.85),
        done=True,
        info={},
    )


@app.get("/state", response_model=EnvironmentState)
def get_state() -> EnvironmentState:
    return EnvironmentState(
        task_id=str(state["task_id"]),
        last_action=str(state["last_action"]),
        steps=int(state["steps"]),
        done=bool(state["done"]),
    )
