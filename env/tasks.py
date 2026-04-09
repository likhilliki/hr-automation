from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from pydantic import BaseModel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

class Job(BaseModel):
    id: str
    title: str
    required_skills: list[str]
    experience_required: int
    description: str

class Candidate(BaseModel):
    id: str
    name: str
    skills: list[str]
    experience: int
    education: str
    cv_text: str

class TaskDefinition(BaseModel):
    id: str
    name: str
    description: str
    job_id: str
    candidate_ids: list[str]


def _load_json(filename: str) -> Any:
    path = DATA_DIR / filename
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jobs() -> list[Job]:
    return [Job(**item) for item in _load_json("jobs.json")]


def load_candidates() -> list[Candidate]:
    return [Candidate(**item) for item in _load_json("candidates.json")]


def load_tasks() -> list[TaskDefinition]:
    return [TaskDefinition(**item) for item in _load_json("tasks.json")]


JOBS = {job.id: job for job in load_jobs()}
CANDIDATES = {candidate.id: candidate for candidate in load_candidates()}
TASKS = {task.id: task for task in load_tasks()}


def get_job(job_id: str) -> Job:
    if job_id not in JOBS:
        raise KeyError(f"Job not found: {job_id}")
    return JOBS[job_id]


def get_candidate(candidate_id: str) -> Candidate:
    if candidate_id not in CANDIDATES:
        raise KeyError(f"Candidate not found: {candidate_id}")
    return CANDIDATES[candidate_id]


def get_candidates(candidate_ids: list[str]) -> list[Candidate]:
    return [get_candidate(candidate_id) for candidate_id in candidate_ids]


def get_task(task_id: str) -> TaskDefinition:
    if task_id not in TASKS:
        raise KeyError(f"Task not found: {task_id}")
    return TASKS[task_id]
