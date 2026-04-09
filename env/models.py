from __future__ import annotations
from pydantic import BaseModel, Field, validator

ALLOWED_ACTIONS = {
    "extract_skills",
    "score_candidate",
    "shortlist_candidate",
    "schedule_interview",
    "reject_candidate",
    "request_more_information",
}

class Observation(BaseModel):
    job_description: str
    candidate_profile: str
    extracted_skills: list[str] = Field(default_factory=list)
    candidate_score: float | None = None
    shortlisted: bool = False
    interview_scheduled: bool = False
    action_history: list[str] = Field(default_factory=list)

class Action(BaseModel):
    action_type: str
    action_input: str | None = None

    @validator("action_type")
    def validate_action_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ALLOWED_ACTIONS:
            raise ValueError(f"Invalid action_type: {value}")
        return normalized

class Reward(BaseModel):
    value: float
    reason: str
