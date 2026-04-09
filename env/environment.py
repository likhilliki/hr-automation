from __future__ import annotations
from typing import Any
from pydantic import ValidationError
from .models import Action, Observation, Reward, ALLOWED_ACTIONS
from .tasks import get_job, get_task, get_candidates, Candidate, Job
from .reward import (
    expected_score,
    should_shortlist,
    skill_accuracy,
    score_match_reward,
    shortlist_reward,
    schedule_reward,
    reject_reward,
    invalid_action_reward,
    repeated_action_penalty,
)


def _normalize_text(value: Any) -> str:
    return str(value).strip()


class HREnvironment:
    def __init__(self, task_id: str = "easy") -> None:
        self.task_id = task_id
        self.reset(task_id)

    def reset(self, task_id: str | None = None) -> Observation:
        if task_id is not None:
            self.task_id = task_id
        self.task = get_task(self.task_id)
        self.job = get_job(self.task.job_id)
        self.candidates = get_candidates(self.task.candidate_ids)
        self.current_candidate_index = 0
        self.step_count = 0
        self.done = False
        self.last_action_type = None
        self.action_history: list[str] = []
        self.decision_history: list[dict[str, Any]] = []
        self.interview_target_id: str | None = None
        self.shortlisted_ids: list[str] = []
        self.rejected_ids: list[str] = []
        self.extracted_skills: list[str] = []
        self.candidate_score: float | None = None
        self.shortlisted: bool = False
        self.interview_scheduled: bool = False
        self._reset_candidate_state()
        return self.observation()

    def state(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "job_id": self.job.id,
            "candidate_ids": [candidate.id for candidate in self.candidates],
            "current_candidate_index": self.current_candidate_index,
            "current_candidate_id": self.current_candidate.id,
            "extracted_skills": self.extracted_skills,
            "candidate_score": self.candidate_score,
            "shortlisted": self.shortlisted,
            "interview_scheduled": self.interview_scheduled,
            "action_history": self.action_history,
            "decision_history": self.decision_history,
            "done": self.done,
            "step_count": self.step_count,
            "interview_target_id": self.interview_target_id,
            "shortlisted_ids": self.shortlisted_ids,
            "rejected_ids": self.rejected_ids,
        }

    def observation(self) -> Observation:
        return Observation(
            job_description=self._job_description,
            candidate_profile=self._candidate_profile,
            extracted_skills=list(self.extracted_skills),
            candidate_score=self.candidate_score,
            shortlisted=self.shortlisted,
            interview_scheduled=self.interview_scheduled,
            action_history=list(self.action_history),
        )

    def step(self, action: dict | Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        if self.done:
            reward = Reward(value=0.0, reason="Episode already terminated")
            return self.observation(), reward, True, {"task_score": self._final_task_score()}

        self.step_count += 1
        action_payload = action if isinstance(action, dict) else action.dict()
        if isinstance(action_payload, dict) and "action_input" in action_payload:
            action_payload["action_input"] = self._normalize_action_input(action_payload["action_input"])

        try:
            action_obj = Action.parse_obj(action_payload)
        except ValidationError as error:
            return self.observation(), Reward(value=-0.2, reason="Invalid action payload"), False, {"error": str(error)}

        if action_obj.action_type not in ALLOWED_ACTIONS:
            reward = Reward(value=invalid_action_reward(), reason="Invalid action type")
            return self.observation(), reward, self.done, {}

        reward = self._apply_action(action_obj)
        self.last_action_type = action_obj.action_type
        self.action_history.append(f"{action_obj.action_type}:{action_obj.action_input}")

        if self.step_count >= 10 and not self.done:
            self.done = True
            reward = Reward(value=reward.value, reason=f"{reward.reason}; reached maximum steps")

        return self.observation(), reward, self.done, {"task_score": self._final_task_score() if self.done else None}

    def _apply_action(self, action: Action) -> Reward:
        if self.last_action_type == action.action_type:
            return Reward(value=repeated_action_penalty(), reason="Repeated action")

        if action.action_type == "extract_skills":
            extracted = self._parse_skill_input(action.action_input)
            if extracted is None:
                return Reward(value=invalid_action_reward(), reason="extract_skills input must list skills as comma-separated text")
            self.extracted_skills = extracted
            accuracy = skill_accuracy(self.extracted_skills, self.current_candidate.skills)
            return Reward(value=round(0.2 * accuracy, 3), reason="Extracted skills evaluated")

        if action.action_type == "score_candidate":
            score = self._parse_numeric_input(action.action_input)
            if score is None or score < 0 or score > 10:
                return Reward(value=invalid_action_reward(), reason="score_candidate input must be numeric between 0 and 10")
            self.candidate_score = float(score)
            reward_value = score_match_reward(self.candidate_score, self.current_candidate, self.job)
            return Reward(value=reward_value, reason="Candidate score evaluated")

        if action.action_type == "shortlist_candidate":
            choice = self._parse_yes_no(action.action_input)
            if choice is None:
                return Reward(value=invalid_action_reward(), reason="shortlist_candidate input must be yes/no")
            self.shortlisted = choice
            reward_value = shortlist_reward(choice, self.current_candidate, self.job)
            if choice:
                self.shortlisted_ids.append(self.current_candidate.id)
            else:
                self.rejected_ids.append(self.current_candidate.id)
            self.decision_history.append({"candidate_id": self.current_candidate.id, "shortlisted": choice})
            return self._advance_or_finish(Reward(value=reward_value, reason="Shortlist decision recorded"))

        if action.action_type == "schedule_interview":
            text = self._parse_text_input(action.action_input)
            if not text:
                return Reward(value=invalid_action_reward(), reason="schedule_interview input must be interview time")
            self.interview_scheduled = True
            self.interview_target_id = self.current_candidate.id
            self.done = True
            reward_value = schedule_reward(self.shortlisted)
            reason = "Interview scheduled" if self.shortlisted else "Interview scheduled without shortlist"
            return Reward(value=reward_value, reason=reason)

        if action.action_type == "reject_candidate":
            reason_text = self._parse_text_input(action.action_input)
            if not reason_text:
                return Reward(value=invalid_action_reward(), reason="reject_candidate input must include a reason")
            self.rejected_ids.append(self.current_candidate.id)
            self.shortlisted = False
            self.interview_scheduled = False
            self.decision_history.append({"candidate_id": self.current_candidate.id, "rejected_reason": reason_text})
            self.done = True
            reward_value = reject_reward(self.current_candidate, self.job)
            return Reward(value=reward_value, reason="Candidate rejected")

        if action.action_type == "request_more_information":
            question = self._parse_text_input(action.action_input)
            if not question:
                return Reward(value=invalid_action_reward(), reason="request_more_information input must be a question")
            return Reward(value=0.05, reason="Requested additional candidate information")

        return Reward(value=invalid_action_reward(), reason="Unhandled action")

    def _advance_or_finish(self, reward: Reward) -> Reward:
        if self.task.id == "hard" and self.current_candidate_index < len(self.candidates) - 1:
            self.current_candidate_index += 1
            self._reset_candidate_state()
            return reward
        if self.task.id == "hard" and self.current_candidate_index >= len(self.candidates) - 1:
            self.done = True
            return reward
        return reward

    def _reset_candidate_state(self) -> None:
        self.current_candidate = self.candidates[self.current_candidate_index]
        self.extracted_skills = []
        self.candidate_score = None
        self.shortlisted = False
        self.interview_scheduled = False
        self._job_description = f"{self.job.title}: {self.job.description} Required skills: {', '.join(self.job.required_skills)}. Experience required: {self.job.experience_required} years."
        self._candidate_profile = (
            f"{self.current_candidate.name} | experience: {self.current_candidate.experience} years | education: {self.current_candidate.education}. "
            f"CV summary: {self.current_candidate.cv_text}"
        )

    def _normalize_action_input(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            return normalized if normalized else None
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
            return ", ".join(items) if items else None
        normalized = str(value).strip()
        return normalized if normalized else None

    def _parse_skill_input(self, value: Any) -> list[str] | None:
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return [item.strip() for item in value if item.strip()]
        if isinstance(value, str):
            skills = [item.strip() for item in value.split(",") if item.strip()]
            return skills if skills else None
        return None

    def _parse_numeric_input(self, value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return None
        return None

    def _parse_yes_no(self, value: Any) -> bool | None:
        text = self._parse_text_input(value)
        if text is None:
            return None
        normalized = text.strip().lower()
        if normalized in {"yes", "y", "true", "t"}:
            return True
        if normalized in {"no", "n", "false", "f"}:
            return False
        return None

    def _parse_text_input(self, value: Any) -> str | None:
        text = _normalize_text(value)
        return text if text else None

    def _final_task_score(self) -> float:
        from .graders import grade_easy, grade_medium, grade_hard

        if self.task.id == "easy":
            return grade_easy(self.extracted_skills, self.current_candidate)
        if self.task.id == "medium":
            return grade_medium(self.candidate_score, self.shortlisted, self.current_candidate, self.job)
        return grade_hard(self.job, self.candidates, self.interview_target_id, self.shortlisted_ids, self.rejected_ids)
