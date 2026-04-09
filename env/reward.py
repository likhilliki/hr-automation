from typing import Iterable
from .tasks import Candidate, Job


def normalize_skills(skills: Iterable[str]) -> set[str]:
    return {skill.strip().lower() for skill in skills if isinstance(skill, str)}


def expected_score(candidate: Candidate, job: Job) -> float:
    required = normalize_skills(job.required_skills)
    matched = normalize_skills(candidate.skills) & required
    skill_ratio = len(matched) / max(1, len(required))
    experience_ratio = min(candidate.experience / max(1, job.experience_required), 2.0)
    base = 2.0 + 6.0 * skill_ratio + min(2.0, experience_ratio)
    return round(max(0.0, min(10.0, base)), 2)


def should_shortlist(candidate: Candidate, job: Job) -> bool:
    required = normalize_skills(job.required_skills)
    matched = normalize_skills(candidate.skills) & required
    skill_ratio = len(matched) / max(1, len(required))
    return skill_ratio >= 0.6 and candidate.experience >= job.experience_required


def skill_accuracy(extracted: list[str], actual: list[str]) -> float:
    extracted_set = normalize_skills(extracted)
    actual_set = normalize_skills(actual)
    if not actual_set:
        return 0.0
    matched = extracted_set & actual_set
    return round(len(matched) / len(actual_set), 3)


def score_match_reward(report_score: float, candidate: Candidate, job: Job) -> float:
    expected = expected_score(candidate, job)
    distance = abs(report_score - expected)
    if distance <= 1.0:
        return 0.3
    if distance <= 3.0:
        return 0.1
    return -0.1


def shortlist_reward(choice: bool, candidate: Candidate, job: Job) -> float:
    recommended = should_shortlist(candidate, job)
    if choice == recommended:
        return 0.5
    return -0.3


def schedule_reward(shortlisted: bool) -> float:
    return 1.0 if shortlisted else -0.2


def reject_reward(candidate: Candidate, job: Job) -> float:
    if should_shortlist(candidate, job):
        return -0.3
    return 0.2


def invalid_action_reward() -> float:
    return -0.2


def repeated_action_penalty() -> float:
    return -0.1
