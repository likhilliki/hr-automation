from typing import Optional
from .tasks import Candidate, Job
from .reward import expected_score, skill_accuracy, should_shortlist


def grade_easy(extracted_skills: list[str], candidate: Candidate) -> float:
    return skill_accuracy(extracted_skills, candidate.skills)


def grade_medium(candidate_score: Optional[float], shortlist_status: Optional[bool], candidate: Candidate, job: Job) -> float:
    if candidate_score is None or shortlist_status is None:
        return 0.0
    expected = expected_score(candidate, job)
    score_quality = max(0.0, 1.0 - abs(candidate_score - expected) / 10.0)
    shortlist_correct = 1.0 if shortlist_status == should_shortlist(candidate, job) else 0.0
    return round(0.5 * score_quality + 0.5 * shortlist_correct, 3)


def grade_hard(task_job: Job, candidate_pool: list[Candidate], interview_target_id: Optional[str], shortlisted_ids: list[str], rejected_ids: list[str]) -> float:
    if not candidate_pool:
        return 0.0
    best_candidate = max(candidate_pool, key=lambda c: expected_score(c, task_job))
    if interview_target_id:
        return 1.0 if interview_target_id == best_candidate.id else 0.0
    if best_candidate.id in shortlisted_ids and all(other_id in rejected_ids for other_id in [c.id for c in candidate_pool if c.id != best_candidate.id]):
        return 0.8
    if best_candidate.id in shortlisted_ids:
        return 0.6
    return 0.0
