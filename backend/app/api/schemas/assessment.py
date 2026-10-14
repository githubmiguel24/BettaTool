from pydantic import BaseModel


class FinalAssessment(BaseModel):
    quality: str
    confidence: float
    criteria_total: int
    criteria_passed: int
    criteria_faulted: int
    criteria_deferred: int
    summary: str
