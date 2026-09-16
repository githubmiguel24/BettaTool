"""Response model for a single IBC criterion result."""

from pydantic import BaseModel


class MeasurementResult(BaseModel):
    criterion_key: str
    label: str
    value: float
    uncertainty: float
    tsi: float
    rmse: float
    decision: str  # "Confident Pass" | "Confident Fault" | "Defer to Judge"
