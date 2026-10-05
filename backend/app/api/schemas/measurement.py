#Response model for a single IBC criterion result.

from pydantic import BaseModel


class MeasurementResult(BaseModel):
    criterion_key: str
    label: str
    value: float
    uncertainty: float
    tsi: float
    rmse: float
    decision: str  # "Confident Pass" | "Confident Fault" | "Defer to Judge"
    # caudal spread angle faults are qualified: "Confident Slight Fault" | "Confident Major Fault" | "Confident Disqualify"
    landmark_indices: list[int] = []  # keypoints this criterion reads, for UI highlighting
