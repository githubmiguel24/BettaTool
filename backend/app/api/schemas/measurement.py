#Response model for a single IBC criterion result.

from pydantic import BaseModel


class MeasurementResult(BaseModel):
    criterion_key: str
    label: str
    value: float
    uncertainty: float  # reported margin of error U(y)
    tsi: float  # threshold sensitivity index, px
    rmse: float  # calibrated predicted keypoint uncertainty sigma_hat, px; Confident iff rmse < tsi
    decision: str  # "Confident Pass" | "Confident Fault" | "Defer to Judge"
    # caudal spread angle faults are qualified: "Confident Slight Fault" | "Confident Major Fault" | "Confident Disqualify"
    landmark_indices: list[int] = []  # keypoints this criterion reads, for UI highlighting
