# response models for the "compare with MFLD-Net" tab
# all coordinates are in ORIGINAL uploaded-image pixels so the frontend can draw them straight over the photo

from pydantic import BaseModel


class ComparedKeypoint(BaseModel):
    index: int
    name: str
    label: str
    group: str
    ours_x: float
    ours_y: float
    mfld_x: float
    mfld_y: float
    distance_px: float  # how far apart the two models place this keypoint
    distance_pct_body: float  # same distance as % of the fish's body length (our snout-to-peduncle estimate)


class ModelInfo(BaseModel):
    name: str
    params_millions: float
    note: str


class CompareSummary(BaseModel):
    mean_distance_pct_body: float
    median_distance_pct_body: float
    n_disagree: int  # keypoints placed more than `disagree_threshold_pct` of body length apart
    disagree_threshold_pct: float
    most_different: list[str]  # labels of the (up to 3) most different keypoints


class ComparisonResponse(BaseModel):
    image_width: int
    image_height: int
    body_length_px: float
    keypoints: list[ComparedKeypoint]
    summary: CompareSummary
    ours: ModelInfo
    mfld: ModelInfo
    notes: list[str]
