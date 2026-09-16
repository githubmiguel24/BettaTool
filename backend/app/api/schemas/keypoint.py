"""Response model for one predicted landmark.

Coordinates are in ORIGINAL uploaded-image pixels (not crop space), so the
frontend can draw them directly over the image the user selected without
knowing anything about letterboxing. The uncertainty is reported as the
parameters of the 2D covariance ellipse in those same pixels.
"""

from pydantic import BaseModel


class KeypointPrediction(BaseModel):
    index: int
    name: str  # short code, e.g. "caudal_center"
    label: str  # human-readable, e.g. "Caudal Fin Center"
    x: float
    y: float
    sigma_x: float  # sqrt(Sigma[0, 0]), original-image pixels
    sigma_y: float  # sqrt(Sigma[1, 1]), original-image pixels
    rho: float  # correlation coefficient in [-1, 1]
    visibility: float  # sigmoid of the visibility head, in [0, 1]
    low_visibility: bool
