# response model for one predicted lndmark
# coords are in original uploaded-image pixels (not crop space) 
# this lets frontend draw directly over the image without worrying about letterBoxing
# uncertainty is just the 2D covariance ellipse params in those same pixels

from pydantic import BaseModel


class KeypointPrediction(BaseModel):
    index: int
    name: str  # shrt code like "caudal_center"
    label: str  # human readable, e.g. "Caudal Fin Center"
    x: float
    y: float
    sigma_x: float  # sqrt(Sigma[0, 0]) in orig pixels
    sigma_y: float  # sqrt(Sigma[1, 1]) in orig pixels
    rho: float  # correlaton coefficient in [-1, 1]
    visibility: float  # sigmoid of visibilty head [0, 1]
    low_visibility: bool