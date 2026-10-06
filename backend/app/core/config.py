"""Application settings, loaded from environment variables (see `.env`)."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "bettafish-backend"
    model_checkpoint_path: str = "training/runs/v3_run/checkpoints/best_nll.pt"
    mfld_checkpoint_path: str = "training/runs/mfld_best.pt"  # only used by the comparison tab; the tab is disabled if the file is absent
    use_flip_tta: bool = False  # flip-TTA doubles inference time for ~1-2% lower error with v3; calibration.json is fitted without it
    device: str = "cpu"  # set to "cuda" on the training/inference GPU box
    cors_origins: list[str] = ["http://localhost:5173"]  # bettafish-frontend (Vite dev server)

    class Config:
        env_file = ".env"


settings = Settings()
