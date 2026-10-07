#FastAPI application entrypoint

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import analyze, compare, export, reports
from app.core.config import settings
from app.perception.loader import ModelNotTrainedError

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(ModelNotTrainedError)
async def model_not_trained_handler(_: Request, exc: ModelNotTrainedError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


app.include_router(analyze.router)
app.include_router(compare.router)
app.include_router(reports.router)
app.include_router(export.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
