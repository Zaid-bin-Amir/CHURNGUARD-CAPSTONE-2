import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request

from app.logging_utils import PredictionLog, configure_logging
from app.schemas import (
    BatchRequest,
    BatchResponse,
    CustomerInput,
    HealthResponse,
    PredictionResponse,
)
from app.security import require_api_key
from app.service import PredictionService
from app.settings import load_settings

DESCRIPTION = (
    "ChurnGuard scores a telecom customer with a calibrated churn probability and explains the "
    "three biggest drivers in plain English. Send the API key in the X-API-Key header."
)


def create_app():
    settings = load_settings()
    configure_logging()
    api_logger = logging.getLogger("churnguard.api")
    prediction_log = PredictionLog(settings.log_dir)

    @asynccontextmanager
    async def lifespan(application):
        application.state.service = PredictionService.load()
        api_logger.info(
            "model loaded",
            extra={"event": {"model_version": application.state.service.version}},
        )
        yield

    application = FastAPI(title="ChurnGuard API", version="1.0.0", description=DESCRIPTION, lifespan=lifespan)
    application.state.settings = settings
    application.state.prediction_log = prediction_log

    @application.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        api_logger.info(
            "request",
            extra={
                "event": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": latency_ms,
                }
            },
        )
        return response

    @application.get("/")
    def root():
        return {"service": "ChurnGuard", "docs": "/docs", "health": "/health"}

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request):
        service = request.app.state.service
        return {"status": "ok", "model_version": service.version, "threshold": service.threshold}

    @application.post("/predict", response_model=PredictionResponse, dependencies=[Depends(require_api_key)])
    def predict(customer: CustomerInput, request: Request):
        service = request.app.state.service
        started = time.perf_counter()
        result = service.predict_one(customer.model_dump())
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        request.app.state.prediction_log.write(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                "request_id": getattr(request.state, "request_id", None),
                "endpoint": "/predict",
                "model_version": result["model_version"],
                "inputs": customer.model_dump(),
                "probability": result["probability"],
                "label": result["label"],
                "risk_level": result["risk_level"],
                "top_drivers": [driver["explanation"] for driver in result["top_drivers"]],
                "latency_ms": latency_ms,
            }
        )
        return result

    @application.post("/predict/batch", response_model=BatchResponse, dependencies=[Depends(require_api_key)])
    def predict_batch(batch: BatchRequest, request: Request):
        service = request.app.state.service
        started = time.perf_counter()
        records = [customer.model_dump() for customer in batch.customers]
        predictions = service.predict_many(records)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        for record, prediction in zip(records, predictions):
            request.app.state.prediction_log.write(
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                    "request_id": getattr(request.state, "request_id", None),
                    "endpoint": "/predict/batch",
                    "model_version": service.version,
                    "inputs": record,
                    "probability": prediction["probability"],
                    "label": prediction["label"],
                    "risk_level": prediction["risk_level"],
                    "latency_ms": latency_ms,
                }
            )
        return {"predictions": predictions, "threshold": round(service.threshold, 4), "model_version": service.version}

    return application


app = create_app()
