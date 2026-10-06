"""Structured logging + request correlation.

One JSON object per line, so the log platform (Datadog / ELK / CloudWatch / Azure Monitor
/ Grafana Loki) can filter by field: `request_id:abc123`, `status>=500`, `latency_ms>2000`.
"""

from __future__ import annotations

import contextvars
import json
import logging
import time
import uuid
from datetime import UTC, datetime

from fastapi import Request

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
log = logging.getLogger("adlens.http")

_RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        out.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()  # stdout/stderr; the platform collects it
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    logging.getLogger("uvicorn.access").disabled = True  # we log requests ourselves
    logging.getLogger("httpx").setLevel(logging.WARNING)


async def request_context_middleware(request: Request, call_next):
    # Reuse the caller's id if present so one id follows the request across services.
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    token = request_id_var.set(rid)
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = rid
        return response
    finally:
        log.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "latency_ms": round((time.perf_counter() - t0) * 1000),
            },
        )
        request_id_var.reset(token)
