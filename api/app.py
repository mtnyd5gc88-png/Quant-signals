from __future__ import annotations
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from api.db import init_db, DB_AVAILABLE
from api.routes import auth as auth_router, diagnostics, performance, portfolio, refresh, regime, research, search, settings as settings_router, signals, validate
from api.scheduler import start_scheduler, stop_scheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting Quant API")
    await init_db()
    if settings.scheduler_enabled:
        start_scheduler()
    yield
    stop_scheduler()
    log.info("Quant API shut down")


app = FastAPI(
    title="Quant Signals API",
    version="1.0.0",
    description="Institutional-grade quantitative signals and portfolio analytics",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_prefix = "/api"
app.include_router(auth_router.router,         prefix=_prefix)
app.include_router(signals.router,             prefix=_prefix)
app.include_router(portfolio.router,           prefix=_prefix)
app.include_router(performance.router,         prefix=_prefix)
app.include_router(regime.router,              prefix=_prefix)
app.include_router(diagnostics.router,         prefix=_prefix)
app.include_router(research.router,            prefix=_prefix)
app.include_router(settings_router.router,     prefix=_prefix)
app.include_router(refresh.router,             prefix=_prefix)
app.include_router(search.router,              prefix=_prefix)
app.include_router(validate.router,            prefix=_prefix)


@app.get("/api/health")
async def health() -> dict:
    from api.db import DB_AVAILABLE as _db
    return {"status": "ok", "db": "connected" if _db else "json-only"}
