from __future__ import annotations

import logging
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from api.config import settings

log = logging.getLogger(__name__)

engine = create_async_engine(
    settings.database_url,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


# Set True after a successful init_db(). All DB-dependent code checks this before querying.
DB_AVAILABLE: bool = False


async def get_db() -> AsyncGenerator[Optional[AsyncSession], None]:
    if not DB_AVAILABLE:
        yield None
        return
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> bool:
    """Create all tables on startup. Returns True if DB is reachable."""
    global DB_AVAILABLE
    try:
        async with engine.begin() as conn:
            from api.models import base  # noqa: F401 — registers all models
            await conn.run_sync(Base.metadata.create_all)
        DB_AVAILABLE = True
        log.info("PostgreSQL connected — tables ready")
    except Exception as exc:
        DB_AVAILABLE = False
        log.warning("PostgreSQL unavailable — running in JSON-only mode: %s", exc)
    return DB_AVAILABLE
