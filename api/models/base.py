"""Re-exports Base so all models share the same metadata registry."""
from api.db import Base

__all__ = ["Base"]
