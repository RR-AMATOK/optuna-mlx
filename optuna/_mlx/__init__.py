# ============================================================================
# Optuna-MLX: Backend detection and utilities for MLX GPU acceleration.
# "One does not simply import without checking availability." - Sheldon
# ============================================================================
from __future__ import annotations

import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import mlx.core as mx
else:
    try:
        import mlx.core as mx

        HAS_MLX = True
    except ImportError:
        HAS_MLX = False
        mx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def is_mlx_available() -> bool:
    """Check if MLX is available."""
    return HAS_MLX


def get_default_device() -> str:
    """Return the MLX default device name, or 'cpu' if MLX unavailable."""
    if HAS_MLX:
        return str(mx.default_device())
    return "cpu"
