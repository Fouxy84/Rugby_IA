"""Package détection - YOLOv8 + ByteTrack."""
from .player_detector import (
    Detection,
    TrackedObject,
    FrameResult,
    PlayerDetector,
    PlayerTracker,
    TeamClassifier,
)

__all__ = [
    "Detection",
    "TrackedObject",
    "FrameResult",
    "PlayerDetector",
    "PlayerTracker",
    "TeamClassifier",
]
