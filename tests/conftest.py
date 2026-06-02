"""Fixtures pytest partagées pour tous les tests."""

import numpy as np
import pytest


@pytest.fixture
def sample_frame():
    """Crée un frame vidéo de test."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_detection():
    """Crée une détection de test."""
    from src.detection.player_detector import Detection
    
    return Detection(
        x1=100,
        y1=150,
        x2=200,
        y2=350,
        confidence=0.9,
        class_id=0,
        class_name="player",
    )


@pytest.fixture
def sample_tracked_object(sample_detection):
    """Crée un objet suivi."""
    from src.detection.player_detector import TrackedObject
    
    obj = TrackedObject(track_id=1, detection=sample_detection)
    obj.team = "home"
    return obj


@pytest.fixture
def sample_frame_result(sample_tracked_object, sample_frame):
    """Crée un FrameResult."""
    from src.detection.player_detector import FrameResult
    
    return FrameResult(
        frame_idx=0,
        timestamp_s=0.0,
        tracked_objects=[sample_tracked_object],
        raw_frame=sample_frame,
    )


@pytest.fixture
def config():
    """Charge la configuration."""
    import yaml
    from pathlib import Path
    
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Crée un répertoire temporaire pour les résultats."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir
