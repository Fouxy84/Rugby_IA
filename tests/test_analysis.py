"""Tests unitaires — Détection d'événements et heatmap."""

import numpy as np
import pytest


def test_game_event_to_dict():
    from src.analysis.event_detector import GameEvent

    ev = GameEvent(
        event_type="try",
        timestamp_s=45.5,
        frame_idx=1137,
        confidence=0.92,
        severity="critical",
        team="home",
        description="Essai domicile",
    )
    d = ev.to_dict()
    assert d["event_type"] == "try"
    assert d["confidence"] == pytest.approx(0.92, abs=0.01)
    assert d["severity"] == "critical"


def test_event_detector_cooldown():
    import time
    from src.analysis.event_detector import EventDetector

    det = EventDetector()
    det._last_event_time["scrum"] = time.monotonic()
    assert not det._can_emit("scrum")
    det._last_event_time["scrum"] = time.monotonic() - 10
    assert det._can_emit("scrum")


def test_heatmap_generator_accumulation():
    from src.detection.player_detector import Detection, TrackedObject, FrameResult
    from src.analysis.heatmap_generator import HeatmapGenerator

    gen = HeatmapGenerator(resolution=(105, 68))
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    player = TrackedObject(
        track_id=1,
        detection=Detection(640, 360, 660, 400, 0.9, 0, "player"),
        team="home",
    )
    fr = FrameResult(frame_idx=0, timestamp_s=0.0, tracked_objects=[player], raw_frame=frame)
    gen.update(fr)

    assert gen._grid_home.sum() > 0
    assert gen._grid_global.sum() > 0


def test_pattern_result_to_dict():
    from src.analysis.pattern_recognizer import PatternResult

    p = PatternResult(
        pattern="linebreak",
        confidence=0.79,
        frame_idx=500,
        timestamp_s=20.0,
        involved_players=[3, 7],
        description="Franchissement détecté",
    )
    d = p.to_dict()
    assert d["pattern"] == "linebreak"
    assert 3 in d["involved_players"]
