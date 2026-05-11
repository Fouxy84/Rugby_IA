"""Tests unitaires — Détection et tracking."""

import numpy as np
import pytest


def test_detection_dataclass():
    from src.detection.player_detector import Detection

    d = Detection(
        x1=10, y1=20, x2=50, y2=80,
        confidence=0.9, class_id=0, class_name="player",
    )
    assert d.center == (30.0, 50.0)
    assert d.area == pytest.approx(40 * 60)


def test_tracked_object_history():
    from src.detection.player_detector import Detection, TrackedObject

    det = Detection(x1=0, y1=0, x2=20, y2=40, confidence=0.8, class_id=0, class_name="player")
    obj = TrackedObject(track_id=1, detection=det)
    for i in range(5):
        obj.update_history(i)
    assert len(obj.position_history) == 5
    assert obj.position_history[0][0] == 0


def test_frame_result_players_filter():
    from src.detection.player_detector import Detection, TrackedObject, FrameResult

    players = [
        TrackedObject(track_id=i, detection=Detection(0, 0, 10, 10, 0.9, 0, "player"))
        for i in range(3)
    ]
    ball = TrackedObject(
        track_id=99,
        detection=Detection(50, 50, 60, 60, 0.8, 2, "ball"),
    )
    fr = FrameResult(
        frame_idx=0,
        timestamp_s=0.0,
        tracked_objects=players + [ball],
    )
    assert len(fr.players) == 3
    assert fr.ball is not None
    assert fr.ball.track_id == 99
