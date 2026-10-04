"""Tests unitaires — Détection et tracking."""

import numpy as np
import pytest
import sys
from types import SimpleNamespace
from src.detection.player_detector import Detection, TrackedObject, FrameResult
from src.detection.player_detector import PlayerDetector

def test_detection_dataclass():
    d = Detection(
        x1=10, y1=20, x2=50, y2=80,
        confidence=0.9, class_id=0, class_name="player",
    )
    assert d.center == (30.0, 50.0)
    assert d.area == pytest.approx(40 * 60)


def test_tracked_object_history():
    det = Detection(x1=0, y1=0, x2=20, y2=40, confidence=0.8, class_id=0, class_name="player")
    obj = TrackedObject(track_id=1, detection=det)
    for i in range(5):
        obj.update_history(i)
    assert len(obj.position_history) == 5
    assert obj.position_history[0][0] == 0


def test_frame_result_players_filter():
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


def test_player_detector_runs_with_pytorch_backend():
    detector = PlayerDetector(weights=None, device="cpu", pretrained=False)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detections = detector.detect(frame)

    assert isinstance(detections, list)
    assert all(hasattr(d, "class_name") for d in detections)


def test_player_detector_converts_yolo_coco_results(monkeypatch):
    class FakeBox:
        cls = SimpleNamespace(item=lambda: 0)
        conf = SimpleNamespace(item=lambda: 0.9)
        xyxy = [SimpleNamespace(tolist=lambda: [10, 20, 50, 80])]

    class FakeYOLO:
        def __init__(self, model):
            assert model == "yolov8n.pt"

        def predict(self, **kwargs):
            assert kwargs["source"].shape == (40, 60, 3)
            return [SimpleNamespace(boxes=[FakeBox()])]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setattr(
        "src.detection.player_detector.load_config",
        lambda: {"detection": {"confidence_threshold": 0.4, "iou_threshold": 0.5, "device": "cpu"}},
    )

    detector = PlayerDetector(weights=None, device="cpu", backend="yolo")
    detections = detector.detect(np.zeros((40, 60, 3), dtype=np.uint8))

    assert detections == [Detection(10, 20, 50, 80, 0.9, 0, "player")]
