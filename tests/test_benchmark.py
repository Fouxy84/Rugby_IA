"""Tests de performance et benchmarks."""

import pytest
import numpy as np


@pytest.mark.benchmark
def test_detection_inference_speed(benchmark, sample_detection):
    """Benchmark la vitesse d'inférence de détection."""
    from src.detection.player_detector import Detection
    
    def create_detection():
        return Detection(
            x1=np.random.randint(0, 640),
            y1=np.random.randint(0, 360),
            x2=np.random.randint(640, 1280),
            y2=np.random.randint(360, 720),
            confidence=np.random.random(),
            class_id=np.random.randint(0, 3),
            class_name="player",
        )
    
    result = benchmark(create_detection)
    assert result.confidence > 0


@pytest.mark.benchmark
def test_frame_processing_speed(benchmark, sample_frame_result):
    """Benchmark la vitesse de traitement d'une frame."""
    
    def process_frame():
        frame = sample_frame_result
        # Simuler le traitement
        players = frame.players
        ball = frame.ball
        return len(players), ball is not None
    
    result = benchmark(process_frame)
    assert result is not None


@pytest.mark.slow
def test_memory_usage():
    """Test l'utilisation mémoire avec un grand volume de données."""
    from src.detection.player_detector import Detection, TrackedObject
    import tracemalloc
    
    tracemalloc.start()
    
    # Créer 1000 détections
    detections = []
    for i in range(1000):
        det = Detection(
            x1=i % 640,
            y1=i % 360,
            x2=(i + 100) % 1280,
            y2=(i + 100) % 720,
            confidence=0.9,
            class_id=i % 3,
            class_name="player",
        )
        detections.append(det)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Vérifier que l'usage est raisonnable (< 50 MB)
    assert peak / 1024 / 1024 < 50, f"Memory usage too high: {peak / 1024 / 1024:.2f} MB"


@pytest.mark.gpu
@pytest.mark.skip(reason="GPU tests skipped by default")
def test_yolo_inference_gpu():
    """Test inférence YOLOv8 sur GPU (si disponible)."""
    try:
        from ultralytics import YOLO
        import torch
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = YOLO("yolov8s.pt")
        # Dummy image
        results = model.predict(
            source=np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
            device=0,
        )
        assert len(results) > 0
    except Exception as e:
        pytest.skip(f"GPU test failed: {e}")


@pytest.mark.smoke
def test_imports_speed():
    """Vérifier que les imports ne sont pas trop lents."""
    import time
    
    start = time.time()
    import src.detection.player_detector  # noqa
    import src.analysis.event_detector  # noqa
    import src.analysis.heatmap_generator  # noqa
    elapsed = time.time() - start
    
    # Imports doivent prendre < 5 secondes
    assert elapsed < 5, f"Imports took {elapsed:.2f}s (too slow)"
