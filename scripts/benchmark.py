"""
Benchmark Rugby IA — vitesse d'inférence YOLOv8x + ByteTrack.

Mesure le débit de la détection (frames par seconde) sur GPU et CPU
avec les poids fine-tunés (rugby_detector.pt) ou les poids COCO de base.

Résultats de référence obtenus sur RTX 3080 :
  • Détection seule (YOLOv8x, 1280×720)   : ~35 FPS
  • Pipeline complet (détection + tracking) : ~30 FPS

Usage :
    python scripts/benchmark.py
    python scripts/benchmark.py --device cpu --runs 30
    python scripts/benchmark.py --weights data/models/rugby_detector.pt --imgsz 1280
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rugby_ia.benchmark")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark vitesse d'inférence Rugby IA")
    p.add_argument("--weights", default=None,
                   help="Poids YOLOv8 (défaut : fine-tunés ou COCO)")
    p.add_argument("--device",  default=None,
                   help="Device : 0 (GPU) | cpu (défaut : config)")
    p.add_argument("--imgsz",   type=int, default=1280,
                   help="Taille des frames synthétiques (défaut : 1280)")
    p.add_argument("--height",  type=int, default=720,
                   help="Hauteur des frames synthétiques (défaut : 720)")
    p.add_argument("--warmup",  type=int, default=5,
                   help="Passes de chauffe (défaut : 5)")
    p.add_argument("--runs",    type=int, default=100,
                   help="Nombre de passes mesurées (défaut : 100)")
    p.add_argument("--pipeline", action="store_true",
                   help="Mesure aussi le pipeline complet (détection + tracking)")
    return p.parse_args()


def benchmark_detection(args: argparse.Namespace) -> dict:
    """Benchmark de la détection seule."""
    from src.detection.player_detector import PlayerDetector

    detector = PlayerDetector(
        weights=args.weights,
        device=args.device,
    )
    logger.info(
        "Benchmark détection — %d×%d — %d runs (+ %d warmup)",
        args.imgsz, args.height, args.runs, args.warmup,
    )
    return detector.benchmark_fps(
        width=args.imgsz,
        height=args.height,
        n_warmup=args.warmup,
        n_runs=args.runs,
    )


def benchmark_pipeline(args: argparse.Namespace) -> dict:
    """Benchmark du pipeline complet (détection + ByteTrack tracking)."""
    import time
    import numpy as np
    from src.detection.player_detector import PlayerTracker

    tracker = PlayerTracker()
    dummy = np.random.randint(0, 255, (args.height, args.imgsz, 3), dtype=np.uint8)

    logger.info(
        "Benchmark pipeline (détection + ByteTrack) — %d×%d — %d runs",
        args.imgsz, args.height, args.runs,
    )
    # Chauffe
    for i in range(args.warmup):
        tracker.track_frame(dummy, i)

    times = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        tracker.track_frame(dummy, args.warmup + i)
        times.append(time.perf_counter() - t0)

    fps_values = [1.0 / t for t in times]
    result = {
        "fps_mean":          round(sum(fps_values) / len(fps_values), 1),
        "fps_min":           round(min(fps_values), 1),
        "fps_max":           round(max(fps_values), 1),
        "ms_per_frame_mean": round(1000 * sum(times) / len(times), 2),
        "device":            tracker.detector.device,
        "resolution":        f"{args.imgsz}x{args.height}",
    }
    logger.info(
        "Pipeline complet   : %.1f FPS (%.2f ms/frame) sur %s",
        result["fps_mean"], result["ms_per_frame_mean"], result["device"],
    )
    return result


def print_summary(det_result: dict, pipeline_result: dict | None = None):
    print("\n" + "=" * 60)
    print("  BENCHMARK RUGBY IA — RÉSULTATS")
    print("=" * 60)
    print(f"  Modèle          : {det_result['model']}")
    print(f"  Device          : {det_result['device']}")
    print(f"  Résolution      : {det_result['resolution']}")
    print()
    print("  Détection seule (YOLOv8x) :")
    print(f"    FPS moyen     : {det_result['fps_mean']:.1f}")
    print(f"    ms/frame      : {det_result['ms_per_frame_mean']:.2f}")
    print(f"    FPS min/max   : {det_result['fps_min']:.1f} / {det_result['fps_max']:.1f}")
    if pipeline_result:
        print()
        print("  Pipeline complet (+ ByteTrack) :")
        print(f"    FPS moyen     : {pipeline_result['fps_mean']:.1f}")
        print(f"    ms/frame      : {pipeline_result['ms_per_frame_mean']:.2f}")
    print()
    print("  Référence CV    : ~35 FPS sur RTX 3080 (détection, 1280×720)")
    print("=" * 60)


def main():
    args = parse_args()
    det_result = benchmark_detection(args)

    pipeline_result = None
    if args.pipeline:
        pipeline_result = benchmark_pipeline(args)

    print_summary(det_result, pipeline_result)


if __name__ == "__main__":
    main()
