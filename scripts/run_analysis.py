#!/usr/bin/env python3
"""
Script principal — lance l'analyse d'un match en ligne de commande.

Usage :
    python scripts/run_analysis.py --video data/raw/match.mp4
    python scripts/run_analysis.py --url https://youtube.com/watch?v=XXX
    python scripts/run_analysis.py --video match.mp4 --no-heatmap
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ajoute le répertoire racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.video_downloader import VideoDownloader
from src.pipeline.realtime_pipeline import RealtimePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_analysis")


def parse_args():
    p = argparse.ArgumentParser(description="Rugby IA — Analyse de match")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Chemin vers le fichier vidéo local")
    src.add_argument("--url", help="URL YouTube / yt-dlp à télécharger et analyser")
    p.add_argument("--output-dir", default="data/outputs", help="Répertoire de sortie")
    p.add_argument("--no-heatmap", action="store_true", help="Désactive la génération de heatmaps")
    p.add_argument("--no-patterns", action="store_true", help="Désactive la reconnaissance de patterns")
    p.add_argument("--device", default=None, help="Device PyTorch (cuda / cpu)")
    p.add_argument("--summary-interval", type=int, default=100, help="Affiche un résumé toutes les N frames")
    return p.parse_args()


async def main():
    args = parse_args()

    # --- Résolution du fichier vidéo ---
    if args.url:
        logger.info("Téléchargement de : %s", args.url)
        dl = VideoDownloader()
        video_path = str(dl.download(args.url))
    else:
        video_path = args.video
        if not Path(video_path).exists():
            logger.error("Fichier introuvable : %s", video_path)
            sys.exit(1)

    logger.info("Analyse de : %s", video_path)

    # --- Pipeline ---
    pipeline = RealtimePipeline(
        device=args.device,
        enable_heatmap=not args.no_heatmap,
        enable_patterns=not args.no_patterns,
    )

    frame_count = 0
    async for snapshot in pipeline.stream(video_path):
        frame_count += 1
        if frame_count % args.summary_interval == 0:
            print(
                f"\r[Frame {snapshot.frame_idx:5d} | "
                f"{snapshot.timestamp_s:6.1f}s] "
                f"Phase: {snapshot.phase:<15} "
                f"Home: {snapshot.n_players_home:2d} "
                f"Away: {snapshot.n_players_away:2d} "
                f"FPS: {snapshot.processing_fps:5.1f}",
                end="",
                flush=True,
            )
            if snapshot.key_insights:
                print()
                for ins in snapshot.key_insights:
                    print(f"  → {ins}")

    print("\n\n=== ANALYSE TERMINÉE ===")
    print(f"Événements détectés : {len(pipeline.event_det.all_events)}")
    if pipeline.pattern_rec:
        print(f"Patterns détectés   : {len(pipeline.pattern_rec.all_patterns)}")

    # Sauvegarde des heatmaps finales
    if pipeline.heatmap_gen:
        out = Path(args.output_dir) / "heatmaps"
        out.mkdir(parents=True, exist_ok=True)
        for mode in ("global", "home", "away", "ball"):
            pipeline.heatmap_gen.generate(
                mode=mode,
                save_path=str(out / f"heatmap_{mode}.png"),
            )
        print(f"Heatmaps sauvegardées dans {out}")

    # Rapport JSON
    import json
    report = {
        "video": video_path,
        "total_frames": frame_count,
        "events": [e.to_dict() for e in pipeline.event_det.all_events],
        "patterns": [p.to_dict() for p in pipeline.pattern_rec.all_patterns] if pipeline.pattern_rec else [],
        "zone_stats": pipeline.heatmap_gen.zone_statistics() if pipeline.heatmap_gen else {},
    }
    report_path = Path(args.output_dir) / "analysis_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Rapport JSON : {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
