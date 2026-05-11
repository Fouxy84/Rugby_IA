"""
Pipeline temps réel : orchestre la détection, le tracking,
la classification de phases, la détection d'événements et les patterns
frame par frame avec broadcasting WebSocket.

Modes :
  - "live"  : flux caméra ou RTSP
  - "file"  : fichier vidéo local (analyse batch avec reporting en temps réel)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import AsyncIterator, Optional
import cv2
import numpy as np
import yaml

from ..detection.player_detector import PlayerTracker, TeamClassifier
from ..analysis.phase_classifier import PhaseClassifier
from ..analysis.event_detector import EventDetector
from ..analysis.heatmap_generator import HeatmapGenerator
from ..analysis.pattern_recognizer import PatternRecognizer

logger = logging.getLogger("rugby_ia.pipeline")


def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Snapshot d'état temps réel (broadcast WebSocket)
# ---------------------------------------------------------------------------

@dataclass
class AnalysisSnapshot:
    """État complet de l'analyse à un instant donné."""
    frame_idx: int
    timestamp_s: float
    phase: str
    phase_confidence: float
    n_players_home: int
    n_players_away: int
    ball_position: Optional[tuple]
    recent_events: list[dict]
    recent_patterns: list[dict]
    zone_stats: dict
    key_insights: list[str]
    processing_fps: float

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "timestamp_s": round(self.timestamp_s, 2),
            "phase": self.phase,
            "phase_confidence": round(self.phase_confidence, 3),
            "n_players_home": self.n_players_home,
            "n_players_away": self.n_players_away,
            "ball_position": self.ball_position,
            "recent_events": self.recent_events,
            "recent_patterns": self.recent_patterns,
            "zone_stats": self.zone_stats,
            "key_insights": self.key_insights,
            "processing_fps": round(self.processing_fps, 1),
        }


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

class RealtimePipeline:
    """
    Orchestre tous les modules d'analyse pour produire un flux
    d'AnalysisSnapshot à haute fréquence.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        enable_heatmap: bool = True,
        enable_patterns: bool = True,
    ):
        cfg = load_config()
        self.fps_target: float = cfg["video"]["default_fps"]
        self.resize = (
            cfg["video"]["resize_width"],
            cfg["video"]["resize_height"],
        )

        logger.info("Initialisation du pipeline temps réel…")
        self.tracker        = PlayerTracker()
        self.team_clf       = TeamClassifier()
        self.phase_clf      = PhaseClassifier(device=device)
        self.event_det      = EventDetector()
        self.heatmap_gen    = HeatmapGenerator() if enable_heatmap else None
        self.pattern_rec    = PatternRecognizer() if enable_patterns else None

        self._team_calibrated = False
        self._frame_times: list[float] = []
        self._last_snapshot: Optional[AnalysisSnapshot] = None
        self._running = False

        # Fenêtre d'insights (10 dernières secondes)
        self._insight_window_s = 10.0

    # ------------------------------------------------------------------
    # Calibration équipes
    # ------------------------------------------------------------------

    def calibrate_teams(self, frame: np.ndarray):
        """Calibre les couleurs d'équipes sur la première frame."""
        detections = self.tracker.detector.detect(frame)
        self.team_clf.fit(frame, detections)
        self._team_calibrated = True
        logger.info("Calibration équipes effectuée.")

    # ------------------------------------------------------------------
    # Traitement d'une frame
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> AnalysisSnapshot:
        """
        Traite une frame et retourne un AnalysisSnapshot.
        """
        t0 = time.perf_counter()

        # Redimensionnement
        frame = cv2.resize(frame, self.resize)

        # --- Calibration auto sur la première frame ---
        if not self._team_calibrated and frame_idx == 0:
            self.calibrate_teams(frame)

        # --- Tracking joueurs ---
        frame_result = self.tracker.track_frame(frame, frame_idx)

        # --- Attribution des équipes ---
        if self._team_calibrated:
            for obj in frame_result.players:
                obj.team = self.team_clf.classify(frame, obj)

        # --- Classification de phase ---
        phase_result = self.phase_clf.update(frame, frame_idx)
        current_phase = phase_result.phase if phase_result else (
            self.phase_clf.current_phase or "jeu_courant"
        )
        phase_conf = phase_result.confidence if phase_result else 0.0

        # --- Détection d'événements ---
        new_events = []
        if phase_result:
            ev = self.event_det.on_phase_change(
                current_phase, phase_conf, frame_idx
            )
            if ev:
                new_events.append(ev)

        new_events += self.event_det.analyze_frame(
            frame_result, current_phase, phase_conf
        )

        # --- Heatmap (mise à jour incrémentale) ---
        if self.heatmap_gen:
            self.heatmap_gen.update(frame_result)

        # --- Patterns ---
        new_patterns = []
        if self.pattern_rec:
            new_patterns = self.pattern_rec.analyze(frame_result)

        # --- Statistiques ---
        home_players = [p for p in frame_result.players if p.team == "home"]
        away_players = [p for p in frame_result.players if p.team == "away"]
        ball = frame_result.ball

        zone_stats = self.heatmap_gen.zone_statistics() if self.heatmap_gen else {}

        # --- Key Insights ---
        insights = self._generate_insights(
            current_phase, phase_conf, frame_idx,
            len(home_players), len(away_players),
            new_events, new_patterns, zone_stats,
        )

        # --- FPS de traitement ---
        elapsed = time.perf_counter() - t0
        self._frame_times.append(elapsed)
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        proc_fps = 1.0 / (sum(self._frame_times) / len(self._frame_times))

        # Événements récents (10 dernières secondes)
        recent_ts = frame_idx / self.fps_target - self._insight_window_s
        recent_events = [
            e.to_dict() for e in self.event_det.events_since(max(0, recent_ts))
        ][-10:]

        recent_patterns = [
            p.to_dict() for p in (self.pattern_rec.all_patterns if self.pattern_rec else [])
            if p.timestamp_s >= max(0, recent_ts)
        ][-5:]

        snapshot = AnalysisSnapshot(
            frame_idx=frame_idx,
            timestamp_s=frame_idx / self.fps_target,
            phase=current_phase,
            phase_confidence=phase_conf,
            n_players_home=len(home_players),
            n_players_away=len(away_players),
            ball_position=ball.center if ball else None,
            recent_events=recent_events,
            recent_patterns=recent_patterns,
            zone_stats=zone_stats,
            key_insights=insights,
            processing_fps=proc_fps,
        )
        self._last_snapshot = snapshot
        return snapshot

    # ------------------------------------------------------------------
    # Insights automatiques
    # ------------------------------------------------------------------

    def _generate_insights(
        self,
        phase: str,
        phase_conf: float,
        frame_idx: int,
        n_home: int,
        n_away: int,
        new_events: list,
        new_patterns: list,
        zone_stats: dict,
    ) -> list[str]:
        insights: list[str] = []

        # Phase de jeu en cours
        if phase_conf > 0.7:
            phase_labels = {
                "melee": "⚡ Mêlée en cours",
                "touche": "📍 Touche",
                "essai": "🏆 ESSAI !",
                "ruck": "🔄 Ruck",
                "maul": "💪 Maul",
                "coup_de_pied": "👟 Coup de pied",
                "hors_jeu": "🚩 Hors-jeu / Pénalité",
            }
            if phase in phase_labels:
                insights.append(phase_labels[phase])

        # Déséquilibre numérique
        if abs(n_home - n_away) >= 3:
            more = "domicile" if n_home > n_away else "visiteur"
            insights.append(
                f"⚠️ Déséquilibre numérique : {abs(n_home - n_away)} joueurs de plus "
                f"du côté {more}"
            )

        # Événements critiques
        for ev in new_events:
            if ev.severity == "critical":
                insights.append(f"🔴 {ev.description}")
            elif ev.severity == "warning":
                insights.append(f"🟡 {ev.description}")

        # Patterns tactiques
        for pat in new_patterns:
            labels = {
                "linebreak": "💥 Linebreak détecté !",
                "pick_and_go": "🏉 Pick & Go",
                "mouvement_de_switch": "↔️ Switch de couloir",
                "defense_rideau": "🛡️ Rideau défensif en place",
                "maul_drive": "💪 Maul Drive offensif",
            }
            if pat.pattern in labels:
                insights.append(labels[pat.pattern])

        # Analyse spatiale
        for zone_name, stats in zone_stats.items():
            home_pct = stats.get("home_pct", 0)
            away_pct = stats.get("away_pct", 0)
            if "En-but adverse" in zone_name and home_pct > 20:
                insights.append(
                    f"🔵 Pression offensive : domicile présent {home_pct}% en zone d'en-but"
                )
            if "En-but adverse" in zone_name and away_pct > 20:
                insights.append(
                    f"🔴 Danger défensif : visiteur à {away_pct}% sur l'en-but domicile"
                )

        return insights[:6]  # Max 6 insights simultanés

    # ------------------------------------------------------------------
    # Générateur asynchrone pour le streaming WebSocket
    # ------------------------------------------------------------------

    async def stream(
        self,
        video_path: str,
        broadcast_interval: float = 0.1,
    ) -> AsyncIterator[AnalysisSnapshot]:
        """
        Générateur asynchrone qui traite une vidéo et yield des snapshots.
        Peut être consommé directement par un endpoint WebSocket.
        """
        from ..ingestion.video_downloader import VideoReader  # noqa: PLC0415
        cfg = load_config()
        ws_cfg = cfg["websocket"]
        self._running = True

        with VideoReader(video_path, resize=self.resize) as reader:
            logger.info(
                "Début du streaming : %s (%d frames, %.1ffps)",
                video_path, reader.total_frames, reader.fps,
            )
            self.fps_target = reader.fps

            loop = asyncio.get_event_loop()
            for frame_idx, frame in reader.frames():
                if not self._running:
                    break

                # Traitement CPU-intensif dans un thread pool
                snapshot = await loop.run_in_executor(
                    None, self.process_frame, frame, frame_idx
                )

                yield snapshot
                await asyncio.sleep(broadcast_interval)

    def stop(self):
        self._running = False

    def reset(self):
        """Réinitialise le pipeline pour un nouveau match."""
        self.tracker.reset()
        self.phase_clf.reset()
        self.event_det.reset()
        if self.heatmap_gen:
            self.heatmap_gen.reset()
        if self.pattern_rec:
            self.pattern_rec.reset()
        self._team_calibrated = False
        self._frame_times.clear()
        self._last_snapshot = None
        logger.info("Pipeline réinitialisé.")

    @property
    def last_snapshot(self) -> Optional[AnalysisSnapshot]:
        return self._last_snapshot
