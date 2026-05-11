"""
Détecteur d'événements clés d'un match de rugby.

Événements détectés :
  - Essai (try)           : joueur porteur du ballon dans l'en-but
  - Touche (lineout)      : ballon sorti du terrain
  - Mêlée (scrum)         : regroupement statique des avants
  - Ruck / Maul           : issu de la phase classifier
  - Coup de pied (kick)   : mouvement balistique du ballon
  - Faute / Hors-jeu      : signalement arbitre

Chaque événement génère un GameEvent horodaté avec les méta-données.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import yaml

from ..detection.player_detector import FrameResult, TrackedObject

logger = logging.getLogger("rugby_ia.analysis.events")


def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------

@dataclass
class GameEvent:
    event_type: str            # "try", "lineout", "scrum", "ruck", "maul", "kick", "penalty"
    timestamp_s: float
    frame_idx: int
    confidence: float
    severity: str = "info"     # "info" | "warning" | "critical"
    team: Optional[str] = None
    player_id: Optional[int] = None
    location: Optional[tuple[float, float]] = None   # (x, y) sur le terrain normalisé
    description: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "timestamp_s": round(self.timestamp_s, 2),
            "frame_idx": self.frame_idx,
            "confidence": round(self.confidence, 3),
            "severity": self.severity,
            "team": self.team,
            "player_id": self.player_id,
            "location": self.location,
            "description": self.description,
            "metadata": self.metadata,
            "detected_at": time.time(),
        }


# ---------------------------------------------------------------------------
# Détecteur principal
# ---------------------------------------------------------------------------

class EventDetector:
    """
    Analyse les FrameResult et les phases de jeu successives pour
    détecter les événements clés d'un match de rugby.
    """

    PHASE_EVENT_MAP = {
        "melee": "scrum",
        "touche": "lineout",
        "essai": "try",
        "ruck": "ruck",
        "maul": "maul",
        "coup_de_pied": "kick",
        "hors_jeu": "penalty",
    }

    def __init__(self):
        cfg = load_config()
        ev_cfg = cfg["events"]
        self.try_zone_width: float = ev_cfg["try_zone_width"]
        self.min_event_duration: float = ev_cfg["min_event_duration"]
        self.cooldown: float = ev_cfg["cooldown_seconds"]
        self.fps: float = cfg["video"]["default_fps"]

        self._events: list[GameEvent] = []
        self._last_event_time: dict[str, float] = {}
        self._prev_phase: Optional[str] = None
        self._phase_start_frame: int = 0
        self._field_homography: Optional[np.ndarray] = None   # optionnel

    # --- Homographie terrain ---

    def set_field_homography(self, H: np.ndarray):
        """
        Définit la matrice homographique pour convertir les coordonnées
        pixel → mètres sur le terrain.
        """
        self._field_homography = H

    def pixel_to_field(
        self, px: float, py: float
    ) -> Optional[tuple[float, float]]:
        """Convertit un point pixel en coordonnées terrain (mètres)."""
        if self._field_homography is None:
            return None
        pt = np.array([[[px, py]]], dtype=np.float32)
        import cv2  # noqa: PLC0415
        transformed = cv2.perspectiveTransform(pt, self._field_homography)
        x, y = transformed[0][0]
        return (float(x), float(y))

    # --- Logique anti-doublon ---

    def _can_emit(self, event_type: str) -> bool:
        last = self._last_event_time.get(event_type, 0.0)
        return (time.monotonic() - last) >= self.cooldown

    def _record(self, event: GameEvent):
        self._events.append(event)
        self._last_event_time[event.event_type] = time.monotonic()
        logger.info(
            "[EVENT] %s @ %.1fs (conf=%.2f) — %s",
            event.event_type, event.timestamp_s, event.confidence, event.description,
        )

    # --- Analyse par phase ---

    def on_phase_change(
        self,
        new_phase: str,
        confidence: float,
        frame_idx: int,
    ) -> Optional[GameEvent]:
        """
        Appelé à chaque changement de phase détecté.
        Génère un événement si la transition est significative.
        """
        if new_phase == self._prev_phase:
            return None

        duration = (frame_idx - self._phase_start_frame) / self.fps
        event_type = self.PHASE_EVENT_MAP.get(new_phase)

        self._prev_phase = new_phase
        self._phase_start_frame = frame_idx

        if event_type and self._can_emit(event_type):
            ev = GameEvent(
                event_type=event_type,
                timestamp_s=frame_idx / self.fps,
                frame_idx=frame_idx,
                confidence=confidence,
                severity="critical" if event_type == "try" else "info",
                description=f"Phase détectée : {new_phase} (durée précédente: {duration:.1f}s)",
            )
            self._record(ev)
            return ev
        return None

    # --- Analyse par frame ---

    def analyze_frame(
        self,
        frame_result: FrameResult,
        current_phase: Optional[str] = None,
        phase_conf: float = 0.0,
    ) -> list[GameEvent]:
        """
        Analyse une FrameResult pour détecter des événements basés
        sur les positions des joueurs et du ballon.

        Returns:
            Liste d'événements détectés sur cette frame.
        """
        events: list[GameEvent] = []

        # --- Détection d'essai par position balle ---
        ball = frame_result.ball
        if ball:
            field_pos = self.pixel_to_field(*ball.center)
            if field_pos:
                fx, fy = field_pos
                # En-but : x > 95m ou x < 0
                in_try_zone = fx > (105 - self.try_zone_width) or fx < self.try_zone_width
                if in_try_zone and self._can_emit("try"):
                    team = "unknown"
                    # Le joueur le plus proche du ballon
                    nearest = self._nearest_player(frame_result, ball)
                    if nearest:
                        team = nearest.team or "unknown"
                    ev = GameEvent(
                        event_type="try",
                        timestamp_s=frame_result.timestamp_s,
                        frame_idx=frame_result.frame_idx,
                        confidence=0.85,
                        severity="critical",
                        team=team,
                        location=field_pos,
                        description=f"Essai potentiel détecté à ({fx:.1f}m, {fy:.1f}m)",
                    )
                    self._record(ev)
                    events.append(ev)

        # --- Détection de mêlée par regroupement avant ---
        events.extend(self._detect_scrum_from_positions(frame_result))

        # --- Détection touche : joueurs sur la ligne de touche ---
        events.extend(self._detect_lineout_from_positions(frame_result))

        return events

    def _nearest_player(
        self, frame_result: FrameResult, ball: TrackedObject
    ) -> Optional[TrackedObject]:
        """Retourne le joueur le plus proche du ballon."""
        players = frame_result.players
        if not players:
            return None
        bx, by = ball.center
        return min(players, key=lambda p: (p.center[0] - bx) ** 2 + (p.center[1] - by) ** 2)

    def _detect_scrum_from_positions(
        self, frame_result: FrameResult
    ) -> list[GameEvent]:
        """
        Détecte une mêlée si ≥ 8 joueurs sont regroupés en cluster serré.
        """
        if not self._can_emit("scrum"):
            return []
        players = frame_result.players
        if len(players) < 8:
            return []
        positions = np.array([p.center for p in players])
        # Vérifie si l'écart-type des positions est faible (cluster serré)
        std = positions.std(axis=0)
        if std[0] < 80 and std[1] < 60:  # pixels
            centroid = positions.mean(axis=0)
            field_pos = self.pixel_to_field(centroid[0], centroid[1])
            ev = GameEvent(
                event_type="scrum",
                timestamp_s=frame_result.timestamp_s,
                frame_idx=frame_result.frame_idx,
                confidence=0.72,
                severity="info",
                location=field_pos,
                description=f"Mêlée détectée ({len(players)} joueurs regroupés)",
            )
            self._record(ev)
            return [ev]
        return []

    def _detect_lineout_from_positions(
        self, frame_result: FrameResult
    ) -> list[GameEvent]:
        """
        Détecte une touche si des joueurs sont alignés verticalement
        près d'une ligne de touche.
        """
        if not self._can_emit("lineout"):
            return []
        players = frame_result.players
        if len(players) < 4:
            return []
        positions = np.array([p.center for p in players])
        # Cherche une ligne verticale serrée (x similaires, y variés)
        x_coords = positions[:, 0]
        if x_coords.std() < 30 and positions[:, 1].std() > 50:
            field_pos = self.pixel_to_field(
                float(x_coords.mean()), float(positions[:, 1].mean())
            )
            ev = GameEvent(
                event_type="lineout",
                timestamp_s=frame_result.timestamp_s,
                frame_idx=frame_result.frame_idx,
                confidence=0.68,
                severity="info",
                location=field_pos,
                description="Touche détectée (alignement vertical des joueurs)",
            )
            self._record(ev)
            return [ev]
        return []

    # --- Accès aux événements ---

    @property
    def all_events(self) -> list[GameEvent]:
        return list(self._events)

    def events_since(self, timestamp_s: float) -> list[GameEvent]:
        return [e for e in self._events if e.timestamp_s >= timestamp_s]

    def reset(self):
        self._events.clear()
        self._last_event_time.clear()
        self._prev_phase = None
        self._phase_start_frame = 0
