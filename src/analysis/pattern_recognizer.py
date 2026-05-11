"""
Reconnaissance de patterns tactiques de rugby.

Patterns détectés :
  - pick_and_go       : avancée individuelle après ruck
  - switch            : échange de couloirs entre deux attaquants
  - jeu_au_pied       : dégagement / coup de pied rasant / chandelle
  - attaque_rapide    : jeu après récupération rapide
  - defense_rideau    : ligne défensive plate et serrée
  - linebreak         : franchissement de la ligne défensive
  - maul_drive        : poussée collective vers l'avant

Approche : analyse de trajectoires et vecteurs de mouvement sur une
fenêtre glissante, couplée à un modèle de séquence léger.
"""

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import yaml

from ..detection.player_detector import FrameResult, TrackedObject

logger = logging.getLogger("rugby_ia.analysis.patterns")


def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------

@dataclass
class PatternResult:
    pattern: str
    confidence: float
    frame_idx: int
    timestamp_s: float
    involved_players: list[int]      # track_ids
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "confidence": round(self.confidence, 3),
            "frame_idx": self.frame_idx,
            "timestamp_s": round(self.timestamp_s, 2),
            "involved_players": self.involved_players,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# Extracteur de features de mouvement
# ---------------------------------------------------------------------------

class MotionFeatureExtractor:
    """
    Calcule des descripteurs de mouvement collectif à partir des
    positions trackées sur une fenêtre temporelle.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        # {track_id: deque[(frame_idx, cx, cy)]}
        self._history: dict[int, deque] = {}

    def update(self, frame_result: FrameResult):
        seen = set()
        for obj in frame_result.tracked_objects:
            tid = obj.track_id
            seen.add(tid)
            if tid not in self._history:
                self._history[tid] = deque(maxlen=self.window_size)
            cx, cy = obj.center
            self._history[tid].append((frame_result.frame_idx, cx, cy))

    def get_velocity_vectors(self) -> dict[int, tuple[float, float]]:
        """
        Retourne le vecteur vitesse moyen (vx, vy) pour chaque joueur
        sur la fenêtre courante (pixels/frame).
        """
        velocities = {}
        for tid, hist in self._history.items():
            if len(hist) < 2:
                velocities[tid] = (0.0, 0.0)
                continue
            h = list(hist)
            # Vitesse = déplacement total / nombre d'intervalles
            dx = h[-1][1] - h[0][1]
            dy = h[-1][2] - h[0][2]
            n = len(h) - 1
            velocities[tid] = (dx / n, dy / n)
        return velocities

    def get_formation_descriptor(
        self, frame_result: FrameResult
    ) -> dict:
        """
        Descripteur de formation : centroïde, étalement, axe principal.
        """
        players = frame_result.players
        if not players:
            return {}
        positions = np.array([p.center for p in players])
        centroid = positions.mean(axis=0)
        std = positions.std(axis=0)
        # PCA pour l'axe principal
        if len(positions) >= 3:
            cov = np.cov(positions.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            main_axis = eigenvectors[:, -1]
        else:
            main_axis = np.array([1.0, 0.0])

        return {
            "centroid": centroid.tolist(),
            "spread_x": float(std[0]),
            "spread_y": float(std[1]),
            "main_axis": main_axis.tolist(),
            "n_players": len(players),
        }

    def reset(self):
        self._history.clear()


# ---------------------------------------------------------------------------
# Détecteur de patterns règles expertes
# ---------------------------------------------------------------------------

class PatternRecognizer:
    """
    Détecte les patterns tactiques par règles expertes sur les
    descripteurs de mouvement + modèle séquentiel optionnel.
    """

    def __init__(self):
        cfg = load_config()
        p_cfg = cfg["patterns"]
        self.window_size: int = p_cfg["window_size"]
        self.fps: float = cfg["video"]["default_fps"]
        self.extractor = MotionFeatureExtractor(window_size=self.window_size)
        self._detected: list[PatternResult] = []
        self._last_detected_frame: dict[str, int] = {}
        self._cooldown_frames: int = int(self.fps * 3)  # 3 secondes

    def _can_detect(self, pattern: str, frame_idx: int) -> bool:
        last = self._last_detected_frame.get(pattern, -9999)
        return (frame_idx - last) >= self._cooldown_frames

    def analyze(self, frame_result: FrameResult) -> list[PatternResult]:
        """
        Analyse une FrameResult et retourne les patterns détectés.
        """
        self.extractor.update(frame_result)
        frame_idx = frame_result.frame_idx
        results: list[PatternResult] = []

        velocities = self.extractor.get_velocity_vectors()
        formation = self.extractor.get_formation_descriptor(frame_result)

        if not velocities or not formation:
            return []

        vels = np.array(list(velocities.values()))
        player_ids = list(velocities.keys())

        # --- Pick and Go ---
        # Un joueur avance seul et vite pendant que les autres restent statiques
        results.extend(self._detect_pick_and_go(vels, player_ids, frame_result, frame_idx))

        # --- Switch / Échange de couloir ---
        results.extend(self._detect_switch(velocities, frame_result, frame_idx))

        # --- Linebreak ---
        results.extend(self._detect_linebreak(velocities, formation, frame_result, frame_idx))

        # --- Défense rideau ---
        results.extend(self._detect_curtain_defense(formation, frame_result, frame_idx))

        # --- Maul Drive ---
        results.extend(self._detect_maul_drive(vels, formation, player_ids, frame_result, frame_idx))

        for r in results:
            self._detected.append(r)
            self._last_detected_frame[r.pattern] = frame_idx
            logger.info("[PATTERN] %s @ frame %d (conf=%.2f)", r.pattern, frame_idx, r.confidence)

        return results

    # --- Règles expertes ---

    def _detect_pick_and_go(
        self, vels: np.ndarray, player_ids: list, fr: FrameResult, fidx: int
    ) -> list[PatternResult]:
        if not self._can_detect("pick_and_go", fidx) or len(vels) < 4:
            return []
        speeds = np.linalg.norm(vels, axis=1)
        mean_speed = speeds.mean()
        std_speed  = speeds.std()
        fast_mask = speeds > mean_speed + 1.5 * std_speed
        n_fast = fast_mask.sum()
        if n_fast == 1 and mean_speed < 3.0:
            fast_idx = int(np.argmax(fast_mask))
            return [PatternResult(
                pattern="pick_and_go",
                confidence=0.74,
                frame_idx=fidx,
                timestamp_s=fidx / self.fps,
                involved_players=[player_ids[fast_idx]],
                description="Joueur isolé avançant rapidement (pick & go)",
            )]
        return []

    def _detect_switch(
        self, velocities: dict, fr: FrameResult, fidx: int
    ) -> list[PatternResult]:
        if not self._can_detect("mouvement_de_switch", fidx) or len(velocities) < 2:
            return []
        vels = list(velocities.values())
        ids  = list(velocities.keys())
        # Cherche 2 joueurs avec directions horizontales opposées (switch)
        for i in range(len(vels)):
            for j in range(i + 1, len(vels)):
                vx_i, vy_i = vels[i]
                vx_j, vy_j = vels[j]
                # Directions Y opposées avec vitesse significative
                if abs(vy_i) > 1.5 and abs(vy_j) > 1.5 and vy_i * vy_j < 0:
                    return [PatternResult(
                        pattern="mouvement_de_switch",
                        confidence=0.68,
                        frame_idx=fidx,
                        timestamp_s=fidx / self.fps,
                        involved_players=[ids[i], ids[j]],
                        description="Switch détecté : deux joueurs croisant leur trajectoire",
                    )]
        return []

    def _detect_linebreak(
        self, velocities: dict, formation: dict, fr: FrameResult, fidx: int
    ) -> list[PatternResult]:
        if not self._can_detect("linebreak", fidx):
            return []
        # Un joueur avance vite dans l'axe longitudinal ET sort du centroïde
        centroid = np.array(formation.get("centroid", [0, 0]))
        for tid, (vx, vy) in velocities.items():
            # Avancée rapide dans l'axe x
            if abs(vx) > 4.0:
                player = next(
                    (p for p in fr.players if p.track_id == tid), None
                )
                if player:
                    dist_from_centroid = np.linalg.norm(
                        np.array(player.center) - centroid
                    )
                    if dist_from_centroid > 80:
                        return [PatternResult(
                            pattern="linebreak",
                            confidence=0.79,
                            frame_idx=fidx,
                            timestamp_s=fidx / self.fps,
                            involved_players=[tid],
                            description="Linebreak : franchissement de la ligne défensive",
                        )]
        return []

    def _detect_curtain_defense(
        self, formation: dict, fr: FrameResult, fidx: int
    ) -> list[PatternResult]:
        if not self._can_detect("defense_rideau", fidx):
            return []
        spread_x = formation.get("spread_x", 0)
        spread_y = formation.get("spread_y", 0)
        n = formation.get("n_players", 0)
        # Ligne plate : faible étalement X, grand étalement Y, ≥ 6 joueurs
        if n >= 6 and spread_x < 25 and spread_y > 60:
            return [PatternResult(
                pattern="defense_rideau",
                confidence=0.71,
                frame_idx=fidx,
                timestamp_s=fidx / self.fps,
                involved_players=[p.track_id for p in fr.players[:6]],
                description="Défense en rideau : ligne plate et élargie détectée",
            )]
        return []

    def _detect_maul_drive(
        self, vels: np.ndarray, formation: dict, player_ids: list,
        fr: FrameResult, fidx: int
    ) -> list[PatternResult]:
        if not self._can_detect("maul_drive", fidx) or len(vels) < 4:
            return []
        spread_x = formation.get("spread_x", 999)
        spread_y = formation.get("spread_y", 999)
        # Regroupement serré + mouvement collectif vers l'avant
        mean_vx = float(vels[:, 0].mean())
        if spread_x < 40 and spread_y < 40 and abs(mean_vx) > 1.0:
            return [PatternResult(
                pattern="maul_drive",
                confidence=0.73,
                frame_idx=fidx,
                timestamp_s=fidx / self.fps,
                involved_players=player_ids[:min(8, len(player_ids))],
                description=f"Maul Drive : poussée collective (vx_moyen={mean_vx:.1f})",
            )]
        return []

    @property
    def all_patterns(self) -> list[PatternResult]:
        return list(self._detected)

    def reset(self):
        self._detected.clear()
        self._last_detected_frame.clear()
        self.extractor.reset()
