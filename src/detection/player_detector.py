"""
Détection et suivi des joueurs, arbitres et ballon.

Le moteur de détection est désormais basé sur PyTorch/torchvision,
ce qui évite la dépendance Ultralytics et reste compatible avec les
poids de modèles exportés en format PyTorch.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml

logger = logging.getLogger("rugby_ia.detection")


def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Une détection brute sur une frame."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass
class TrackedObject:
    """Un joueur / arbitre / ballon suivi dans le temps."""
    track_id: int
    detection: Detection
    team: Optional[str] = None          # "home" | "away" | "referee"
    jersey_number: Optional[int] = None
    position_history: list[tuple] = field(default_factory=list)  # [(frame, cx, cy)]

    @property
    def center(self) -> tuple[float, float]:
        return self.detection.center

    def update_history(self, frame_idx: int):
        cx, cy = self.center
        self.position_history.append((frame_idx, cx, cy))
        # Limite la mémoire à 300 frames (~12 s à 25fps)
        if len(self.position_history) > 300:
            self.position_history.pop(0)


@dataclass
class FrameResult:
    """Résultat complet de l'analyse d'une frame."""
    frame_idx: int
    timestamp_s: float
    tracked_objects: list[TrackedObject]
    raw_frame: Optional[np.ndarray] = None
    annotated_frame: Optional[np.ndarray] = None

    @property
    def players(self) -> list[TrackedObject]:
        return [o for o in self.tracked_objects if o.detection.class_name == "player"]

    @property
    def ball(self) -> Optional[TrackedObject]:
        balls = [o for o in self.tracked_objects if o.detection.class_name == "ball"]
        return balls[0] if balls else None

    @property
    def referees(self) -> list[TrackedObject]:
        return [o for o in self.tracked_objects if o.detection.class_name == "referee"]


# ---------------------------------------------------------------------------
# Détecteur PyTorch
# ---------------------------------------------------------------------------

class PlayerDetector:
    """
    Détecte joueurs, arbitres et ballon sur une frame avec un modèle PyTorch.
    Les poids fine-tunés sont pris si disponibles, sinon le modèle est initialisé
    de façon aléatoire et reste compatible avec les interfaces existantes.
    """

    CLASS_NAMES = {0: "player", 1: "referee", 2: "ball"}
    COCO_FALLBACK = {0: "player", 32: "ball"}

    def __init__(self, weights: Optional[str] = None, device: Optional[str] = None):
        cfg = load_config()["detection"]
        self.conf = cfg["confidence_threshold"]
        self.iou = cfg["iou_threshold"]
        self.device = torch.device(device or cfg.get("device", "cpu"))

        weights_path = weights or cfg.get("fine_tuned_weights")
        if weights_path and Path(weights_path).exists():
            self.model_path = weights_path
            self.use_coco_fallback = False
        else:
            self.model_path = ""
            self.use_coco_fallback = True
            logger.info("Aucun poids PyTorch fine-tuné détecté, utilisation du backend torchvision initialisé par défaut.")

        self.model = self._build_model()
        logger.info("Modèle PyTorch chargé sur %s", self.device)

    def _build_model(self):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn  # noqa: PLC0415

        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=3,
        )
        model.to(self.device)

        if self.model_path:
            state = torch.load(self.model_path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"], strict=False)
            elif isinstance(state, dict) and any(key.startswith("backbone") or key.startswith("roi_heads") for key in state):
                model.load_state_dict(state, strict=False)
            elif hasattr(state, "state_dict"):
                model.load_state_dict(state.state_dict(), strict=False)

        model.eval()
        return model

    def _to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).to(self.device)
        return tensor

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Retourne les détections sur une frame BGR."""
        if frame is None or frame.size == 0:
            return []

        image = self._to_tensor(frame)
        with torch.no_grad():
            outputs = self.model([image])[0]

        detections: list[Detection] = []
        boxes = outputs.get("boxes", torch.empty((0, 4), device=self.device))
        scores = outputs.get("scores", torch.empty((0,), device=self.device))
        labels = outputs.get("labels", torch.empty((0,), device=self.device))

        for box, score, label in zip(boxes, scores, labels):
            confidence = float(score.item())
            if confidence < self.conf:
                continue

            cid = int(label.item())
            if cid not in self.CLASS_NAMES:
                if self.use_coco_fallback and cid in self.COCO_FALLBACK:
                    class_name = self.COCO_FALLBACK[cid]
                else:
                    continue
            else:
                class_name = self.CLASS_NAMES[cid]

            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=confidence,
                    class_id=cid,
                    class_name=class_name,
                )
            )
        return detections

    def benchmark_fps(
        self,
        width: int = 1280,
        height: int = 720,
        n_warmup: int = 5,
        n_runs: int = 100,
    ) -> dict:
        """Mesure la vitesse d'inférence sur des frames synthétiques."""
        import time  # noqa: PLC0415

        dummy = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        for _ in range(n_warmup):
            self.detect(dummy)

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.detect(dummy)
            times.append(time.perf_counter() - t0)

        fps_values = [1.0 / t for t in times]
        result = {
            "fps_mean": round(sum(fps_values) / len(fps_values), 1),
            "fps_min": round(min(fps_values), 1),
            "fps_max": round(max(fps_values), 1),
            "ms_per_frame_mean": round(1000 * sum(times) / len(times), 2),
            "device": str(self.device),
            "model": str(self.model_path or "torchvision_fasterrcnn"),
            "resolution": f"{width}x{height}",
        }
        logger.info(
            "Benchmark inférence : %.1f FPS (%.2f ms/frame) sur %s [%s]",
            result["fps_mean"],
            result["ms_per_frame_mean"],
            result["device"],
            result["resolution"],
        )
        return result


# ---------------------------------------------------------------------------
# Tracker PyTorch-compatible
# ---------------------------------------------------------------------------

class PlayerTracker:
    """Suivi multi-objets basé sur les détections PyTorch."""

    def __init__(self, detector: Optional[PlayerDetector] = None):
        self.detector = detector or PlayerDetector()
        cfg = load_config()
        self.fps = cfg["video"]["default_fps"]
        self._tracks: dict[int, TrackedObject] = {}
        self._next_track_id = 0

    def track_frame(self, frame: np.ndarray, frame_idx: int) -> FrameResult:
        """Détecte + suit les objets sur une frame."""
        detections = self.detector.detect(frame)
        tracked_objects: list[TrackedObject] = []
        seen_ids: set[int] = set()

        for det in detections:
            tid = self._next_track_id
            self._next_track_id += 1

            obj = TrackedObject(track_id=tid, detection=det)
            self._tracks[tid] = obj
            obj.update_history(frame_idx)
            tracked_objects.append(obj)
            seen_ids.add(tid)

        cfg_track = load_config()["tracking"]
        max_lost = cfg_track["max_time_lost"]
        stale = [
            tid for tid in list(self._tracks)
            if tid not in seen_ids
            and frame_idx - (self._tracks[tid].position_history[-1][0] if self._tracks[tid].position_history else 0) > max_lost
        ]
        for tid in stale:
            del self._tracks[tid]

        annotated = frame.copy()
        return FrameResult(
            frame_idx=frame_idx,
            timestamp_s=frame_idx / self.fps,
            tracked_objects=tracked_objects,
            raw_frame=frame,
            annotated_frame=annotated,
        )

    def get_all_trajectories(self) -> dict[int, list[tuple]]:
        """Retourne les trajectoires de tous les tracks actifs."""
        return {tid: obj.position_history for tid, obj in self._tracks.items()}

    def reset(self):
        """Réinitialise le tracker (nouveau match)."""
        self._tracks.clear()
        self._next_track_id = 0


# ---------------------------------------------------------------------------
# Classification équipe par couleur de maillot
# ---------------------------------------------------------------------------

class TeamClassifier:
    """
    Classe les joueurs dans leur équipe (home/away) par analyse
    de la couleur dominante du maillot via K-Means sur le crop du joueur.
    """

    def __init__(self, n_teams: int = 2):
        self.n_teams = n_teams
        self._team_colors: Optional[np.ndarray] = None  # (2, 3) BGR centroids

    def fit(self, frame: np.ndarray, detections: list[Detection]):
        """
        Apprend les deux couleurs d'équipe depuis les premières détections.
        À appeler sur la première frame où les deux équipes sont visibles.
        """
        from sklearn.cluster import KMeans  # noqa: PLC0415

        crops_colors = []
        for det in detections:
            if det.class_name != "player":
                continue
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # Zone milieu du crop = maillot (évite le visage et les pieds)
            h = crop.shape[0]
            torso = crop[h // 4: 3 * h // 4, :]
            mean_color = torso.reshape(-1, 3).mean(axis=0)
            crops_colors.append(mean_color)

        if len(crops_colors) < self.n_teams:
            logger.warning("Pas assez de joueurs pour calibrer les équipes.")
            return

        km = KMeans(n_clusters=self.n_teams, n_init=10, random_state=42)
        km.fit(crops_colors)
        self._team_colors = km.cluster_centers_
        logger.info("Couleurs d'équipes calibrées : %s", self._team_colors)

    def classify(self, frame: np.ndarray, obj: TrackedObject) -> str:
        """Assigne "home" ou "away" à un joueur tracké."""
        if self._team_colors is None:
            return "unknown"

        det = obj.detection
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "unknown"

        h = crop.shape[0]
        torso = crop[h // 4: 3 * h // 4, :]
        mean_color = torso.reshape(-1, 3).mean(axis=0)

        dists = np.linalg.norm(self._team_colors - mean_color, axis=1)
        team_idx = int(np.argmin(dists))
        return "home" if team_idx == 0 else "away"
