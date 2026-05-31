"""
Détection et suivi des joueurs, arbitres et ballon.

Utilise YOLOv8 (Ultralytics) pour la détection objet et
ByteTrack pour le suivi multi-personnes entre les frames.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
import yaml

logger = logging.getLogger("rugby_ia.detection")

# Chargement paresseux pour ne pas bloquer si ultralytics n'est pas installé
_YOLO = None


def _get_yolo():
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO  # noqa: PLC0415
        _YOLO = YOLO
    return _YOLO


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
# Détecteur YOLOv8
# ---------------------------------------------------------------------------

class PlayerDetector:
    """
    Détecte joueurs, arbitres et ballon sur une frame avec YOLOv8.
    Utilise les poids fine-tunés si disponibles, sinon les poids COCO.
    """

    CLASS_NAMES = {0: "player", 1: "referee", 2: "ball"}
    # Mapping classes COCO → classes rugby (pour poids non fine-tunés)
    COCO_FALLBACK = {0: "player", 32: "ball"}  # person=0, sports ball=32

    def __init__(self, weights: Optional[str] = None, device: Optional[str] = None):
        cfg = load_config()["detection"]
        self.conf = cfg["confidence_threshold"]
        self.iou = cfg["iou_threshold"]
        self.device = device or cfg.get("device", "cpu")

        # Poids fine-tunés ou COCO
        weights_path = weights or cfg.get("fine_tuned_weights")
        if weights_path and Path(weights_path).exists():
            self.model_path = weights_path
            self.use_coco_fallback = False
        else:
            self.model_path = cfg["model_name"]  # "yolov8x.pt" téléchargé auto
            self.use_coco_fallback = True
            logger.info(
                "Poids fine-tunés absents, utilisation de %s (COCO)", self.model_path
            )

        YOLO = _get_yolo()
        self.model = YOLO(self.model_path)
        logger.info("Modèle chargé : %s | device=%s", self.model_path, self.device)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Retourne les détections sur une frame BGR.

        Args:
            frame: Image BGR (numpy array).

        Returns:
            Liste de Detection.
        """
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            cid = int(box.cls[0])
            # Filtre selon les classes pertinentes
            if self.use_coco_fallback:
                if cid not in self.COCO_FALLBACK:
                    continue
                class_name = self.COCO_FALLBACK[cid]
            else:
                if cid not in self.CLASS_NAMES:
                    continue
                class_name = self.CLASS_NAMES[cid]

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(box.conf[0]),
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
        """
        Mesure la vitesse d'inférence sur des frames synthétiques.

        Args:
            width:    Largeur de la frame de test (pixels).
            height:   Hauteur de la frame de test (pixels).
            n_warmup: Nombre de passes de chauffe (non comptées).
            n_runs:   Nombre de passes mesurées.

        Returns:
            Dict avec fps_mean, fps_min, fps_max, ms_per_frame_mean.
        """
        import time  # noqa: PLC0415

        dummy = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        # Chauffe
        for _ in range(n_warmup):
            self.detect(dummy)

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.detect(dummy)
            times.append(time.perf_counter() - t0)

        fps_values = [1.0 / t for t in times]
        result = {
            "fps_mean":          round(sum(fps_values) / len(fps_values), 1),
            "fps_min":           round(min(fps_values), 1),
            "fps_max":           round(max(fps_values), 1),
            "ms_per_frame_mean": round(1000 * sum(times) / len(times), 2),
            "device":            self.device,
            "model":             str(self.model_path),
            "resolution":        f"{width}x{height}",
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
# Tracker ByteTrack (via Ultralytics intégré)
# ---------------------------------------------------------------------------

class PlayerTracker:
    """
    Suivi multi-objets basé sur ByteTrack (intégré dans Ultralytics).
    Maintient une mémoire des trajectoires par track_id.
    """

    def __init__(self, detector: Optional[PlayerDetector] = None):
        self.detector = detector or PlayerDetector()
        cfg = load_config()
        self.fps = cfg["video"]["default_fps"]
        self._tracks: dict[int, TrackedObject] = {}  # track_id → TrackedObject

    def track_frame(self, frame: np.ndarray, frame_idx: int) -> FrameResult:
        """
        Détecte + suit les objets sur une frame.

        Args:
            frame:     Frame BGR.
            frame_idx: Index de la frame dans la vidéo.

        Returns:
            FrameResult avec les objets trackés et la frame annotée.
        """
        # Utilise le mode track intégré Ultralytics (ByteTrack)
        YOLO = _get_yolo()
        results = self.detector.model.track(
            frame,
            conf=self.detector.conf,
            iou=self.detector.iou,
            device=self.detector.device,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        tracked_objects: list[TrackedObject] = []
        seen_ids: set[int] = set()

        if results.boxes.id is not None:
            for box, tid in zip(results.boxes, results.boxes.id.int().tolist()):
                cid = int(box.cls[0])
                if self.detector.use_coco_fallback:
                    if cid not in self.detector.COCO_FALLBACK:
                        continue
                    class_name = self.detector.COCO_FALLBACK[cid]
                else:
                    if cid not in self.detector.CLASS_NAMES:
                        continue
                    class_name = self.detector.CLASS_NAMES[cid]

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                det = Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=float(box.conf[0]),
                    class_id=cid,
                    class_name=class_name,
                )

                if tid in self._tracks:
                    obj = self._tracks[tid]
                    obj.detection = det
                else:
                    obj = TrackedObject(track_id=tid, detection=det)
                    self._tracks[tid] = obj

                obj.update_history(frame_idx)
                tracked_objects.append(obj)
                seen_ids.add(tid)

        # Nettoyage des tracks perdus depuis longtemps
        cfg_track = load_config()["tracking"]
        max_lost = cfg_track["max_time_lost"]
        stale = [
            tid for tid in list(self._tracks)
            if tid not in seen_ids
            and frame_idx - (self._tracks[tid].position_history[-1][0] if self._tracks[tid].position_history else 0) > max_lost
        ]
        for tid in stale:
            del self._tracks[tid]

        annotated = results.plot()

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
