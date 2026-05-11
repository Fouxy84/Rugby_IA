"""
Classificateur de phases de jeu rugby.

Architecture : CNN léger (MobileNetV3) + LSTM temporel
Input : séquence de N frames annotées (heatmap positions joueurs + frame RGB)
Output : phase parmi {jeu_courant, melee, touche, essai, coup_de_pied,
                       ruck, maul, hors_jeu, remise_en_jeu}
"""

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import yaml

logger = logging.getLogger("rugby_ia.analysis.phase")


def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------

@dataclass
class PhaseResult:
    phase: str
    confidence: float
    top3: list[tuple[str, float]]       # [(phase, proba), ...]
    frame_idx: int
    timestamp_s: float


# ---------------------------------------------------------------------------
# Modèle CNN-LSTM
# ---------------------------------------------------------------------------

class PhaseClassifierModel(nn.Module):
    """
    Modèle léger pour la classification de phases de jeu.

    Input  : (batch, seq_len, C, H, W) — séquence de frames redimensionnées
    Output : (batch, n_classes) — logits
    """

    def __init__(self, n_classes: int = 9, seq_len: int = 30):
        super().__init__()
        self.seq_len = seq_len

        # Extracteur de features par frame (MobileNetV3 Small)
        import torchvision.models as models  # noqa: PLC0415
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        # Supprime le classifieur final
        self.cnn = nn.Sequential(*list(backbone.features.children()))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        cnn_out_dim = 576  # sortie MobileNetV3 Small

        # Encodeur temporel LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        # Traitement CNN frame par frame
        x = x.view(b * t, c, h, w)
        x = self.cnn(x)
        x = self.pool(x).view(b, t, -1)
        # LSTM
        _, (hn, _) = self.lstm(x)
        out = self.classifier(hn[-1])
        return out


# ---------------------------------------------------------------------------
# Classificateur haut niveau
# ---------------------------------------------------------------------------

class PhaseClassifier:
    """
    Prédit la phase de jeu en cours à partir d'une séquence de frames.
    Maintient un buffer glissant pour la prédiction en temps réel.
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        cfg = load_config()
        phase_cfg = cfg["phase_classifier"]

        self.classes: list[str] = phase_cfg["classes"]
        self.n_classes = len(self.classes)
        self.seq_len: int = phase_cfg["sequence_length"]
        self.conf_thresh: float = phase_cfg["confidence_threshold"]
        self.fps: float = cfg["video"]["default_fps"]

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = PhaseClassifierModel(
            n_classes=self.n_classes,
            seq_len=self.seq_len,
        ).to(self.device)

        # Chargement des poids si disponibles
        weights = model_path or phase_cfg.get("model_path", "")
        if weights and Path(weights).exists():
            state = torch.load(weights, map_location=self.device)
            self.model.load_state_dict(state)
            logger.info("Phase classifier chargé : %s", weights)
        else:
            logger.warning(
                "Poids phase classifier absents (%s). Mode aléatoire activé.", weights
            )

        self.model.eval()

        # Buffer glissant de frames prétraitées
        self._frame_buffer: deque[torch.Tensor] = deque(maxlen=self.seq_len)
        self._last_result: Optional[PhaseResult] = None

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Redimensionne et normalise une frame BGR → tensor (3, 112, 112)."""
        import cv2  # noqa: PLC0415
        from torchvision import transforms  # noqa: PLC0415
        img = cv2.resize(frame, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = transforms.ToTensor()(img)
        t = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(t)
        return t

    def update(self, frame: np.ndarray, frame_idx: int) -> Optional[PhaseResult]:
        """
        Ajoute une frame au buffer et prédit la phase si le buffer est plein.

        Returns:
            PhaseResult ou None si le buffer n'est pas encore plein.
        """
        self._frame_buffer.append(self._preprocess(frame))

        if len(self._frame_buffer) < self.seq_len:
            return None

        seq = torch.stack(list(self._frame_buffer)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(seq)
            probas = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        top_idx = int(np.argmax(probas))
        confidence = float(probas[top_idx])
        phase = self.classes[top_idx]

        top3 = sorted(
            [(self.classes[i], float(probas[i])) for i in range(self.n_classes)],
            key=lambda x: x[1],
            reverse=True,
        )[:3]

        result = PhaseResult(
            phase=phase,
            confidence=confidence,
            top3=top3,
            frame_idx=frame_idx,
            timestamp_s=frame_idx / self.fps,
        )
        self._last_result = result
        return result

    @property
    def current_phase(self) -> Optional[str]:
        return self._last_result.phase if self._last_result else None

    def reset(self):
        self._frame_buffer.clear()
        self._last_result = None
