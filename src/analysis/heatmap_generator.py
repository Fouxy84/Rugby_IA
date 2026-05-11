"""
Générateur de heatmaps pour l'analyse de l'utilisation de l'espace.

Fonctionnalités :
  - Heatmap de présence globale (équipe / individu)
  - Heatmap de densité d'occupation par zone de terrain
  - Heatmap de trajectoire du ballon
  - Export PNG + JSON
  - Mise à jour incrémentale pour le temps réel
"""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")           # backend non-interactif pour le serveur
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
import yaml

from ..detection.player_detector import FrameResult, TrackedObject

logger = logging.getLogger("rugby_ia.analysis.heatmap")


def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Dimensions réelles d'un terrain de rugby (en mètres)
FIELD_LENGTH = 105.0   # x
FIELD_WIDTH  = 68.0    # y


class HeatmapGenerator:
    """
    Accumule les positions des joueurs / ballon au fil des frames
    et génère des cartes de chaleur sur l'image du terrain.
    """

    def __init__(
        self,
        field_homography: Optional[np.ndarray] = None,
        resolution: tuple[int, int] = (105, 68),
    ):
        cfg = load_config()
        hm_cfg = cfg["heatmap"]
        self.sigma: float = hm_cfg["sigma"]
        self.colormap: str = hm_cfg["colormap"]
        self.resolution = resolution

        # Grilles d'accumulation (metres → pixels sur la grille)
        self._grid_home   = np.zeros(resolution, dtype=np.float32)
        self._grid_away   = np.zeros(resolution, dtype=np.float32)
        self._grid_ball   = np.zeros(resolution, dtype=np.float32)
        self._grid_global = np.zeros(resolution, dtype=np.float32)

        # Compteur de frames
        self._n_frames = 0

        # Homographie pixel → terrain (optionnelle)
        self._H = field_homography

    # ------------------------------------------------------------------
    # Conversion coordonnées
    # ------------------------------------------------------------------

    def pixel_to_grid(
        self, px: float, py: float, frame_w: int, frame_h: int
    ) -> Optional[tuple[int, int]]:
        """
        Convertit un point pixel en case de grille.
        Si une homographie est fournie, passe d'abord en coordonnées terrain.
        Sinon, utilise une normalisation linéaire simple.
        """
        if self._H is not None:
            import cv2  # noqa: PLC0415
            pt = np.array([[[px, py]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self._H)
            mx, my = transformed[0][0]
        else:
            # Normalisation naïve : on suppose que la caméra couvre tout le terrain
            mx = (px / frame_w) * FIELD_LENGTH
            my = (py / frame_h) * FIELD_WIDTH

        gx = int(np.clip(mx, 0, FIELD_LENGTH - 0.01) / FIELD_LENGTH * self.resolution[0])
        gy = int(np.clip(my, 0, FIELD_WIDTH - 0.01)  / FIELD_WIDTH  * self.resolution[1])
        return gx, gy

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def update(self, frame_result: FrameResult):
        """
        Ajoute les positions de la frame courante aux grilles d'accumulation.
        """
        if frame_result.raw_frame is None:
            return

        h, w = frame_result.raw_frame.shape[:2]
        self._n_frames += 1

        for obj in frame_result.tracked_objects:
            cx, cy = obj.center
            cell = self.pixel_to_grid(cx, cy, w, h)
            if cell is None:
                continue
            gx, gy = cell

            if obj.detection.class_name == "ball":
                self._grid_ball[gx, gy] += 1.0
            elif obj.detection.class_name == "player":
                self._grid_global[gx, gy] += 1.0
                team = obj.team or "unknown"
                if team == "home":
                    self._grid_home[gx, gy] += 1.0
                elif team == "away":
                    self._grid_away[gx, gy] += 1.0

    # ------------------------------------------------------------------
    # Génération des cartes
    # ------------------------------------------------------------------

    def _smooth(self, grid: np.ndarray) -> np.ndarray:
        """Applique un filtre gaussien et normalise."""
        smoothed = gaussian_filter(grid.T, sigma=self.sigma)   # transpose pour axes (y, x)
        if smoothed.max() > 0:
            smoothed = smoothed / smoothed.max()
        return smoothed

    def generate(
        self,
        mode: str = "global",
        field_img_path: Optional[str] = None,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> np.ndarray:
        """
        Génère et retourne une heatmap sous forme d'image numpy (RGB).

        Args:
            mode:           "global" | "home" | "away" | "ball"
            field_img_path: Chemin vers l'image de fond du terrain (optionnel).
            save_path:      Chemin de sauvegarde PNG (optionnel).
            title:          Titre du graphique.

        Returns:
            Image numpy RGB (H, W, 3).
        """
        grid_map = {
            "global": self._grid_global,
            "home":   self._grid_home,
            "away":   self._grid_away,
            "ball":   self._grid_ball,
        }
        grid = grid_map.get(mode, self._grid_global)
        heatmap = self._smooth(grid)

        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

        # Fond terrain
        if field_img_path and Path(field_img_path).exists():
            import cv2  # noqa: PLC0415
            bg = cv2.imread(str(field_img_path))
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
            ax.imshow(bg, extent=[0, FIELD_LENGTH, 0, FIELD_WIDTH], aspect="auto")
        else:
            ax.set_facecolor("#2d6a2d")
            # Lignes de terrain simplifiées
            self._draw_field_lines(ax)

        # Heatmap superposée
        cmap = plt.get_cmap(self.colormap)
        cmap_alpha = self._cmap_with_alpha(cmap)
        im = ax.imshow(
            heatmap,
            extent=[0, FIELD_LENGTH, 0, FIELD_WIDTH],
            origin="upper",
            cmap=cmap_alpha,
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
        )
        plt.colorbar(im, ax=ax, fraction=0.03, label="Densité normalisée")

        mode_labels = {
            "global": "Tous joueurs",
            "home": "Équipe domicile",
            "away": "Équipe visiteur",
            "ball": "Trajectoire ballon",
        }
        ax.set_title(title or f"Heatmap — {mode_labels.get(mode, mode)}", fontsize=14)
        ax.set_xlabel("Longueur (m)")
        ax.set_ylabel("Largeur (m)")

        # Annotations zones
        cfg = load_config()
        for zone in cfg["heatmap"]["zones"]:
            x0, x1 = zone["x"]
            y0, y1 = zone["y"]
            rect = plt.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=1, edgecolor="white", facecolor="none", alpha=0.5
            )
            ax.add_patch(rect)
            ax.text(
                (x0 + x1) / 2, (y0 + y1) / 2, zone["name"],
                ha="center", va="center", color="white", fontsize=7, alpha=0.8
            )

        plt.tight_layout()

        # Conversion en numpy
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w_px, h_px = fig.canvas.get_width_height()
        img = buf.reshape(h_px, w_px, 3)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=100)
            logger.info("Heatmap sauvegardée : %s", save_path)

        plt.close(fig)
        return img

    @staticmethod
    def _cmap_with_alpha(cmap):
        """Rend le bas du colormap transparent pour ne pas masquer le terrain."""
        cmap_data = cmap(np.linspace(0, 1, 256))
        cmap_data[:, -1] = np.linspace(0, 0.85, 256)  # alpha 0→0.85
        return mcolors.ListedColormap(cmap_data)

    @staticmethod
    def _draw_field_lines(ax):
        """Dessine les lignes principales d'un terrain de rugby."""
        ax.set_xlim(0, FIELD_LENGTH)
        ax.set_ylim(0, FIELD_WIDTH)

        lw = 1.2
        color = "white"
        alpha = 0.6

        # Contour
        for x in [0, 22, 52.5, 83, 105]:
            ax.axvline(x=x, color=color, linewidth=lw, alpha=alpha)
        for y in [0, FIELD_WIDTH]:
            ax.axhline(y=y, color=color, linewidth=lw, alpha=alpha)

        # En-but
        for x in [0, 105]:
            ax.axvline(x=x, color=color, linewidth=2.0, alpha=0.9)

    def zone_statistics(self) -> dict:
        """
        Retourne des statistiques de possession par zone pour
        chaque équipe.
        """
        cfg = load_config()
        stats = {}
        total_home = self._grid_home.sum() + 1e-9
        total_away = self._grid_away.sum() + 1e-9

        for zone in cfg["heatmap"]["zones"]:
            zname = zone["name"]
            x0, x1 = zone["x"]
            y0, y1 = zone["y"]
            gx0 = int(x0 / FIELD_LENGTH * self.resolution[0])
            gx1 = int(x1 / FIELD_LENGTH * self.resolution[0])
            gy0 = int(y0 / FIELD_WIDTH  * self.resolution[1])
            gy1 = int(y1 / FIELD_WIDTH  * self.resolution[1])

            home_z = self._grid_home[gx0:gx1, gy0:gy1].sum()
            away_z = self._grid_away[gx0:gx1, gy0:gy1].sum()
            stats[zname] = {
                "home_pct": round(100 * home_z / total_home, 1),
                "away_pct": round(100 * away_z / total_away, 1),
            }
        return stats

    def reset(self):
        self._grid_home[:] = 0
        self._grid_away[:] = 0
        self._grid_ball[:] = 0
        self._grid_global[:] = 0
        self._n_frames = 0
