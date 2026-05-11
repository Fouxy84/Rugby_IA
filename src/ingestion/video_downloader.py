"""
Module d'ingestion vidéo :
- Téléchargement depuis YouTube / URLs directes via yt-dlp
- Chargement de fichiers locaux
- Accès aux flux RSS/API de données rugby publiques
"""

import os
import logging
from pathlib import Path
from typing import Optional, Generator
import cv2
import yt_dlp
import requests
import yaml

logger = logging.getLogger("rugby_ia.ingestion")


def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Téléchargement vidéo
# ---------------------------------------------------------------------------

class VideoDownloader:
    """
    Télécharge des matchs de rugby depuis YouTube ou n'importe quelle URL
    supportée par yt-dlp (Dailymotion, Vimeo, ESPN, etc.).
    """

    def __init__(self, output_dir: Optional[str] = None):
        cfg = load_config()
        self.output_dir = Path(output_dir or cfg["paths"]["data_raw"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format_selector = cfg["data_sources"]["youtube_dl_format"]

    def _build_ydl_opts(self, filename: str) -> dict:
        return {
            "format": self.format_selector,
            "outtmpl": str(self.output_dir / f"{filename}.%(ext)s"),
            "quiet": False,
            "no_warnings": False,
            "merge_output_format": "mp4",
            "postprocessors": [
                {
                    "key": "FFmpegVideoConvertor",
                    "preferedformat": "mp4",
                }
            ],
        }

    def download(self, url: str, filename: Optional[str] = None) -> Path:
        """
        Télécharge une vidéo et retourne le chemin local du fichier.

        Args:
            url:      URL YouTube ou autre plateforme compatible yt-dlp.
            filename: Nom de fichier (sans extension). Si None, utilise le titre.

        Returns:
            Chemin vers le fichier téléchargé.
        """
        if filename is None:
            # Récupère le titre pour nommer le fichier
            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                info = ydl.extract_info(url, download=False)
                filename = ydl.prepare_filename(info).rsplit(".", 1)[0]
                filename = Path(filename).name  # garde uniquement le nom de base

        opts = self._build_ydl_opts(filename)
        logger.info("Téléchargement de %s → %s", url, filename)
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

        # Retrouve le fichier téléchargé
        for ext in ("mp4", "webm", "mkv", "avi"):
            candidate = self.output_dir / f"{filename}.{ext}"
            if candidate.exists():
                logger.info("Fichier prêt : %s", candidate)
                return candidate

        raise FileNotFoundError(
            f"Fichier téléchargé introuvable dans {self.output_dir} pour '{filename}'"
        )

    def list_available(self, url: str) -> list[dict]:
        """Retourne les formats disponibles pour une URL sans télécharger."""
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        return info.get("formats", [])


# ---------------------------------------------------------------------------
# Lecteur de vidéo locale (frame-by-frame)
# ---------------------------------------------------------------------------

class VideoReader:
    """
    Lit une vidéo locale et expose un générateur de frames ainsi que
    ses métadonnées (fps, résolution, durée).
    """

    def __init__(self, video_path: str | Path, resize: Optional[tuple[int, int]] = None):
        self.path = Path(video_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Vidéo introuvable : {self.path}")

        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise IOError(f"Impossible d'ouvrir la vidéo : {self.path}")

        self.fps: float = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames: int = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_s: float = self.total_frames / self.fps
        self.resize = resize

    @property
    def metadata(self) -> dict:
        return {
            "path": str(self.path),
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
            "duration_s": round(self.duration_s, 2),
        }

    def frames(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1,
    ) -> Generator:
        """
        Générateur de frames (frame_index, frame_bgr).

        Args:
            start_frame: Index de départ (inclusif).
            end_frame:   Index de fin (exclusif). None = jusqu'à la fin.
            step:        Pas (1 = toutes les frames, 2 = une sur deux…).
        """
        if end_frame is None:
            end_frame = self.total_frames

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        idx = start_frame
        while idx < end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break
            if self.resize:
                frame = cv2.resize(frame, self.resize)
            yield idx, frame
            # Sauter des frames si step > 1
            for _ in range(step - 1):
                self.cap.read()
                idx += 1
            idx += 1

    def frame_at(self, frame_index: int):
        """Retourne une frame spécifique par son index."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if not ret:
            raise IndexError(f"Frame {frame_index} introuvable")
        if self.resize:
            frame = cv2.resize(frame, self.resize)
        return frame

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


# ---------------------------------------------------------------------------
# Connecteur aux données publiques rugby
# ---------------------------------------------------------------------------

class RugbyDataConnector:
    """
    Récupère des informations de matchs depuis des APIs rugby publiques.
    Sources :  ESPN Rugby, World Rugby (si clé API fournie).
    """

    ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/rugby"
    LEAGUES = {
        "top14": "top.14",
        "premiership": "eng.1",
        "super_rugby": "pac.1",
        "six_nations": "irb.6nations",
        "world_cup": "irb.worldcup",
        "champions_cup": "cha.1",
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("WORLD_RUGBY_API_KEY", "")
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def get_recent_matches(self, league: str = "top14", limit: int = 10) -> list[dict]:
        """
        Retourne les matchs récents d'une compétition via l'API ESPN.

        Args:
            league: Identifiant de compétition (top14, premiership, etc.).
            limit:  Nombre maximum de matchs.

        Returns:
            Liste de dicts avec id, date, équipes, score, statut.
        """
        league_id = self.LEAGUES.get(league, league)
        url = f"{self.ESPN_BASE}/{league_id}/scoreboard"
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            events = data.get("events", [])[:limit]
            matches = []
            for ev in events:
                comp = ev.get("competitions", [{}])[0]
                competitors = comp.get("competitors", [])
                teams = [
                    {
                        "name": c.get("team", {}).get("displayName"),
                        "score": c.get("score"),
                        "home": c.get("homeAway") == "home",
                    }
                    for c in competitors
                ]
                matches.append(
                    {
                        "id": ev.get("id"),
                        "name": ev.get("name"),
                        "date": ev.get("date"),
                        "status": ev.get("status", {}).get("type", {}).get("description"),
                        "teams": teams,
                    }
                )
            return matches
        except requests.RequestException as exc:
            logger.warning("Erreur ESPN API : %s", exc)
            return []

    def search_match_videos(self, query: str) -> list[dict]:
        """
        Recherche des vidéos de matchs sur YouTube via yt-dlp (sans télécharger).

        Args:
            query: Terme de recherche (ex: "Top14 2024 Toulouse Racing highlights").

        Returns:
            Liste de dicts {title, url, duration, view_count}.
        """
        search_url = f"ytsearch10:{query} rugby match highlights"
        results = []
        with yt_dlp.YoutubeDL({"quiet": True, "extract_flat": True}) as ydl:
            try:
                info = ydl.extract_info(search_url, download=False)
                for entry in info.get("entries", []):
                    if entry:
                        results.append(
                            {
                                "title": entry.get("title"),
                                "url": f"https://www.youtube.com/watch?v={entry.get('id')}",
                                "duration": entry.get("duration"),
                                "view_count": entry.get("view_count"),
                            }
                        )
            except Exception as exc:
                logger.warning("Erreur recherche YouTube : %s", exc)
        return results
