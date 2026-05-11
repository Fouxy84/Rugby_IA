"""
API FastAPI — Rugby IA
Endpoints REST + WebSocket pour l'analyse temps réel.

Routes :
  POST /api/video/upload          — Upload d'un fichier vidéo
  POST /api/video/download        — Téléchargement depuis URL
  GET  /api/video/search          — Recherche de matchs en ligne
  GET  /api/matches               — Liste des matchs disponibles
  GET  /api/matches/{id}/events   — Événements d'un match
  GET  /api/matches/{id}/heatmap  — Heatmap PNG d'un match
  GET  /api/matches/{id}/insights — Key insights d'un match
  GET  /api/leagues/{league}      — Matchs récents d'une compétition
  WS   /ws/analysis/{match_id}    — Stream temps réel d'un match
"""

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import yaml
from fastapi import (
    FastAPI, File, Form, HTTPException, UploadFile, WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ..ingestion.video_downloader import VideoDownloader, RugbyDataConnector
from ..pipeline.realtime_pipeline import RealtimePipeline

logger = logging.getLogger("rugby_ia.api")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_config()
DATA_RAW   = Path(CFG["paths"]["data_raw"])
DATA_OUT   = Path(CFG["paths"]["outputs"])
HEATMAP_DIR = Path(CFG["paths"]["heatmaps"])

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_OUT.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = CFG["api"]["max_upload_size_mb"] * 1024 * 1024

# ---------------------------------------------------------------------------
# Application FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Rugby IA API",
    description="Analyse IA temps réel de matchs de rugby",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CFG["api"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# State partagé : pipelines actifs par match_id
_pipelines: dict[str, RealtimePipeline] = {}
_match_metadata: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Modèles Pydantic
# ---------------------------------------------------------------------------

class DownloadRequest(BaseModel):
    url: str
    filename: Optional[str] = None

class SearchRequest(BaseModel):
    query: str

class AnalysisRequest(BaseModel):
    match_id: str
    enable_heatmap: bool = True
    enable_patterns: bool = True


# ---------------------------------------------------------------------------
# Routes Vidéo
# ---------------------------------------------------------------------------

@app.post("/api/video/upload", summary="Upload d'un fichier vidéo")
async def upload_video(
    file: UploadFile = File(...),
    match_name: str = Form(default=""),
):
    """Reçoit un fichier vidéo et le stocke localement."""
    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in CFG["video"]["supported_formats"]:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté : {ext}. Formats acceptés : {CFG['video']['supported_formats']}",
        )

    match_id = str(uuid.uuid4())[:8]
    filename = f"{match_id}_{file.filename}"
    dest = DATA_RAW / filename

    # Lecture par chunks pour éviter la surcharge mémoire
    size = 0
    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Fichier trop volumineux (max {CFG['api']['max_upload_size_mb']} MB)",
                )
            f.write(chunk)

    _match_metadata[match_id] = {
        "match_id": match_id,
        "filename": filename,
        "path": str(dest),
        "match_name": match_name or file.filename,
        "size_mb": round(size / 1024 / 1024, 1),
        "source": "upload",
    }
    logger.info("Vidéo uploadée : %s (%s MB)", filename, _match_metadata[match_id]["size_mb"])
    return {"match_id": match_id, "filename": filename, "size_mb": _match_metadata[match_id]["size_mb"]}


@app.post("/api/video/download", summary="Télécharger une vidéo depuis une URL")
async def download_video(req: DownloadRequest):
    """Télécharge une vidéo depuis YouTube ou une URL yt-dlp compatible."""
    try:
        downloader = VideoDownloader(output_dir=str(DATA_RAW))
        loop = asyncio.get_event_loop()
        video_path = await loop.run_in_executor(
            None, downloader.download, req.url, req.filename
        )
        match_id = str(uuid.uuid4())[:8]
        _match_metadata[match_id] = {
            "match_id": match_id,
            "filename": video_path.name,
            "path": str(video_path),
            "match_name": req.filename or video_path.stem,
            "size_mb": round(video_path.stat().st_size / 1024 / 1024, 1),
            "source": "download",
            "url": req.url,
        }
        return {"match_id": match_id, "filename": video_path.name}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/video/search", summary="Rechercher des matchs en ligne")
async def search_videos(q: str):
    """Recherche des vidéos de rugby sur YouTube sans télécharger."""
    connector = RugbyDataConnector()
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, connector.search_match_videos, q)
    return {"results": results}


# ---------------------------------------------------------------------------
# Routes Matchs
# ---------------------------------------------------------------------------

@app.get("/api/matches", summary="Lister les matchs disponibles")
async def list_matches():
    """Retourne tous les matchs uploadés ou téléchargés."""
    return {"matches": list(_match_metadata.values())}


@app.get("/api/matches/{match_id}", summary="Détail d'un match")
async def get_match(match_id: str):
    if match_id not in _match_metadata:
        raise HTTPException(status_code=404, detail="Match introuvable")
    return _match_metadata[match_id]


@app.get("/api/matches/{match_id}/events", summary="Événements détectés")
async def get_events(match_id: str):
    pipeline = _pipelines.get(match_id)
    if not pipeline:
        return {"events": []}
    return {
        "events": [e.to_dict() for e in pipeline.event_det.all_events]
    }


@app.get("/api/matches/{match_id}/heatmap", summary="Heatmap PNG")
async def get_heatmap(
    match_id: str,
    mode: str = "global",
):
    """Retourne la heatmap PNG pour un match analysé."""
    pipeline = _pipelines.get(match_id)
    if not pipeline or not pipeline.heatmap_gen:
        raise HTTPException(status_code=404, detail="Heatmap non disponible")

    save_path = HEATMAP_DIR / f"{match_id}_{mode}.png"
    pipeline.heatmap_gen.generate(
        mode=mode,
        save_path=str(save_path),
        title=f"Heatmap {mode} — {_match_metadata.get(match_id, {}).get('match_name', match_id)}",
    )
    return FileResponse(str(save_path), media_type="image/png")


@app.get("/api/matches/{match_id}/insights", summary="Key insights d'un match")
async def get_insights(match_id: str):
    pipeline = _pipelines.get(match_id)
    if not pipeline or not pipeline.last_snapshot:
        return {"insights": []}
    snap = pipeline.last_snapshot
    return {
        "insights": snap.key_insights,
        "current_phase": snap.phase,
        "zone_stats": snap.zone_stats,
        "n_events": len(pipeline.event_det.all_events),
        "n_patterns": len(pipeline.pattern_rec.all_patterns) if pipeline.pattern_rec else 0,
    }


@app.get("/api/matches/{match_id}/patterns", summary="Patterns tactiques")
async def get_patterns(match_id: str):
    pipeline = _pipelines.get(match_id)
    if not pipeline or not pipeline.pattern_rec:
        return {"patterns": []}
    return {
        "patterns": [p.to_dict() for p in pipeline.pattern_rec.all_patterns]
    }


# ---------------------------------------------------------------------------
# Routes Ligues / compétitions
# ---------------------------------------------------------------------------

@app.get("/api/leagues/{league}", summary="Matchs récents d'une compétition")
async def get_league_matches(league: str, limit: int = 10):
    """Récupère les matchs récents depuis l'API ESPN."""
    connector = RugbyDataConnector()
    loop = asyncio.get_event_loop()
    matches = await loop.run_in_executor(
        None, connector.get_recent_matches, league, limit
    )
    return {"league": league, "matches": matches}


# ---------------------------------------------------------------------------
# WebSocket — Streaming temps réel
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Gère les connexions WebSocket actives."""

    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, ws: WebSocket, match_id: str):
        await ws.accept()
        self._connections.setdefault(match_id, []).append(ws)
        logger.info("WS connecté : match_id=%s (total=%d)", match_id,
                    len(self._connections[match_id]))

    def disconnect(self, ws: WebSocket, match_id: str):
        conns = self._connections.get(match_id, [])
        if ws in conns:
            conns.remove(ws)
        logger.info("WS déconnecté : match_id=%s", match_id)

    async def broadcast(self, match_id: str, data: dict):
        """Envoie un message à tous les clients connectés pour un match."""
        conns = self._connections.get(match_id, [])
        dead = []
        for ws in conns:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws, match_id)


manager = ConnectionManager()


@app.websocket("/ws/analysis/{match_id}")
async def websocket_analysis(websocket: WebSocket, match_id: str):
    """
    WebSocket temps réel.
    Envoie un AnalysisSnapshot JSON toutes les N frames.
    Le client peut envoyer {"command": "stop"} pour interrompre.
    """
    await manager.connect(websocket, match_id)

    if match_id not in _match_metadata:
        await websocket.send_json({"error": f"Match {match_id} introuvable"})
        await websocket.close()
        return

    meta = _match_metadata[match_id]
    pipeline = RealtimePipeline(enable_heatmap=True, enable_patterns=True)
    _pipelines[match_id] = pipeline

    broadcast_interval = CFG["websocket"]["broadcast_interval"]

    try:
        await websocket.send_json({"status": "started", "match_id": match_id})

        async for snapshot in pipeline.stream(meta["path"], broadcast_interval):
            # Vérifier si le client a envoyé une commande
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                cmd = json.loads(msg)
                if cmd.get("command") == "stop":
                    pipeline.stop()
                    break
            except (asyncio.TimeoutError, json.JSONDecodeError):
                pass

            await manager.broadcast(match_id, snapshot.to_dict())

        await websocket.send_json({"status": "completed", "match_id": match_id})

    except WebSocketDisconnect:
        logger.info("Client déconnecté pendant l'analyse : %s", match_id)
        pipeline.stop()
    finally:
        manager.disconnect(websocket, match_id)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_pipelines": len(_pipelines),
        "active_connections": sum(len(v) for v in manager._connections.values()),
    }
