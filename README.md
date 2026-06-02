# 🏉 Rugby IA — Analyse Vidéo Temps Réel par Computer Vision

**Projet personnel** · [github.com/Fouxy84/Rugby_IA](https://github.com/Fouxy84/Rugby_IA) · 2026

> Système d'analyse IA temps réel de matchs de rugby : détection des joueurs, classification des phases de jeu, heatmaps, reconnaissance de patterns tactiques et key insights automatiques.

**Stack principale :** YOLOv8 · ByteTrack · CNN-LSTM · MLflow · Docker · FastAPI · Streamlit

---

## 🏆 Résultats clés

| Métrique | Valeur | Conditions |
|---|---|---|
| **mAP@0.5 joueurs** | **≈ 0,88** | YOLOv8x fine-tuné, datasets Roboflow Top14/Premiership |
| **mAP@0.5 ballon** | **≈ 0,79** | YOLOv8x fine-tuné |
| **FPS inférence** | **~35 FPS** | Détection seule, 1280×720, RTX 3080 |
| **FPS pipeline complet** | **~30 FPS** | Détection + ByteTrack + classification de phase |

Augmentations adaptées au rugby : occlusions entre joueurs (`erasing=0.4`), variations d'éclairage de stade (`hsv_v=0.4`), flips horizontaux (`fliplr=0.5`).

```bash
# Reproduire le benchmark FPS
python scripts/benchmark.py --device 0 --pipeline

# Reproduire le fine-tuning (après téléchargement du dataset Roboflow)
python scripts/finetune_yolo_rugby.py --data data/roboflow/merged/data.yaml --epochs 100
```

---

## 🏗️ Architecture

```
Rugby_IA/
├── config/                    # Configuration YAML
├── src/
│   ├── ingestion/             # Téléchargement & lecture vidéo
│   │   └── video_downloader.py  (VideoDownloader, VideoReader, RugbyDataConnector)
│   ├── detection/             # YOLOv8 + ByteTrack
│   │   └── player_detector.py   (PlayerDetector, PlayerTracker, TeamClassifier)
│   ├── analysis/              # Modules d'analyse
│   │   ├── phase_classifier.py  (CNN-LSTM, 9 phases de jeu)
│   │   ├── event_detector.py    (essais, mêlées, touches, rucks…)
│   │   ├── heatmap_generator.py (cartes de chaleur par équipe)
│   │   └── pattern_recognizer.py (switch, linebreak, pick & go…)
│   ├── pipeline/              # Orchestration temps réel
│   │   └── realtime_pipeline.py
│   └── api/                   # FastAPI + WebSocket
│       └── main.py
├── dashboard/                 # Interface Streamlit
│   └── app.py
├── scripts/                   # CLI utilitaires
│   ├── run_analysis.py        # Analyse en ligne de commande
│   └── train_model.py         # Entraînement du phase classifier
├── tests/                     # Tests unitaires
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## ⚙️ Fonctionnalités

| Fonctionnalité | Détails |
|---|---|
| **Détection joueurs** | YOLOv8x (fine-tunable) + ByteTrack multi-objet |
| **Classification équipes** | K-Means sur couleur de maillot |
| **Phases de jeu** | CNN-LSTM : mêlée, touche, essai, ruck, maul, coup de pied, jeu courant |
| **Événements** | Détection automatique : essais, mêlées, touches, pénalités |
| **Heatmaps** | Présence par équipe, trajectoire du ballon, occupation par zone |
| **Patterns tactiques** | Pick & go, switch, linebreak, défense rideau, maul drive |
| **Real-time tagging** | Événements horodatés exportables JSON |
| **Key Insights** | 6 insights automatiques par frame |
| **Sources vidéo** | Upload local, YouTube, Dailymotion, Vimeo (yt-dlp) |
| **API REST + WS** | FastAPI + WebSocket pour streaming temps réel |
| **Dashboard** | Streamlit multi-pages |
| **MLOps** | MLflow pour le suivi des expériences |

---

## 🚀 Installation

### 1. Environnement Python

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Variables d'environnement

```bash
cp .env.example .env
# Éditer .env si nécessaire (clé API World Rugby, GPU…)
```

### 3. Créer les répertoires

```bash
# Linux / Mac
mkdir -p data/raw data/processed data/models data/outputs/heatmaps logs mlruns
```

```powershell
# Windows PowerShell
"data/raw","data/processed","data/models","data/outputs/heatmaps","logs","mlruns" | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ }
```

---

## ▶️ Démarrage

### Mode développement (sans Docker)

**Terminal 1 — API FastAPI :**
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Dashboard Streamlit :**
```bash
streamlit run dashboard/app.py --server.port 8501
```

**Terminal 3 — MLflow (optionnel) :**
```bash
mlflow ui --port 5000
```

Ouvrir : http://localhost:8501

---

### Mode Docker Compose

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Dashboard | http://localhost:8501 |
| API docs | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |

---

## 🎮 Utilisation

### Via le Dashboard (recommandé)

1. **📥 Sources** → Importer un match (upload ou URL YouTube)
2. **▶️ Analyse temps réel** → Lancer l'analyse et voir les insights en direct
3. **🗺️ Heatmaps** → Visualiser l'utilisation de l'espace par équipe
4. **⚡ Événements** → Timeline des événements clés
5. **🧩 Patterns** → Patterns tactiques reconnus
6. **📋 Statistiques** → Tableau de bord global du match

### Via la ligne de commande

```bash
# Analyser un fichier local
python scripts/run_analysis.py --video data/raw/match.mp4

# Télécharger et analyser depuis YouTube
python scripts/run_analysis.py --url https://www.youtube.com/watch?v=XXXXX

# Sans heatmaps (plus rapide)
python scripts/run_analysis.py --video match.mp4 --no-heatmap
```

### Via l'API REST

```bash
# Upload d'une vidéo
curl -X POST http://localhost:8000/api/video/upload \
  -F "file=@match.mp4" -F "match_name=Top14 Toulouse"

# Rechercher un match
curl "http://localhost:8000/api/video/search?q=Top14+2024+highlights"

# Matchs récents Top14
curl http://localhost:8000/api/leagues/top14

# Heatmap PNG
curl "http://localhost:8000/api/matches/{id}/heatmap?mode=home" --output heatmap.png
```

---

## 🏋️ Entraînement du modèle

### Préparer les données

```
data/annotations/
├── melee/
│   ├── clip_001/  (frames JPEG numérotées)
│   ├── clip_002/
│   └── ...
├── touche/
├── essai/
├── ruck/
└── ...
```

### Lancer l'entraînement

```bash
python scripts/train_model.py \
  --data-dir data/annotations \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4
```

Le meilleur modèle est sauvegardé dans `data/models/phase_classifier.pt`.  
Les métriques sont suivies dans MLflow.

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 📡 Sources de données rugby publiques

| Source | Description | Accès |
|---|---|---|
| **YouTube** | Highlights, matchs complets | yt-dlp (gratuit) |
| **ESPN Rugby** | Scores, résultats, calendrier | API publique |
| **World Rugby** | Stats officielles | Clé API (`.env`) |
| **Dailymotion** | Résumés Top14, Premiership | yt-dlp (gratuit) |

---

## 🏋️ Fine-tuning YOLOv8 sur Roboflow Rugby

### 1. Obtenir une clé API Roboflow

1. Connectez-vous sur [app.roboflow.com/adams-workspace-ppons](https://app.roboflow.com/adams-workspace-ppons)
2. **Settings → API** → copier la clé
3. L'ajouter dans `.env` :
```
ROBOFLOW_API_KEY=votre_cle_roboflow_ici
```

### 2. Explorer les datasets rugby disponibles

```bash
# Listing local + recherche live Universe
python scripts/download_roboflow_dataset.py --list
python scripts/download_roboflow_dataset.py --api-key <KEY> --list
```

| # | Workspace | Projet | v | Classes |
|---|---|---|---|---|
| 1 | `roboflow-100` | `rugby-players-2` | 2 | player, referee, ball |
| 2 | `roboflow-100` | `rugby-detection` | 1 | player, ball |
| 3 | `rugby-analysis` | `rugby-player-detection` | 5 | player, referee, ball |
| 4 | `sports-detection` | `rugby-ball-detection` | 1 | ball |

Exploration web : [universe.roboflow.com/search?q=rugby+detection](https://universe.roboflow.com/search?q=rugby+detection)  
Votre workspace : [app.roboflow.com/adams-workspace-ppons](https://app.roboflow.com/adams-workspace-ppons)

### 3. Télécharger le dataset

```bash
# Téléchargement par défaut (2 premiers datasets, fallback automatique)
python scripts/download_roboflow_dataset.py --api-key <KEY>

# Dataset spécifique
python scripts/download_roboflow_dataset.py \
  --api-key <KEY> \
  --workspace rugby-analysis \
  --project rugby-player-detection \
  --version 5

# Tous les datasets rugby connus (fusion automatique)
python scripts/download_roboflow_dataset.py --api-key <KEY> --all

# Recherche live sur Roboflow Universe
python scripts/download_roboflow_dataset.py --api-key <KEY> --search "rugby detection"
```

Le script :
- Télécharge au format **YOLOv8**
- **Remappe automatiquement les classes** (toutes les variantes → `player=0 / referee=1 / ball=2`)
- Fusionne plusieurs datasets en un seul `data.yaml`
- Génère un rapport de statistiques

### 4. Fine-tuner YOLOv8x

```bash
# Entraînement complet (GPU recommandé)
python scripts/finetune_yolo_rugby.py \
  --data data/roboflow/merged/data.yaml \
  --epochs 100 \
  --imgsz 1280 \
  --batch 16

# Sur CPU (plus lent)
python scripts/finetune_yolo_rugby.py \
  --data data/roboflow/merged/data.yaml \
  --device cpu --batch 4 --epochs 30

# Reprendre un entraînement interrompu
python scripts/finetune_yolo_rugby.py \
  --data data.yaml \
  --resume runs/detect/rugby_v1/weights/last.pt
```

Le script :
- Part de `yolov8x.pt` (pré-entraîné ImageNet/COCO)
- Applique les **augmentations rugby** (occlusions, éclairages de stade, flips)
- Log toutes les métriques dans **MLflow** (`http://localhost:5000`)
- Copie automatiquement le meilleur modèle → `data/models/rugby_detector.pt`
- Exporte en **ONNX** + **TorchScript** pour déploiement

### 5. Validation et export seuls

```bash
# Évaluer un modèle existant
python scripts/finetune_yolo_rugby.py \
  --val-only \
  --weights data/models/rugby_detector.pt \
  --data data/roboflow/merged/data.yaml

# Export ONNX uniquement
python scripts/finetune_yolo_rugby.py \
  --export-only \
  --weights data/models/rugby_detector.pt
```

### Résultats obtenus

| Métrique | Objectif | Obtenu (Top14 / Premiership) |
|---|---|---|
| **mAP@0.5 joueurs** | > 0.85 | **≈ 0,88** |
| **mAP@0.5 ballon** | > 0.75 | **≈ 0,79** |
| **FPS inférence (GPU)** | > 25 | **~35 sur RTX 3080** |

---

## 🔮 Roadmap

- [x] Fine-tuning YOLOv8x sur dataset rugby annoté (Roboflow)
- [x] Export ONNX + TorchScript pour déploiement
- [x] Benchmark FPS (`scripts/benchmark.py`)
- [ ] Reconnaissance du numéro de maillot (OCR)
- [ ] Homographie automatique pour calibration terrain
- [ ] Export clip automatique des événements clés
- [ ] Analyse comparative multi-matchs
- [ ] Mode RTSP pour flux caméra en direct
- [ ] Intégration SportsCode / Hudl

---

## 📄 Licence

MIT — Projet DataScientest MLOps
