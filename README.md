# 🏉 Rugby IA — Analyse Vidéo Temps Réel par Computer Vision

**Projet personnel** · [github.com/Fouxy84/Rugby_IA](https://github.com/Fouxy84/Rugby_IA) · 2026

[![CI/CD](https://github.com/Fouxy84/Rugby_IA/actions/workflows/ci.yml/badge.svg)](https://github.com/Fouxy84/Rugby_IA/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Fouxy84/Rugby_IA/branch/main/graph/badge.svg)](https://codecov.io/gh/Fouxy84/Rugby_IA)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Système d'analyse IA temps réel de matchs de rugby : détection des joueurs, classification des phases de jeu, heatmaps, reconnaissance de patterns tactiques et key insights automatiques.

**Stack principale :** YOLOv8 (Small pour CPU) · ByteTrack · CNN-LSTM · MLflow · Docker · FastAPI · Streamlit

---

## 🚀 Quick Start (30 secondes)

```bash
# 1. Clone & setup
git clone https://github.com/Fouxy84/Rugby_IA.git
cd Rugby_IA
python -m venv .venv
.venv\Scripts\activate  # ou source .venv/bin/activate (Linux/Mac)

# 2. Install & configure
pip install -r requirements.txt
cp .env.example .env

# 3. Run
make dev  # Démarrer API + Dashboard
# Ouvrir: http://localhost:8501
```

[→ Installation complète](## 🚀 Installation)

---

## 🏆 Résultats clés (CPU-Optimized)

| Métrique | Valeur | Conditions |
|---|---|---|
| **Modèle détection** | **YOLOv8s** | Petit modèle pour CPU (1.6M params vs 68M YOLOv8x) |
| **Résolution** | **640×360** | Réduit de 1280×720 pour performance CPU |
| **Batch size** | **4** | Réduit de 16 pour RAM limitée |
| **FPS CPU** | **~8-12 FPS** | Estimation (1280×720 → 640×360) |
| **RAM utilisée** | **~2-3 GB** | Vs 10-12 GB en GPU |
| **Fine-tuning** | **15 epochs** | Vs 100 en version GPU |

Augmentations adaptées au rugby : occlusions entre joueurs (`erasing=0.4`), variations d'éclairage de stade (`hsv_v=0.4`), flips horizontaux (`fliplr=0.5`).

```bash
# Reproduire le fine-tuning (CPU, ~30-45 min)
python scripts/finetune_yolo_rugby.py --data data/roboflow/merged/data.yaml --epochs 15 --device cpu --batch 4
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

### Mode développement (sans Docker) — Makefile

Utiliser les **commandes Makefile** pour simplifier les opérations courantes :

```bash
# --- Setup initial ---
make install-dev       # Installer dépendances + dev tools

# --- Développement ---
make dev              # Démarrer API FastAPI + Dashboard Streamlit
make api              # API seule (port 8000)
make dashboard        # Dashboard seul (port 8501)
make mlflow           # MLflow UI (port 5000)

# --- Tests ---
make test             # Tous les tests
make test-cov         # Tests avec couverture HTML
make test-fast        # Tests rapides (exclure les lents)

# --- Code Quality ---
make lint             # Vérifier linting + formatage
make format           # Formater le code automatiquement
make format-check     # Vérifier sans modifier

# --- Données & Entraînement ---
make download-data    # Télécharger dataset Roboflow
make train            # Fine-tune YOLOv8 (15 epochs CPU)
make train-quick      # Test rapide (3 epochs)

# --- Cleanup ---
make clean            # Supprimer fichiers temp
make clean-all        # Supprimer modèles + runs aussi
```

Voir [Makefile](Makefile) pour toutes les commandes.

### Mode traditionnel

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

### Exécution locale

```bash
# Tous les tests
pytest tests/ -v

# Tests avec couverture
pytest tests/ -v --cov=src --cov-report=html

# Tests spécifiques
pytest tests/test_detection.py -v
pytest tests/test_api.py -v
```

Via Makefile (recommandé) :
```bash
# Tous les tests
make test

# Tests avec couverture
make test-cov

# Tests rapides (excluant les tests lents)
make test-fast
```

---

## 🔄 CI/CD Pipeline

### Déclenchement automatique

La pipeline s'exécute sur :
- **Push** vers `main` et `develop`
- **Pull Requests** vers `main` et `develop`

### Étapes (GitHub Actions)

| Étape | Description |
|---|---|
| **Test** | pytest sur Ubuntu/Windows, Python 3.10-3.12 |
| **Linting** | Flake8, Pylint, Black, isort |
| **Coverage** | Rapport HTML + upload Codecov |
| **Docker** | Build image (si `main`) |
| **Code Quality** | Cyclomatic complexity, maintainability |
| **Security** | Bandit, Safety checks |
| **Performance** | Benchmarks (si `main`) |

Voir [docs/CI-CD.md](docs/CI-CD.md) pour plus de détails.

### Code Quality Local

```bash
# Vérifier le linting
make lint

# Formater le code
make format

# Vérifier le formatage
make format-check

# Linter seulement (non-blocking)
pylint src --disable=all --enable=E,F --exit-zero
```

### Setup des hooks pré-commit

```bash
# Installation automatique
python scripts/setup_cicd.py

# Ou manuel
pre-commit install
pre-commit run --all-files
```

Les hooks pré-commit vérifieront avant chaque commit :
- Formatage (Black, isort)
- Linting (Flake8, Pylint)
- Sécurité (Bandit)
- Trailing whitespace, conflits de merge, etc.

---

---

## 📡 Sources de données rugby publiques

| Source | Description | Accès |
|---|---|---|
| **YouTube** | Highlights, matchs complets | yt-dlp (gratuit) |
| **ESPN Rugby** | Scores, résultats, calendrier | API publique |
| **World Rugby** | Stats officielles | Clé API (`.env`) |
| **Dailymotion** | Résumés Top14, Premiership | yt-dlp (gratuit) |

---

## 🏋️ Fine-tuning YOLOv8 (CPU-Optimized)

### 1. Obtenir une clé API Roboflow

1. Connectez-vous sur [app.roboflow.com/adams-workspace-ppons](https://app.roboflow.com/adams-workspace-ppons)
2. **Settings → API** → copier la clé
3. L'ajouter dans `.env` :
```
ROBOFLOW_API_KEY=votre_cle_roboflow_ici
```

### 2. Télécharger le dataset

```bash
# Via Makefile
make download-data

# Ou manuellement
python scripts/download_roboflow_dataset.py --api-key <YOUR_API_KEY>

# Tous les datasets rugby
python scripts/download_roboflow_dataset.py --api-key <KEY> --all

# Dataset spécifique
python scripts/download_roboflow_dataset.py \
  --api-key <KEY> \
  --workspace rugby-analysis \
  --project rugby-player-detection \
  --version 5
```

### 3. Lancer l'entraînement

```bash
# Via Makefile (recommandé)
make train            # 15 epochs sur CPU (~30-45 min)
make train-quick      # 3 epochs test (~10-15 min)

# Manuellement
python scripts/finetune_yolo_rugby.py \
  --data data/roboflow/merged/data.yaml \
  --epochs 15 \
  --device cpu \
  --batch 4
```

Le script :
- Utilise **YOLOv8s** (petit modèle : 1.6M params)
- Résolution **640×360** (réduit CPU)
- Batch size **4** (RAM limitée)
- Applique **augmentations rugby** (occlusions, éclairages, flips)
- Log dans **MLflow** → `http://localhost:5000`
- Exporte en **ONNX** + **TorchScript**
- Copie le meilleur modèle → `data/models/rugby_detector.pt`

### 4. Évaluation

```bash
# Valider sur set de validation
python scripts/finetune_yolo_rugby.py \
  --val-only \
  --weights data/models/rugby_detector.pt \
  --data data/roboflow/merged/data.yaml

# Export ONNX seulement
python scripts/finetune_yolo_rugby.py \
  --export-only \
  --weights data/models/rugby_detector.pt
```

### Optimisations CPU

| Paramètre | GPU (yolov8x) | CPU (yolov8s) |
|---|---|---|
| **Modèle** | 68M params | 1.6M params |
| **Résolution** | 1280×720 | 640×360 |
| **Batch** | 16 | 4 |
| **Epochs** | 100 | 15 |
| **AMP** | Activé | Désactivé |
| **RAM** | 10-12 GB | 2-3 GB |
| **Durée (15 epochs)** | ~3 heures | ~45 min |
| **FPS inférence** | ~35 FPS | ~8-12 FPS |

---

---

## � Documentation

| Document | Description |
|---|---|
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | Guide pour contribuer (fork, PR, conventions) |
| **[docs/CI-CD.md](docs/CI-CD.md)** | Pipeline GitHub Actions, tests, pre-commit |
| **[CHANGELOG.md](CHANGELOG.md)** | Versioning, release notes, historique |
| **[Makefile](Makefile)** | Toutes les commandes dev disponibles |
| **[requirements.txt](requirements.txt)** | Dépendances Python |
| **[config/config.yaml](config/config.yaml)** | Configuration centralisée |

---

## 🤝 Contribuer

Les contributions sont bienvenues ! Voir [CONTRIBUTING.md](CONTRIBUTING.md) pour :
- Workflow de contribution (fork → branch → PR)
- Conventions de commit (Conventional Commits)
- Standards de code (Black, Pylint, tests)
- Instructions de setup développement

### Quick contribution flow

```bash
# 1. Fork et clone
git clone https://github.com/votre-username/Rugby_IA.git
cd Rugby_IA

# 2. Setup dev
python -m venv .venv
.venv\Scripts\activate
make install-dev
python scripts/setup_cicd.py

# 3. Créer feature branch
git checkout -b feature/votre-feature

# 4. Développer avec tests
make test
make lint
make format

# 5. Push et créer PR
git push origin feature/votre-feature
```

---

## �🔮 Roadmap

### ✅ Complété (v1.0)
- [x] Détection YOLOv8 multi-classe (joueurs, ballon, arbitre)
- [x] Tracking temps réel ByteTrack
- [x] Classification phases de jeu (CNN-LSTM)
- [x] Heatmaps et patterns tactiques
- [x] API FastAPI + WebSocket
- [x] Dashboard Streamlit
- [x] Fine-tuning sur Roboflow
- [x] MLflow experiment tracking
- [x] Docker + docker-compose
- [x] Benchmark FPS

### 🚀 En cours (v1.1 — CPU-Optimized)
- [x] Support CPU-only (YOLOv8s)
- [x] Réduction résolution (640×360)
- [x] CI/CD Pipeline GitHub Actions
- [x] Tests unitaires extensifs (70%+ coverage)
- [x] Pre-commit hooks
- [x] Makefile pour dev
- [x] Contributing guidelines
- [ ] Documentation API (Swagger complet)
- [ ] Performance monitoring

### 📋 Futures améliorations (v1.2+)
- [ ] Reconnaissance numéro de maillot (OCR)
- [ ] Homographie pour calibration terrain automatique
- [ ] Export clips automatique des événements
- [ ] Analyse comparative multi-matchs
- [ ] Mode RTSP pour flux caméra en direct
- [ ] Intégration SportsCode / Hudl
- [ ] Web dashboard (React/Vue)
- [ ] Mobile app
- [ ] Modèles phase classifier améliorés
- [ ] Détection plus fine (carton rouge, talonnade, etc.)

## ❓ FAQ & Troubleshooting

### Installation

**Q: Erreur `pip: command not found`**
> Utilisez `python -m pip` à la place de `pip`

**Q: Erreur `numpy._core._multiarray_umath` / CPU dispatcher**
> Mettez à jour numpy : `pip install --upgrade numpy`

**Q: Erreur CUDA / GPU not found**
> Normal ! Le projet fonctionne en CPU-only. Utiliser `--device cpu`

### Utilisation

**Q: Dashboard lent / timeout**
> Réduire `video.batch_size` dans `config.yaml` (essayer 2-4 au lieu de 8)

**Q: Training très lent sur CPU**
> C'est normal ! YOLOv8s prend ~45 min pour 15 epochs sur CPU. Réduire epochs avec `--epochs 3` pour tester.

**Q: Où sont mes modèles ?**
> `data/models/rugby_detector.pt` après training (copié depuis les runs YOLOv8)

**Q: Comment utiliser mon propre dataset ?**
> Créer `data.yaml` avec le format YOLOv8 et passer `--data chemin/vers/data.yaml`

### Development

**Q: Tests échouent localement**
> Exécuter `make test-fast` (exclut les tests lents/GPU)

**Q: Comment contribuer ?**
> Voir [CONTRIBUTING.md](CONTRIBUTING.md) pour le workflow complet

**Q: Où reporter un bug ?**
> [Ouvrir une issue sur GitHub](https://github.com/Fouxy84/Rugby_IA/issues)

---

## 📞 Support

- **Issues** : [GitHub Issues](https://github.com/Fouxy84/Rugby_IA/issues)
- **Discussions** : [GitHub Discussions](https://github.com/Fouxy84/Rugby_IA/discussions)
- **Email** : [contact@rugby-ia.com](mailto:contact@rugby-ia.com)
- **Documentation** : [Voir docs/](docs/)

---

- **Roboflow** : Datasets et labeling tools
- **Ultralytics** : YOLOv8 framework
- **FastAPI** : API framework
- **Streamlit** : Dashboard
- **MLflow** : Experiment tracking
- **Community** : Contributions et feedback

---

## 📄 License

MIT — Projet DataScientest MLOps

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.

---

**Made with ❤️ by [Fouxy84](https://github.com/Fouxy84)**
