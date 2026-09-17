# Changelog

Tous les changements notables pour le projet Rugby IA seront documentés dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
et ce projet suit [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Pipeline de tests validée sur environnement PyTorch CPU
- Documentation alignée sur la stack PyTorch/TorchVision
- Support de dépendances optionnelles pour téléchargement vidéo (`yt-dlp`)
- Tests de régression pour la détection

### Changed
- Migration du moteur de détection vers PyTorch/TorchVision
- Configuration du projet alignée sur un runtime Python/PyTorch compatible
- Nettoyage des références obsolètes à Ultralytics / YOLOv8 dans le cœur applicatif

### Fixed
- Import bloquant de `yt_dlp` lors des tests API sans dépendance installée
- Régression de compatibilité avec les tests async/benchmark via installation des plugins pytest requis

## [1.0.0] - 2026-06-01

### Added
- Détection d’objets via modèle PyTorch/TorchVision
- Classification des phases de jeu (CNN-LSTM)
- Génération de heatmaps
- Reconnaissance de patterns tactiques
- Détection d’événements (essais, mêlées, etc.)
- API FastAPI avec WebSocket
- Dashboard Streamlit
- Support dataset Roboflow optionnel
- MLflow experiment tracking
- Docker + docker-compose
- Configuration YAML centralisée

### Security
- Bandit security checks
- Safety dependency checks
- Pre-commit security hooks

---

## Guide de versioning

Suivre [Semantic Versioning](https://semver.org/) :

- **MAJOR** : Changements incompatibles (v1.0.0 → v2.0.0)
- **MINOR** : Nouvelles fonctionnalités rétro-compatibles (v1.0.0 → v1.1.0)
- **PATCH** : Corrections de bugs rétro-compatibles (v1.0.0 → v1.0.1)

Exemples :
- `v1.0.0-alpha.1` : Version alpha (pré-release)
- `v1.0.0-rc.1` : Release candidate (pré-release)
- `v1.0.0` : Release stable

## Processus de release

1. **Checkout release branch** :
   ```bash
   git checkout -b release/v1.1.0
   ```

2. **Mettre à jour la version** dans `pyproject.toml` :
   ```toml
   version = "1.1.0"
   ```

3. **Mettre à jour CHANGELOG.md** :
   ```markdown
   ## [1.1.0] - 2026-06-15
   
   ### Added
   - Feature 1
   - Feature 2
   
   ### Fixed
   - Bug fix 1
   ```

4. **Commit et push** :
   ```bash
   git commit -am "chore: bump version to v1.1.0"
   git push origin release/v1.1.0
   ```

5. **Créer une PR** vers `main` avec le template "Release"

6. **Review et merge** après approbation

7. **Créer un tag** (sur `main` après merge) :
   ```bash
   git tag -a v1.1.0 -m "Release v1.1.0"
   git push origin v1.1.0
   ```

8. **Créer la release GitHub** :
   - Allez sur GitHub Releases
   - "Draft a new release"
   - Tag : v1.1.0
   - Title : v1.1.0
   - Description : Copier depuis CHANGELOG.md

---

## Notes de version passées

### v1.0.0 - 2026-06-01

**Résumé** : Release initiale stable avec all-in-one rugby analysis.

**Highlights** :
- YOLOv8x multi-class detection (players, ball, referee)
- Real-time ByteTrack multi-object tracking
- CNN-LSTM phase classification (9 rugby phases)
- Heatmaps et pattern recognition
- Full FastAPI + Streamlit stack
- Docker support
- MLflow tracking

**Known limitations** :
- Requires GPU for real-time inference (>25 FPS)
- Phase classifier requires 30-frame context (1.2s @ 25fps)
- Patterns detection needs >5 frames history

**Next priorities for v1.1.0** :
- CPU-optimized models (YOLOv8s, 640px)
- Improved phase classifier accuracy
- OCR for jersey number recognition
- Homography for automatic field calibration
