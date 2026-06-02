# Changelog

Tous les changements notables pour le projet Rugby IA seront documentés dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
et ce projet suit [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CI/CD Pipeline avec GitHub Actions
- Pre-commit hooks pour code quality
- Tests unitaires étendus avec couverture
- Makefile pour commandes dev simplifiées
- Documentation CI/CD
- CONTRIBUTING.md pour les contributeurs
- Support CPU-only (YOLOv8s, résolution 640×360)

### Changed
- config.yaml optimisé pour CPU
- Réduction epochs fine-tuning : 100 → 15
- Batch size réduit pour CPU : 16 → 4

### Fixed
- NumPy CPU dispatcher initialization error
- Terminal encoding issues en Windows

## [1.0.0] - 2026-06-01

### Added
- Détection YOLOv8x avec ByteTrack
- Classification phases de jeu (CNN-LSTM)
- Génération de heatmaps
- Reconnaissance de patterns tactiques
- Détection d'événements (essais, mêlées, etc.)
- API FastAPI avec WebSocket
- Dashboard Streamlit
- Fine-tuning YOLOv8 sur Roboflow
- Download dataset depuis Roboflow
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
