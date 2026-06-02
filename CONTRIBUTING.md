# Contributing to Rugby IA

Merci pour votre intérêt à contribuer à Rugby IA ! Ce document fournit des guidelines pour contribuer au projet.

## Code de conduite

Nous nous engageons à maintenir un environnement respectueux et inclusif pour tous les contributeurs.

## Comment contribuer

### 1. Fork et Clone

```bash
# Fork le repo sur GitHub
# Clone votre fork
git clone https://github.com/votre-username/Rugby_IA.git
cd Rugby_IA
```

### 2. Créer une branche

```bash
# À partir de `develop`
git checkout develop
git pull origin develop
git checkout -b feature/votre-feature
```

**Conventions de nommage** :
- `feature/description` — nouvelle fonctionnalité
- `fix/description` — correction de bug
- `docs/description` — documentation
- `test/description` — tests
- `refactor/description` — refactorisation

### 3. Installation du développement

```bash
# Créer un venv
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate (Windows)

# Installer les dépendances
make install-dev

# Setup CI/CD local (pre-commit hooks)
python scripts/setup_cicd.py
```

### 4. Développement

```bash
# Coder votre feature
# Ajouter des tests
# Vérifier les tests localement
make test

# Linter et formater
make lint
make format

# ou utiliser les hooks pre-commit automatiquement sur chaque commit
```

### 5. Commit et Push

```bash
# Les hooks pré-commit vont s'exécuter automatiquement
git add .
git commit -m "feat: description courte de la feature"

# Push
git push origin feature/votre-feature
```

**Convention de commits** (Conventional Commits) :
```
<type>(<scope>): <subject>

<body>

<footer>
```

Types autorisés :
- `feat` : nouvelle fonctionnalité
- `fix` : correction de bug
- `docs` : changements de documentation
- `style` : formatage, sans changer la logique
- `refactor` : refactorisation
- `perf` : amélioration de performance
- `test` : ajout/modification de tests
- `ci` : changements CI/CD
- `chore` : maintenance, dépendances

Exemples :
```
feat(detection): add multi-class support to PlayerDetector
fix(api): handle empty frame gracefully
docs(readme): update installation instructions
test(benchmark): add performance tests for inference
```

### 6. Pull Request

1. Allez sur GitHub et créez une Pull Request vers `develop`
2. Remplissez le template PR :
   - Description claire du changement
   - Lien vers l'issue associée (si applicable)
   - Screenshots (pour UI changes)
   - Checklist

3. Attendre la review et les commentaires des mainteneurs

### 7. Code Review

Les reviewers vérifieront :
- ✓ La qualité du code
- ✓ Les tests unitaires (couverture > 70%)
- ✓ La documentation
- ✓ La compatibilité CI/CD

### 8. Merge

Une fois approuvé et tous les checks passés :
- Squash merges sont préférés pour garder `main` clean
- Votre branche sera supprimée après merge

## Directives de qualité de code

### Tests

- Écrire des tests pour toute nouvelle fonctionnalité
- Couverture minimale : 70%
- Couverture modules critiques (détection) : 85%

```bash
# Vérifier la couverture
pytest tests/ --cov=src --cov-report=html
```

### Code Style

- Utiliser **Black** pour le formatage (120 char max)
- Utiliser **isort** pour les imports
- Suivre PEP8 (Flake8)

```bash
# Format automatique
make format

# Vérifier seulement
make format-check
```

### Documentation

- Docstrings pour tous les modules/classes/fonctions
- Format Google-style docstrings

```python
def my_function(param1: str, param2: int) -> bool:
    """Short description of what the function does.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something is invalid
    """
    pass
```

### Type Hints

Utiliser les type hints autant que possible :

```python
from typing import Optional, List, Dict

def process_frame(frame: np.ndarray, config: Dict) -> Optional[List[str]]:
    """Process a video frame."""
    pass
```

### Logging

Utiliser le logging au lieu de print() :

```python
import logging

logger = logging.getLogger(__name__)

logger.debug("Detailed info for debugging")
logger.info("General informational message")
logger.warning("Warning about something")
logger.error("An error occurred")
```

## Processus de release

1. Bump version dans `pyproject.toml`
2. Mettre à jour `CHANGELOG.md`
3. Créer un tag Git : `git tag v1.0.0`
4. Push le tag : `git push origin v1.0.0`
5. La pipeline CI/CD crée automatiquement la release

## Reporting Issues

Utiliser le template issue :

1. **Title** : Clair et concis
2. **Description** : Contexte et détails
3. **Steps to reproduce** : Étapes exactes
4. **Expected behavior** : Ce qui devrait arriver
5. **Actual behavior** : Ce qui se passe réellement
6. **Environment** : OS, Python version, etc.

## Questions ?

- Ouvrir une Issue pour les questions générales
- Utiliser les Discussions pour les suggestions

## License

En contribuant, vous acceptez que votre code soit sous licence MIT.

---

**Merci de contribuer !** 🏉
