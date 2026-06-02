# CI/CD Pipeline — Rugby IA

Documentation de la pipeline de Continuous Integration / Continuous Deployment.

## 📋 Flux CI/CD

### 1. **Déclenchement automatique**

La pipeline s'exécute automatiquement sur :
- **Pushs** vers `main` et `develop`
- **Pull Requests** vers `main` et `develop`

### 2. **Étapes de la pipeline**

#### **Test (Multi-plateforme)**
- **OS** : Ubuntu, Windows
- **Python** : 3.10, 3.11, 3.12
- **Tests** : Coverage, pytest
- **Artefacts** : Rapports HTML, rapports XML Codecov

```bash
# Exécution locale
pytest tests/ -v --cov=src --cov-report=html
```

#### **Linting & Formatting**
- **Flake8** : Vérification PEP8
- **Pylint** : Analyse statique
- **Black** : Vérification de formatage
- **isort** : Vérification d'imports

```bash
# Exécution locale
make lint
make format-check
```

#### **Code Quality**
- **Cyclomatic Complexity** : via Radon
- **Maintainability Index** : via Radon
- **Sécurité** : Bandit, Safety

```bash
# Exécution locale
radon cc src -a
radon mi src
bandit -r src
```

#### **Docker Build** (si `main`)
- Construit l'image Docker
- Teste la présence des tests unitaires

```bash
docker build -t rugby-ia:latest .
docker run --rm rugby-ia:latest pytest tests/ --co
```

#### **Benchmarks** (si `main`)
- Teste les performances
- Génère des métriques

```bash
pytest tests/test_benchmark.py -v --benchmark-only
```

### 3. **Artifacts générés**

| Artifact | Description | Emplacement |
|---|---|---|
| **Coverage Report** | Rapport de couverture (HTML) | Onglet GitHub Actions |
| **Test Results** | Résultats des tests par OS/Python | Onglet GitHub Actions |
| **Codecov Report** | Intégration Codecov | codecov.io |
| **Linting Report** | Rapport Pylint/Flake8 | Logs GitHub Actions |
| **Docker Image** | Image Docker (main seulement) | ghcr.io (optionnel) |

## 🚀 Exécution locale

### Installation des dépendances dev

```bash
make install-dev
```

### Exécuter les tests

```bash
# Tous les tests
make test

# Tests unitaires seulement
make test-unit

# Tests avec couverture
make test-cov

# Tests rapides (excluant les tests lents)
make test-fast
```

### Linting & Formatage

```bash
# Vérifier le linting
make lint

# Formater le code
make format

# Vérifier le formatage sans modifier
make format-check
```

### Nettoyage

```bash
# Nettoyer les fichiers temporaires
make clean

# Nettoyer tout (modèles, runs, etc.)
make clean-all
```

## 🔒 Sécurité

### Checks de sécurité automatiques

- **Bandit** : Identifie les problèmes de sécurité Python
- **Safety** : Vérifie les vulnérabilités dans les dépendances
- **Secrets Detection** : Détecte les secrets (clés API, tokens)

### Bonnes pratiques

1. **Jamais committer** d'API keys ou tokens
   - Utiliser `.env.example` pour les templates
   - Utiliser GitHub Secrets pour les variables confidentielles

2. **Dépendances** : Garder les dépendances à jour
   ```bash
   pip list --outdated
   ```

3. **Code review** : Toutes les PR doivent être revues avant merge

## 📊 Métriques

### Coverage Minimum

- **Global** : 70%
- **Modules critiques** : 85% (détection, tracking)

### Code Quality

- **Cyclomatic Complexity** : < 10
- **Maintainability Index** : > 60
- **Pylint Score** : > 7.0

## 🐛 Troubleshooting

### Les tests échouent localement mais pas en CI

**Cause** : Différences d'environnement

**Solution** :
```bash
# Reproduire l'env CI
python -m venv test_env
source test_env/bin/activate  # ou test_env\Scripts\activate (Windows)
pip install -r requirements.txt
pip install pytest pytest-cov pylint black flake8
pytest tests/
```

### Timeout sur les tests GPU

**Cause** : Tests GPU s'exécutent sans GPU

**Solution** : Sauter les tests GPU
```bash
pytest tests/ -m "not gpu"
```

### Failing Codecov

**Cause** : Impossible de se connecter à codecov.io

**Solution** : Ce n'est pas critique, la pipeline continue

## 📝 Ajout de nouveaux tests

### Structure recommandée

```python
"""Tests pour mon_module."""

import pytest

@pytest.mark.smoke
def test_basic_functionality():
    """Test basique."""
    assert True

@pytest.mark.slow
def test_slow_operation():
    """Test qui prend du temps."""
    pass

@pytest.mark.gpu
@pytest.mark.skip(reason="GPU not available")
def test_gpu_operation():
    """Test GPU."""
    pass
```

### Markers disponibles

| Marker | Utilisation |
|---|---|
| `@pytest.mark.smoke` | Tests rapides et critiques |
| `@pytest.mark.slow` | Tests qui prennent du temps |
| `@pytest.mark.gpu` | Tests qui nécessitent GPU |
| `@pytest.mark.integration` | Tests d'intégration |
| `@pytest.mark.benchmark` | Tests de performance |

### Exécuter des tests spécifiques

```bash
# Seulement les smoke tests
pytest tests/ -m smoke

# Exclure les tests lents
pytest tests/ -m "not slow"

# Exclure GPU
pytest tests/ -m "not gpu"
```

## 🔗 Intégrations

### Codecov

- Automatiquement synchronisé depuis la pipeline
- URL : https://codecov.io/gh/Fouxy84/Rugby_IA

### GitHub

- Protection de branche `main` : Nécessite passage des tests
- Commit status checks : Bloque les merges si tests échouent

### Status Badge

Ajouter au README :
```markdown
[![CI/CD](https://github.com/Fouxy84/Rugby_IA/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/Fouxy84/Rugby_IA/actions)
[![codecov](https://codecov.io/gh/Fouxy84/Rugby_IA/branch/main/graph/badge.svg)](https://codecov.io/gh/Fouxy84/Rugby_IA)
```

## 📖 Ressources

- [Pytest Documentation](https://docs.pytest.org/)
- [GitHub Actions](https://docs.github.com/en/actions)
- [Codecov](https://codecov.io/docs)
- [Black Code Formatter](https://black.readthedocs.io/)
- [Pylint](https://www.pylint.org/)
