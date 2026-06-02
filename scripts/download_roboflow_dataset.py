"""
Téléchargement datasets rugby depuis Roboflow Universe.

Source : https://universe.roboflow.com/search?q=rugby+detection
Workspace : https://app.roboflow.com/adams-workspace-ppons

Ce script utilise l'API REST Roboflow directement (sans SDK) pour éviter
les problèmes de téléchargement (zips invalides, liens GCS expirés, etc.).

Pipeline :
  1. Interroge l'API REST pour obtenir le lien d'export YOLOv8
  2. Attend la génération de l'export si nécessaire (polling)
  3. Télécharge le zip avec vérification d'intégrité
  4. Remappe les classes → player=0 / referee=1 / ball=2
  5. Génère data.yaml fusionné compatible YOLOv8

Usage :
    # Lister les datasets rugby disponibles sur Universe
    python scripts/download_roboflow_dataset.py --list

    # Télécharger les datasets par défaut (clé dans .env ou --api-key)
    python scripts/download_roboflow_dataset.py --api-key <KEY>

    # Télécharger un dataset précis
    python scripts/download_roboflow_dataset.py --api-key <KEY> \\
        --workspace rugby-analysis --project rugby-player-detection --version 5

    # Tous les datasets rugby connus
    python scripts/download_roboflow_dataset.py --api-key <KEY> --all

    # Recherche live sur Roboflow Universe
    python scripts/download_roboflow_dataset.py --api-key <KEY> --search "rugby detection"
"""

import argparse
import io
import json
import logging
import os
import sys
import time
import zipfile
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_roboflow")

ROBOFLOW_API = "https://api.roboflow.com"
USER_WORKSPACE = "adams-workspace-ppons"

# ---------------------------------------------------------------------------
# Datasets rugby disponibles sur Roboflow Universe
# https://universe.roboflow.com/search?q=rugby+detection
# ---------------------------------------------------------------------------
RUGBY_UNIVERSE_DATASETS = [
    # rugby-analysis/rugby-player-detection — 5 versions, accessible avec cette clé API
    # (les liens GCS peuvent être expirés, le script essaie toutes les versions)
    {
        "workspace":   "rugby-analysis",
        "project":     "rugby-player-detection",
        "version":     1,
        "description": "Joueurs Top14/Premiership v1",
        "classes":     ["player", "referee", "ball"],
        "priority":    1,
    },
    {
        "workspace":   "rugby-analysis",
        "project":     "rugby-player-detection",
        "version":     2,
        "description": "Joueurs Top14/Premiership v2",
        "classes":     ["player", "referee", "ball"],
        "priority":    2,
    },
    {
        "workspace":   "rugby-analysis",
        "project":     "rugby-player-detection",
        "version":     3,
        "description": "Joueurs Top14/Premiership v3",
        "classes":     ["player", "referee", "ball"],
        "priority":    3,
    },
    {
        "workspace":   "rugby-analysis",
        "project":     "rugby-player-detection",
        "version":     4,
        "description": "Joueurs Top14/Premiership v4",
        "classes":     ["player", "referee", "ball"],
        "priority":    4,
    },
    {
        "workspace":   "rugby-analysis",
        "project":     "rugby-player-detection",
        "version":     5,
        "description": "Joueurs Top14/Premiership v5",
        "classes":     ["player", "referee", "ball"],
        "priority":    5,
    },
    # Datasets Roboflow-100 (nécessitent un accès workspace — peuvent ne pas être disponibles)
    {
        "workspace":   "roboflow-100",
        "project":     "rugby-players-2",
        "version":     2,
        "description": "Joueurs, arbitres, ballon — Roboflow-100",
        "classes":     ["player", "referee", "ball"],
        "priority":    6,
    },
    {
        "workspace":   "roboflow-100",
        "project":     "rugby-detection",
        "version":     1,
        "description": "Joueurs + ballon — Roboflow-100",
        "classes":     ["player", "ball"],
        "priority":    7,
    },
]

# Mapping labels → indices internes Rugby IA
CLASS_REMAP = {
    "player":       0, "Player":       0, "rugby player": 0, "rugby_player": 0,
    "players":      0, "Players":      0, "person":       0, "Person":       0,
    "referee":      1, "Referee":      1, "ref":          1, "Ref":          1,
    "umpire":       1,
    "ball":         2, "Ball":         2, "rugby ball":   2, "rugby_ball":   2,
    "Rugby Ball":   2,
}
RUGBY_IA_CLASSES = {0: "player", 1: "referee", 2: "ball"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_api_key(cli_key: str | None) -> str:
    key = cli_key or os.getenv("ROBOFLOW_API_KEY", "")
    if not key:
        logger.error(
            "Clé API Roboflow manquante.\n"
            "  → Obtenez-la sur https://app.roboflow.com/adams-workspace-ppons/settings/api\n"
            "  → Passez-la via --api-key OU ajoutez ROBOFLOW_API_KEY=... dans .env"
        )
        sys.exit(1)
    return key


def _api_get(endpoint: str, api_key: str, timeout: int = 30, **params) -> dict:
    """GET vers l'API REST Roboflow avec gestion d'erreurs."""
    url = f"{ROBOFLOW_API}/{endpoint.lstrip('/')}"
    params["api_key"] = api_key
    resp = requests.get(url, params=params, timeout=timeout)
    if resp.status_code == 401:
        raise PermissionError(f"Clé API refusée par Roboflow ({endpoint})")
    if resp.status_code == 404:
        raise FileNotFoundError(f"Ressource introuvable : {endpoint}")
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Recherche sur Roboflow Universe
# ---------------------------------------------------------------------------

def search_universe(api_key: str, query: str, max_results: int = 20) -> list[dict]:
    """
    Recherche des projets sur Roboflow Universe.
    GET https://api.roboflow.com/search?q=<query>&type=project
    """
    try:
        data = _api_get("search", api_key, q=query, type="project", n=max_results)
        results = data.get("results", [])
        logger.info("Roboflow Universe : %d résultats pour '%s'", len(results), query)
        return results
    except Exception as exc:
        logger.warning("Recherche Universe indisponible : %s", exc)
        return []


# ---------------------------------------------------------------------------
# Téléchargement via API REST (sans SDK)
# ---------------------------------------------------------------------------

def _get_export_link(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    fmt: str = "yolov8",
    max_wait_s: int = 300,
) -> str:
    """
    Appelle l'API Roboflow pour obtenir le lien de téléchargement.
    Déclenche la génération si nécessaire et attend avec polling.
    """
    endpoint = f"{workspace}/{project}/{version}/{fmt}"
    logger.info("Requête export API : %s v%d (%s)…", f"{workspace}/{project}", version, fmt)

    deadline = time.time() + max_wait_s
    attempt  = 0

    while time.time() < deadline:
        attempt += 1
        try:
            data = _api_get(endpoint, api_key)
        except FileNotFoundError:
            logger.error("Projet introuvable sur Roboflow : %s/%s v%d", workspace, project, version)
            logger.error(
                "Vérifiez sur https://universe.roboflow.com/search?q=rugby+detection\n"
                "Ou consultez votre workspace : https://app.roboflow.com/%s", USER_WORKSPACE
            )
            raise

        link = (data.get("export") or {}).get("link")
        if link:
            # Valider que le fichier GCS existe réellement (les liens peuvent être périmés)
            try:
                head = requests.head(link, timeout=15, allow_redirects=True)
                if head.status_code == 200:
                    logger.info("Lien export validé (tentative %d)", attempt)
                    return link
                else:
                    logger.warning(
                        "Lien GCS périmé (HTTP %d) — relance de la génération export…",
                        head.status_code,
                    )
                    # Appel API supplémentaire pour forcer la régénération
                    try:
                        _api_get(f"{endpoint}?regenerate=true", api_key)
                    except Exception:
                        pass  # le paramètre est peut-être ignoré, on continue à poller
            except requests.RequestException as exc:
                logger.warning("HEAD check échoué : %s — on continue le polling", exc)

        # Export en cours de génération ou régénération côté Roboflow
        if attempt == 1:
            logger.info("Export en génération côté Roboflow, polling (max %ds)…", max_wait_s)
        wait = min(15 * attempt, 90)
        logger.debug("Tentative %d — export non prêt/périmé, attente %ds…", attempt, wait)
        time.sleep(wait)

    raise TimeoutError(
        f"Export {workspace}/{project} v{version} non disponible après {max_wait_s}s.\n"
        "Le fichier GCS est probablement supprimé. Essayez une autre version du dataset.\n"
        "→ Ouvrez https://universe.roboflow.com/{workspace}/{project} et cliquez 'Export'."
    )


def _download_and_extract(url: str, dest_dir: Path, retries: int = 3) -> None:
    """
    Télécharge un zip depuis `url`, vérifie les magic bytes PK,
    et extrait dans `dest_dir`. Retry avec backoff exponentiel.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        logger.info(
            "Téléchargement (tentative %d/%d) depuis %s…", attempt, retries, url[:80]
        )
        try:
            resp = requests.get(url, timeout=300, stream=True)
            resp.raise_for_status()

            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    chunks.append(chunk)
                    total += len(chunk)

            content = b"".join(chunks)
            logger.info("Téléchargé : %.2f MB", total / 1_048_576)

            if len(content) < 1_000:
                raise ValueError(
                    f"Réponse trop courte ({len(content)} octets) — probablement une erreur :\n"
                    f"{content[:300]!r}"
                )
            if content[:2] != b"PK":
                raise ValueError(
                    f"Ce n'est pas un fichier zip valide (magic bytes : {content[:4]!r})\n"
                    f"Début de la réponse : {content[:300]!r}"
                )

            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                zf.extractall(dest_dir)

            n_files = sum(1 for _ in dest_dir.rglob("*"))
            logger.info("Extrait dans %s (%d fichiers)", dest_dir, n_files)
            return

        except (ValueError, zipfile.BadZipFile) as exc:
            logger.error("Zip invalide (tentative %d) : %s", attempt, exc)
            if attempt == retries:
                raise
            time.sleep(5 * attempt)

        except requests.RequestException as exc:
            logger.error("Erreur réseau (tentative %d) : %s", attempt, exc)
            if attempt == retries:
                raise
            time.sleep(5 * attempt)


def download_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    dest_dir: Path,
    fmt: str = "yolov8",
) -> Path:
    """
    Télécharge et extrait un dataset Roboflow via l'API REST.

    Returns:
        Chemin vers le répertoire contenant train/valid/test + data.yaml.
    """
    if dest_dir.exists() and (dest_dir / "data.yaml").exists():
        logger.info("Dataset déjà présent : %s (skip)", dest_dir)
        return dest_dir

    link = _get_export_link(api_key, workspace, project, version, fmt)
    _download_and_extract(link, dest_dir)

    # Certains exports placent data.yaml dans un sous-dossier — on le remonte
    if not (dest_dir / "data.yaml").exists():
        candidates = list(dest_dir.rglob("data.yaml"))
        if candidates:
            root = candidates[0].parent
            if root != dest_dir:
                logger.info("Réorganisation : déplacement de %s → %s", root, dest_dir)
                for item in root.iterdir():
                    item.rename(dest_dir / item.name)
                try:
                    root.rmdir()
                except OSError:
                    pass
        else:
            logger.warning("data.yaml introuvable dans %s après extraction", dest_dir)

    return dest_dir


# ---------------------------------------------------------------------------
# Remapping des classes
# ---------------------------------------------------------------------------

def remap_labels(dataset_dir: Path) -> dict:
    """
    Lit data.yaml pour connaître les labels originaux, puis
    réécrit toutes les annotations .txt avec les indices Rugby IA.
    """
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        logger.warning("data.yaml absent dans %s — remapping ignoré", dataset_dir)
        return {}

    with open(data_yaml) as f:
        meta = yaml.safe_load(f)

    original_names: list[str] = meta.get("names", [])
    logger.info("Classes originales dans '%s' : %s", dataset_dir.name, original_names)

    idx_map: dict[int, int | None] = {}
    for orig_idx, name in enumerate(original_names):
        rugby_idx = CLASS_REMAP.get(name) or CLASS_REMAP.get(name.lower())
        idx_map[orig_idx] = rugby_idx
        label = RUGBY_IA_CLASSES[rugby_idx] if rugby_idx is not None else "IGNORÉ"
        logger.info("  [%d] %-25s → %s", orig_idx, f"'{name}'", label)

    n_remapped = n_skipped = 0
    for split in ("train", "valid", "test"):
        label_dir = dataset_dir / split / "labels"
        if not label_dir.exists():
            continue
        for lf in label_dir.glob("*.txt"):
            new_lines = []
            for line in lf.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                new_cls = idx_map.get(int(parts[0]))
                if new_cls is None:
                    n_skipped += 1
                    continue
                new_lines.append(f"{new_cls} " + " ".join(parts[1:]))
                n_remapped += 1
            lf.write_text("\n".join(new_lines))

    logger.info(
        "Remapping '%s' : %d annotations OK, %d ignorées",
        dataset_dir.name, n_remapped, n_skipped,
    )
    return idx_map


# ---------------------------------------------------------------------------
# Génération du data.yaml fusionné
# ---------------------------------------------------------------------------

def generate_data_yaml(dataset_dirs: list[Path], output_path: Path) -> dict:
    """Crée un data.yaml unifié pointant vers tous les splits disponibles."""
    train_paths, val_paths, test_paths = [], [], []
    for d in dataset_dirs:
        for split, lst in (("train", train_paths), ("valid", val_paths), ("test", test_paths)):
            img_dir = d / split / "images"
            if img_dir.exists() and any(img_dir.iterdir()):
                lst.append(str(img_dir.resolve()))

    data = {
        "path":  str(output_path.parent.resolve()),
        "train": train_paths[0] if len(train_paths) == 1 else (train_paths or ""),
        "val":   val_paths[0]   if len(val_paths)   == 1 else (val_paths   or ""),
        "test":  test_paths[0]  if len(test_paths)  == 1 else (test_paths  or ""),
        "nc":    len(RUGBY_IA_CLASSES),
        "names": list(RUGBY_IA_CLASSES.values()),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    logger.info("data.yaml fusionné écrit : %s", output_path)
    return data


# ---------------------------------------------------------------------------
# Statistiques
# ---------------------------------------------------------------------------

def dataset_statistics(dataset_dirs: list[Path]) -> dict:
    stats: dict = {
        "splits":        {},
        "class_counts":  {v: 0 for v in RUGBY_IA_CLASSES.values()},
    }
    for d in dataset_dirs:
        for split in ("train", "valid", "test"):
            img_dir   = d / split / "images"
            label_dir = d / split / "labels"
            if not img_dir.exists():
                continue
            n_img = sum(
                1 for p in img_dir.glob("*")
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            n_ann = 0
            if label_dir.exists():
                for lf in label_dir.glob("*.txt"):
                    for line in lf.read_text().splitlines():
                        if line.strip():
                            cls  = int(line.split()[0])
                            name = RUGBY_IA_CLASSES.get(cls, "unknown")
                            stats["class_counts"][name] = (
                                stats["class_counts"].get(name, 0) + 1
                            )
                            n_ann += 1
            stats["splits"][f"{d.name}/{split}"] = {"images": n_img, "annotations": n_ann}
    return stats


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def list_datasets(api_key: str | None = None):
    print(f"\n{'#':<3} {'Workspace':<26} {'Project':<38} {'v':<3}  Description")
    print("─" * 95)
    for i, ds in enumerate(RUGBY_UNIVERSE_DATASETS, 1):
        print(
            f"{i:<3} {ds['workspace']:<26} {ds['project']:<38} "
            f"{ds['version']:<3}  {ds['description']}"
        )
    print()
    print(f"  Universe : https://universe.roboflow.com/search?q=rugby+detection")
    print(f"  Workspace: https://app.roboflow.com/{USER_WORKSPACE}")
    print()

    if api_key:
        results = search_universe(api_key, "rugby detection", max_results=15)
        if results:
            print("  Résultats live Roboflow Universe :")
            print(f"  {'ID':<60} Versions")
            print("  " + "─" * 70)
            for r in results:
                print(f"  {r.get('id', ''):<60} {r.get('versions', '?')}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Téléchargement datasets rugby depuis Roboflow Universe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--api-key",   default=None,
                   help="Clé API Roboflow (ou ROBOFLOW_API_KEY dans .env)")
    p.add_argument("--workspace", default=None,
                   help="Workspace Roboflow Universe (ex: rugby-analysis)")
    p.add_argument("--project",   default=None,
                   help="Nom du projet (ex: rugby-player-detection)")
    p.add_argument("--version",   type=int, default=None,
                   help="Version du dataset (défaut : celle du dataset choisi)")
    p.add_argument("--format",    default="yolov8",
                   help="Format d'export Roboflow (défaut : yolov8)")
    p.add_argument("--output",    default="data/roboflow/merged",
                   help="Répertoire de sortie (défaut : data/roboflow/merged)")
    p.add_argument("--list",      action="store_true",
                   help="Lister les datasets rugby disponibles sur Universe")
    p.add_argument("--all",       action="store_true",
                   help="Télécharger tous les datasets rugby connus")
    p.add_argument("--search",    default=None,
                   help="Rechercher sur Roboflow Universe (ex: 'rugby detection')")
    p.add_argument("--no-remap",  action="store_true",
                   help="Désactiver le remapping automatique des classes")
    # Alias rétrocompatibilité
    p.add_argument("--all-known", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args    = parse_args()
    api_key = None if args.list and not args.api_key else None  # lazy

    if args.list:
        api_key = os.getenv("ROBOFLOW_API_KEY") or args.api_key
        list_datasets(api_key)
        return

    api_key = get_api_key(args.api_key)

    if args.search:
        results = search_universe(api_key, args.search)
        if results:
            print(f"\n{'ID':<60} Versions")
            print("─" * 70)
            for r in results[:20]:
                print(f"  {r.get('id', ''):<60} {r.get('versions', '?')}")
        else:
            print(
                "Aucun résultat live.\n"
                "→ https://universe.roboflow.com/search?q=rugby+detection\n"
                f"→ https://app.roboflow.com/{USER_WORKSPACE}"
            )
        return

    output_base      = Path(args.output)
    downloaded_dirs: list[Path] = []

    # ── Mode explicite : --workspace + --project ──────────────────────────
    if args.workspace and args.project:
        version = args.version or 1
        dest    = output_base / f"{args.workspace}__{args.project}_v{version}"
        try:
            download_dataset(api_key, args.workspace, args.project, version, dest, args.format)
            downloaded_dirs.append(dest)
        except Exception as exc:
            logger.error("Téléchargement échoué : %s", exc)
            sys.exit(1)

    # ── Mode --all (ou alias --all-known) ─────────────────────────────────
    elif args.all or args.all_known:
        for ds in sorted(RUGBY_UNIVERSE_DATASETS, key=lambda x: x["priority"]):
            version = args.version or ds["version"]
            dest    = output_base / f"{ds['workspace']}__{ds['project']}_v{version}"
            logger.info("=== %s ===", ds["description"])
            try:
                download_dataset(
                    api_key, ds["workspace"], ds["project"], version, dest, args.format
                )
                downloaded_dirs.append(dest)
            except Exception as exc:
                logger.warning("Échec %s/%s : %s", ds["workspace"], ds["project"], exc)

    # ── Mode par défaut : priorité 1 et 2, fallback sur les suivants ──────
    else:
        for ds in sorted(RUGBY_UNIVERSE_DATASETS, key=lambda x: x["priority"]):
            version = args.version or ds["version"]
            dest    = output_base / f"{ds['workspace']}__{ds['project']}_v{version}"
            logger.info("=== %s ===", ds["description"])
            try:
                download_dataset(
                    api_key, ds["workspace"], ds["project"], version, dest, args.format
                )
                downloaded_dirs.append(dest)
                if len(downloaded_dirs) >= 2:  # 2 datasets suffisent pour démarrer
                    break
            except Exception as exc:
                logger.warning(
                    "Échec %s/%s : %s — tentative sur le dataset suivant",
                    ds["workspace"], ds["project"], exc,
                )

    if not downloaded_dirs:
        logger.error(
            "Aucun dataset téléchargé.\n"
            "  Consultez : https://universe.roboflow.com/search?q=rugby+detection\n"
            "  Puis relancez avec : --workspace <ws> --project <proj> --version <v>"
        )
        sys.exit(1)

    # ── Remapping des classes ─────────────────────────────────────────────
    if not args.no_remap:
        for d in downloaded_dirs:
            remap_labels(d)

    # ── data.yaml fusionné ────────────────────────────────────────────────
    merged_yaml = output_base / "data.yaml"
    generate_data_yaml(downloaded_dirs, merged_yaml)

    # ── Statistiques ──────────────────────────────────────────────────────
    stats     = dataset_statistics(downloaded_dirs)
    total_img = sum(v["images"] for v in stats["splits"].values())
    total_ann = sum(v["annotations"] for v in stats["splits"].values())

    print("\n" + "=" * 60)
    print("  DATASET RUGBY IA — PRÊT")
    print("=" * 60)
    print(f"  Images totales      : {total_img}")
    print(f"  Annotations totales : {total_ann}")
    print("\n  Par split :")
    for key, s in stats["splits"].items():
        print(f"    {key:<44} {s['images']:>5} img  {s['annotations']:>7} ann")
    print("\n  Par classe :")
    for cls, cnt in stats["class_counts"].items():
        bar = "█" * min(30, cnt // max(1, total_ann // 30))
        print(f"    {cls:<12} {cnt:>7}  {bar}")
    print(f"\n  data.yaml prêt : {merged_yaml}")
    print(f"\n  Prochaine étape :")
    print(f"    python scripts/finetune_yolo_rugby.py --data {merged_yaml}")
    print("=" * 60)

    # Rapport JSON
    report = output_base / "dataset_report.json"
    with open(report, "w") as f:
        json.dump(
            {"datasets": [str(d) for d in downloaded_dirs], "stats": stats},
            f, indent=2,
        )
    print(f"\n  Rapport : {report}")


if __name__ == "__main__":
    main()



def load_config() -> dict:
    with open(Path(__file__).parent.parent / "config" / "config.yaml", "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Téléchargement d'un dataset
# ---------------------------------------------------------------------------

def download_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    export_dir: str,
    fmt: str = "yolov8",
) -> Path:
    """
    Télécharge un dataset Roboflow au format YOLOv8.

    Returns:
        Chemin vers le répertoire téléchargé (contient train/valid/test + data.yaml).
    """
    try:
        from roboflow import Roboflow  # noqa: PLC0415
    except ImportError:
        logger.error("Package roboflow absent. Lancez : pip install roboflow")
        sys.exit(1)

    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Connexion à Roboflow…")
    rf = Roboflow(api_key=api_key)

    logger.info("Workspace : %s | Projet : %s | Version : %d", workspace, project, version)
    try:
        proj = rf.workspace(workspace).project(project)
        ds   = proj.version(version)
    except Exception as exc:
        logger.error("Projet introuvable : %s/%s v%d — %s", workspace, project, version, exc)
        raise

    logger.info("Téléchargement au format %s dans %s …", fmt, out)
    ds.download(fmt, location=str(out), overwrite=True)
    logger.info("Dataset téléchargé : %s", out)
    return out


# ---------------------------------------------------------------------------
# Remapping des classes
# ---------------------------------------------------------------------------

def remap_labels(dataset_dir: Path) -> dict:
    """
    Relit data.yaml pour connaître les labels originaux, puis
    réécrit toutes les annotations .txt pour mapper vers les indices Rugby IA.

    Returns:
        Dictionnaire de mapping original_idx → rugby_ia_idx.
    """
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        logger.warning("data.yaml absent dans %s, pas de remapping.", dataset_dir)
        return {}

    with open(data_yaml, "r") as f:
        meta = yaml.safe_load(f)

    original_names: list[str] = meta.get("names", [])
    logger.info("Classes originales : %s", original_names)

    # Construire le mapping index → index
    idx_map: dict[int, int | None] = {}
    for orig_idx, name in enumerate(original_names):
        rugby_idx = CLASS_REMAP.get(name)
        if rugby_idx is None:
            rugby_idx = CLASS_REMAP.get(name.lower())
        idx_map[orig_idx] = rugby_idx
        status = f"→ {RUGBY_IA_CLASSES[rugby_idx]}" if rugby_idx is not None else "→ IGNORÉ"
        logger.info("  Classe %d '%s' %s", orig_idx, name, status)

    # Réécrire les fichiers .txt
    n_remapped = n_skipped = 0
    for split in ("train", "valid", "test"):
        label_dir = dataset_dir / split / "labels"
        if not label_dir.exists():
            continue
        for lf in label_dir.glob("*.txt"):
            new_lines = []
            for line in lf.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                orig_cls = int(parts[0])
                new_cls = idx_map.get(orig_cls)
                if new_cls is None:
                    n_skipped += 1
                    continue   # Classe non reconnue → exclue
                new_lines.append(f"{new_cls} " + " ".join(parts[1:]))
                n_remapped += 1
            lf.write_text("\n".join(new_lines))

    logger.info(
        "Remapping terminé : %d annotations remappées, %d ignorées.", n_remapped, n_skipped
    )
    return idx_map


# ---------------------------------------------------------------------------
# Génération du data.yaml fusionné
# ---------------------------------------------------------------------------

def generate_data_yaml(dataset_dirs: list[Path], output_path: Path):
    """
    Crée un data.yaml unifié pointant vers les splits de tous les datasets.
    Gère la fusion de plusieurs sources.
    """
    train_paths, val_paths, test_paths = [], [], []

    for d in dataset_dirs:
        for split, lst in (("train", train_paths), ("valid", val_paths), ("test", test_paths)):
            img_dir = d / split / "images"
            if img_dir.exists() and any(img_dir.iterdir()):
                lst.append(str(img_dir.resolve()))

    data = {
        "path": str(output_path.parent.resolve()),
        "train": train_paths if len(train_paths) > 1 else (train_paths[0] if train_paths else ""),
        "val":   val_paths   if len(val_paths)   > 1 else (val_paths[0]   if val_paths   else ""),
        "test":  test_paths  if len(test_paths)  > 1 else (test_paths[0]  if test_paths  else ""),
        "nc": len(RUGBY_IA_CLASSES),
        "names": list(RUGBY_IA_CLASSES.values()),
    }

    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    logger.info("data.yaml généré : %s", output_path)
    return data


# ---------------------------------------------------------------------------
# Statistiques dataset
# ---------------------------------------------------------------------------

def dataset_statistics(dataset_dirs: list[Path]) -> dict:
    """Compte les images et annotations par split et par classe."""
    stats: dict = {
        "splits": {},
        "class_counts": {v: 0 for v in RUGBY_IA_CLASSES.values()},
    }

    for d in dataset_dirs:
        for split in ("train", "valid", "test"):
            img_dir   = d / split / "images"
            label_dir = d / split / "labels"
            if not img_dir.exists():
                continue
            n_images = sum(1 for _ in img_dir.glob("*") if _.suffix in (".jpg", ".png", ".jpeg"))
            n_labels = 0
            if label_dir.exists():
                for lf in label_dir.glob("*.txt"):
                    for line in lf.read_text().splitlines():
                        if line.strip():
                            cls = int(line.split()[0])
                            class_name = RUGBY_IA_CLASSES.get(cls, "unknown")
                            stats["class_counts"][class_name] = (
                                stats["class_counts"].get(class_name, 0) + 1
                            )
                            n_labels += 1

            key = f"{d.name}/{split}"
            stats["splits"][key] = {"images": n_images, "annotations": n_labels}

    return stats


# ---------------------------------------------------------------------------
# NOTE : les fonctions list_datasets, parse_args et main sont définies
#        plus bas (nouvelles implémentations REST sans SDK).
# ---------------------------------------------------------------------------

