"""
Téléchargement des datasets Roboflow pour le fine-tuning Rugby IA.

Ce script :
  1. Authentifie via l'API Key Roboflow
  2. Télécharge les datasets rugby (joueurs, arbitres, ballon)
  3. Fusionne plusieurs datasets si nécessaire
  4. Génère le fichier data.yaml compatible YOLOv8
  5. Affiche les statistiques du dataset

Usage :
    python scripts/download_roboflow_dataset.py --api-key <KEY>
    python scripts/download_roboflow_dataset.py --api-key <KEY> --list
    python scripts/download_roboflow_dataset.py --api-key <KEY> --workspace <ws> --project <proj> --version 2
    python scripts/download_roboflow_dataset.py --api-key <KEY> --search "rugby"

Vous pouvez aussi poser ROBOFLOW_API_KEY dans .env pour éviter de passer --api-key.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("download_roboflow")

# Datasets rugby publics connus sur Roboflow Universe
# Format : (workspace, project, version_recommandée, description)
KNOWN_RUGBY_DATASETS = [
    ("roboflow-100",      "rugby-detection",           1, "Joueurs + ballon (Roboflow-100)"),
    ("roboflow-100",      "rugby-players-2",            2, "Joueurs, arbitres, ballon"),
    ("rugby-analysis",    "rugby-player-detection",     1, "Détection joueurs Top14 / Premiership"),
    ("sports-detection",  "rugby-ball-detection",       1, "Détection ballon uniquement"),
    ("rugby-ia-datasets", "rugby-players-tracking",     1, "Joueurs + tracking ID"),
    ("roboflow-100",      "rugby-players-tracking",     1, "Tracking joueurs avec IDs"),
]

CLASS_REMAP = {
    # toutes les variantes de label → index interne Rugby IA
    "player":       0, "Player":       0, "rugby player": 0, "rugby_player": 0,
    "players":      0, "Players":      0, "person":       0,
    "referee":      1, "Referee":      1, "ref":          1, "umpire":       1,
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
            "API Key Roboflow manquante.\n"
            "  → Obtenez-la sur https://app.roboflow.com/settings/api\n"
            "  → Passez-la via --api-key OU ajoutez ROBOFLOW_API_KEY dans .env"
        )
        sys.exit(1)
    return key


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
# Recherche de datasets sur Roboflow Universe
# ---------------------------------------------------------------------------

def search_roboflow(api_key: str, query: str):
    """Recherche des datasets rugby sur Roboflow Universe."""
    try:
        from roboflow import Roboflow  # noqa: PLC0415
        rf = Roboflow(api_key=api_key)
        logger.info("Recherche de '%s' sur Roboflow Universe…", query)
        # L'API Roboflow Universe n'expose pas directement la recherche
        # → On liste les datasets connus et affiche une suggestion
        logger.info("Datasets rugby publics connus :")
        for ws, proj, ver, desc in KNOWN_RUGBY_DATASETS:
            logger.info("  %-35s  v%d  —  %s", f"{ws}/{proj}", ver, desc)
        logger.info(
            "\nPour explorer davantage : https://universe.roboflow.com/search?q=%s", query
        )
    except Exception as exc:
        logger.error("Erreur Roboflow : %s", exc)


# ---------------------------------------------------------------------------
# Listing des datasets connus
# ---------------------------------------------------------------------------

def list_known_datasets():
    print("\n Datasets rugby disponibles sur Roboflow Universe :\n")
    print(f"{'#':<3} {'Workspace':<25} {'Project':<35} {'v':<4} {'Description'}")
    print("-" * 90)
    for i, (ws, proj, ver, desc) in enumerate(KNOWN_RUGBY_DATASETS, 1):
        print(f"{i:<3} {ws:<25} {proj:<35} {ver:<4} {desc}")
    print(
        "\n → Lien Universe : https://universe.roboflow.com/search?q=rugby\n"
        " → Créez votre propre projet : https://app.roboflow.com\n"
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Téléchargement dataset Roboflow pour Rugby IA")
    p.add_argument("--api-key",  default=None, help="Clé API Roboflow (ou ROBOFLOW_API_KEY dans .env)")
    p.add_argument("--workspace", default=None, help="Workspace Roboflow")
    p.add_argument("--project",   default=None, help="Nom du projet")
    p.add_argument("--version",   type=int, default=1, help="Version du dataset (défaut: 1)")
    p.add_argument("--format",    default="yolov8", help="Format d'export (défaut: yolov8)")
    p.add_argument("--output",    default="data/roboflow/merged", help="Répertoire de sortie")
    p.add_argument("--list",      action="store_true", help="Lister les datasets rugby connus")
    p.add_argument("--search",    default=None, help="Rechercher un terme sur Roboflow Universe")
    p.add_argument(
        "--all-known",
        action="store_true",
        help="Télécharger tous les datasets rugby connus (peut être long)",
    )
    p.add_argument(
        "--no-remap",
        action="store_true",
        help="Désactive le remapping automatique des classes",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.list:
        list_known_datasets()
        return

    api_key = get_api_key(args.api_key)

    if args.search:
        search_roboflow(api_key, args.search)
        return

    cfg = load_config()
    ft_cfg = cfg["finetune"]
    output_base = Path(args.output)
    downloaded_dirs: list[Path] = []

    if args.workspace and args.project:
        # Dataset explicite fourni en argument
        out_dir = output_base / f"{args.workspace}__{args.project}_v{args.version}"
        dl_dir = download_dataset(
            api_key, args.workspace, args.project, args.version,
            str(out_dir), args.format,
        )
        downloaded_dirs.append(dl_dir)

    elif args.all_known:
        for ws, proj, ver, desc in KNOWN_RUGBY_DATASETS:
            logger.info("=== %s ===", desc)
            out_dir = output_base / f"{ws}__{proj}_v{ver}"
            try:
                dl_dir = download_dataset(api_key, ws, proj, ver, str(out_dir), args.format)
                downloaded_dirs.append(dl_dir)
            except Exception as exc:
                logger.warning("Échec pour %s/%s : %s", ws, proj, exc)

    else:
        # Par défaut : télécharge le dataset principal et le dataset ballon
        for ds_key in ("players", "ball"):
            ds_cfg = ft_cfg["datasets"][ds_key]
            out_dir = Path(ds_cfg["export_dir"])
            try:
                dl_dir = download_dataset(
                    api_key,
                    ds_cfg["workspace"],
                    ds_cfg["project"],
                    ds_cfg["version"],
                    str(out_dir),
                    ds_cfg["format"],
                )
                downloaded_dirs.append(dl_dir)
            except Exception as exc:
                logger.warning("Dataset '%s' indisponible : %s", ds_key, exc)

    if not downloaded_dirs:
        logger.error("Aucun dataset téléchargé.")
        sys.exit(1)

    # --- Remapping des classes ---
    if not args.no_remap:
        for d in downloaded_dirs:
            logger.info("Remapping des classes dans %s …", d)
            remap_labels(d)

    # --- Génération du data.yaml fusionné ---
    merged_yaml = output_base / "data.yaml"
    generate_data_yaml(downloaded_dirs, merged_yaml)

    # --- Statistiques ---
    stats = dataset_statistics(downloaded_dirs)
    print("\n=== Statistiques du dataset ===")
    total_img = sum(v["images"] for v in stats["splits"].values())
    total_ann = sum(v["annotations"] for v in stats["splits"].values())
    print(f"Total images      : {total_img}")
    print(f"Total annotations : {total_ann}")
    print("\nPar split :")
    for split, s in stats["splits"].items():
        print(f"  {split:<40} {s['images']:>5} images  {s['annotations']:>7} annotations")
    print("\nPar classe (annotations totales) :")
    for cls, cnt in stats["class_counts"].items():
        bar = "█" * min(40, cnt // max(1, total_ann // 40))
        print(f"  {cls:<12} {cnt:>7}  {bar}")

    # Sauvegarde du rapport
    report_path = output_base / "dataset_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(
            {"datasets": [str(d) for d in downloaded_dirs], "stats": stats},
            f, indent=2,
        )
    print(f"\nRapport sauvegardé : {report_path}")
    print(f"data.yaml prêt    : {merged_yaml}")
    print("\nProchaine étape :")
    print(f"  python scripts/finetune_yolo_rugby.py --data {merged_yaml}")


if __name__ == "__main__":
    main()
