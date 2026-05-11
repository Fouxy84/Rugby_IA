"""
Fine-tuning YOLOv8 sur dataset rugby Roboflow.

Pipeline complet :
  1. Chargement de la config (config.yaml)
  2. Vérification du data.yaml (issu de download_roboflow_dataset.py)
  3. Entraînement YOLOv8 avec tous les hyperparamètres Rugby IA
  4. Évaluation sur le set de validation (mAP50, mAP50-95, précision, rappel)
  5. Export en ONNX + TorchScript pour déploiement
  6. Copie du modèle final → data/models/rugby_detector.pt
  7. Tracking complet MLflow

Usage :
    # Après avoir téléchargé le dataset
    python scripts/finetune_yolo_rugby.py --data data/roboflow/merged/data.yaml

    # Reprendre depuis un checkpoint
    python scripts/finetune_yolo_rugby.py --data data.yaml --resume runs/detect/rugby_v1/weights/last.pt

    # Validation seule sur un modèle entraîné
    python scripts/finetune_yolo_rugby.py --data data.yaml --val-only --weights data/models/rugby_detector.pt

    # Export ONNX uniquement
    python scripts/finetune_yolo_rugby.py --export-only --weights data/models/rugby_detector.pt
"""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("finetune_yolo")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open(Path(__file__).parent.parent / "config" / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def resolve_device(device_str: str) -> str:
    """Vérifie la disponibilité CUDA et retourne le device effectif."""
    import torch  # noqa: PLC0415
    if device_str in ("cpu",):
        return "cpu"
    if not torch.cuda.is_available():
        logger.warning("CUDA indisponible, basculement sur CPU.")
        return "cpu"
    if device_str == "0":
        logger.info("GPU : %s", torch.cuda.get_device_name(0))
    return device_str


# ---------------------------------------------------------------------------
# Vérification du data.yaml
# ---------------------------------------------------------------------------

def validate_data_yaml(data_path: Path) -> dict:
    """Valide le data.yaml et retourne les métadonnées."""
    if not data_path.exists():
        logger.error(
            "data.yaml introuvable : %s\n"
            "Lancez d'abord : python scripts/download_roboflow_dataset.py",
            data_path,
        )
        sys.exit(1)

    with open(data_path, "r") as f:
        meta = yaml.safe_load(f)

    required = ["train", "val", "nc", "names"]
    for k in required:
        if k not in meta:
            logger.error("Clé '%s' manquante dans data.yaml", k)
            sys.exit(1)

    logger.info("data.yaml validé : %d classe(s) — %s", meta["nc"], meta["names"])

    # Vérification des chemins
    for split_key in ("train", "val", "test"):
        paths = meta.get(split_key, [])
        if isinstance(paths, str):
            paths = [paths]
        for p in (paths if isinstance(paths, list) else [paths]):
            if p and not Path(p).exists():
                logger.warning("Répertoire %s introuvable : %s", split_key, p)

    return meta


# ---------------------------------------------------------------------------
# Création du fichier d'augmentation custom
# ---------------------------------------------------------------------------

def write_augment_yaml(cfg: dict, output_path: Path):
    """
    Écrit un fichier YAML d'hyperparamètres d'augmentation custom
    compatibles Ultralytics.
    """
    aug = cfg["finetune"]["augment"]
    with open(output_path, "w") as f:
        yaml.dump(aug, f, default_flow_style=False)
    logger.info("Augmentations écrites : %s", output_path)


# ---------------------------------------------------------------------------
# Entraînement principal
# ---------------------------------------------------------------------------

def train(args, cfg: dict):
    from ultralytics import YOLO  # noqa: PLC0415
    import mlflow  # noqa: PLC0415

    ft = cfg["finetune"]
    device = resolve_device(args.device or ft.get("device", "cpu"))
    run_name = f"rugby_yolo_{time.strftime('%Y%m%d_%H%M%S')}"

    # --- Modèle de départ ---
    if args.resume:
        logger.info("Reprise depuis : %s", args.resume)
        model = YOLO(args.resume)
    else:
        base = args.weights or ft["base_model"]
        logger.info("Modèle de base : %s", base)
        model = YOLO(base)

    # --- Hyperparamètres ---
    train_kwargs = dict(
        data        = str(args.data),
        epochs      = args.epochs    or ft["epochs"],
        imgsz       = args.imgsz     or ft["imgsz"],
        batch       = args.batch     or ft["batch"],
        patience    = ft["patience"],
        optimizer   = ft["optimizer"],
        lr0         = ft["lr0"],
        lrf         = ft["lrf"],
        momentum    = ft["momentum"],
        weight_decay= ft["weight_decay"],
        warmup_epochs    = ft["warmup_epochs"],
        close_mosaic     = ft["close_mosaic"],
        device      = device,
        workers     = ft["workers"],
        cache       = ft["cache"],
        rect        = ft["rect"],
        cos_lr      = ft["cos_lr"],
        amp         = ft["amp"],
        plots       = ft["plots"],
        val         = ft["val"],
        exist_ok    = ft["exist_ok"],
        name        = args.run_name or run_name,
        project     = "runs/detect",
        # Augmentations
        hsv_h       = ft["augment"]["hsv_h"],
        hsv_s       = ft["augment"]["hsv_s"],
        hsv_v       = ft["augment"]["hsv_v"],
        degrees     = ft["augment"]["degrees"],
        translate   = ft["augment"]["translate"],
        scale       = ft["augment"]["scale"],
        shear       = ft["augment"]["shear"],
        perspective = ft["augment"]["perspective"],
        flipud      = ft["augment"]["flipud"],
        fliplr      = ft["augment"]["fliplr"],
        mosaic      = ft["augment"]["mosaic"],
        mixup       = ft["augment"]["mixup"],
        copy_paste  = ft["augment"]["copy_paste"],
        erasing     = ft["augment"]["erasing"],
    )

    logger.info("Démarrage de l'entraînement sur device=%s …", device)
    logger.info("Epochs : %d | imgsz : %d | batch : %d",
                train_kwargs["epochs"], train_kwargs["imgsz"], train_kwargs["batch"])

    # --- MLflow ---
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment("rugby_yolo_finetune")

    with mlflow.start_run(run_name=run_name) as run:
        # Log des hyperparamètres
        mlflow.log_params({k: v for k, v in train_kwargs.items()
                           if isinstance(v, (int, float, str, bool))})
        mlflow.log_param("base_model", ft["base_model"])
        mlflow.log_param("data_yaml",  str(args.data))
        mlflow.log_param("run_name",   run_name)

        # Entraînement
        results = model.train(**train_kwargs)

        # Log des métriques finales
        metrics = results.results_dict if hasattr(results, "results_dict") else {}
        for k, v in metrics.items():
            try:
                mlflow.log_metric(k.replace("/", "_"), float(v))
            except (ValueError, TypeError):
                pass

        # Chemin du meilleur modèle
        best_weights = Path("runs/detect") / train_kwargs["name"] / "weights" / "best.pt"
        if best_weights.exists():
            mlflow.log_artifact(str(best_weights), artifact_path="weights")
            logger.info("Meilleur modèle : %s", best_weights)
        else:
            # Fallback sur last.pt
            best_weights = Path("runs/detect") / train_kwargs["name"] / "weights" / "last.pt"

        mlflow.log_param("best_weights_path", str(best_weights))
        logger.info("MLflow run ID : %s", run.info.run_id)

    return model, best_weights, results


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def evaluate(model, data_path: Path, device: str, imgsz: int = 1280) -> dict:
    """Lance la validation et retourne les métriques."""
    logger.info("Évaluation sur le set de validation…")
    val_results = model.val(
        data=str(data_path),
        imgsz=imgsz,
        device=device,
        plots=True,
        save_json=True,
    )
    metrics = {
        "mAP50":    getattr(val_results.box, "map50",  None),
        "mAP50_95": getattr(val_results.box, "map",    None),
        "precision":getattr(val_results.box, "mp",     None),
        "recall":   getattr(val_results.box, "mr",     None),
    }
    logger.info(
        "Résultats : mAP50=%.3f | mAP50-95=%.3f | P=%.3f | R=%.3f",
        metrics["mAP50"]    or 0,
        metrics["mAP50_95"] or 0,
        metrics["precision"]or 0,
        metrics["recall"]   or 0,
    )
    return metrics


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_model(weights_path: Path, imgsz: int = 1280, device: str = "cpu"):
    """Exporte le modèle en ONNX et TorchScript."""
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(str(weights_path))

    logger.info("Export ONNX …")
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,
        dynamic=False,
        opset=17,
        device=device,
    )
    logger.info("ONNX exporté : %s", onnx_path)

    logger.info("Export TorchScript …")
    ts_path = model.export(
        format="torchscript",
        imgsz=imgsz,
        device=device,
    )
    logger.info("TorchScript exporté : %s", ts_path)

    return onnx_path, ts_path


# ---------------------------------------------------------------------------
# Copie vers data/models
# ---------------------------------------------------------------------------

def deploy_weights(best_weights: Path, cfg: dict):
    """Copie le meilleur modèle vers le chemin de déploiement configuré."""
    ft = cfg["finetune"]
    dest = Path(ft["output_dir"]) / f"{ft['output_name']}.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, dest)
    logger.info("Modèle déployé : %s → %s", best_weights, dest)
    return dest


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tuning YOLOv8 sur dataset rugby Roboflow")
    p.add_argument("--data",     default="data/roboflow/merged/data.yaml",
                   help="Chemin vers data.yaml (généré par download_roboflow_dataset.py)")
    p.add_argument("--weights",  default=None, help="Poids de départ (défaut : config)")
    p.add_argument("--resume",   default=None, help="Reprendre depuis un checkpoint (last.pt)")
    p.add_argument("--epochs",   type=int, default=None, help="Nombre d'epochs (défaut : config)")
    p.add_argument("--imgsz",    type=int, default=None, help="Taille des images (défaut : config)")
    p.add_argument("--batch",    type=int, default=None, help="Batch size (défaut : config)")
    p.add_argument("--device",   default=None, help="Device : 0 | cpu | 0,1 (défaut : config)")
    p.add_argument("--run-name", default=None, help="Nom du run MLflow/Ultralytics")
    p.add_argument("--val-only",    action="store_true", help="Validation uniquement (pas d'entraînement)")
    p.add_argument("--export-only", action="store_true", help="Export ONNX/TorchScript uniquement")
    p.add_argument("--no-export",   action="store_true", help="Désactive l'export post-entraînement")
    p.add_argument("--no-deploy",   action="store_true", help="Ne copie pas le modèle dans data/models/")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config()
    ft   = cfg["finetune"]
    data_path = Path(args.data)
    device = resolve_device(args.device or ft.get("device", "cpu"))
    imgsz  = args.imgsz or ft["imgsz"]

    # --- Export uniquement ---
    if args.export_only:
        weights = args.weights or cfg["detection"]["fine_tuned_weights"]
        export_model(Path(weights), imgsz=imgsz, device=device)
        return

    # --- Validation uniquement ---
    if args.val_only:
        from ultralytics import YOLO  # noqa: PLC0415
        weights = args.weights or cfg["detection"]["fine_tuned_weights"]
        validate_data_yaml(data_path)
        model = YOLO(weights)
        metrics = evaluate(model, data_path, device, imgsz)
        print("\n=== Métriques de validation ===")
        for k, v in metrics.items():
            print(f"  {k:<15} {v:.4f}" if v is not None else f"  {k:<15} N/A")
        return

    # --- Entraînement ---
    validate_data_yaml(data_path)
    model, best_weights, results = train(args, cfg)

    # --- Évaluation finale ---
    metrics = evaluate(model, data_path, device, imgsz)

    # --- Export ---
    if not args.no_export:
        export_model(best_weights, imgsz=imgsz, device=device)

    # --- Déploiement ---
    if not args.no_deploy and best_weights.exists():
        deployed = deploy_weights(best_weights, cfg)

        print("\n" + "=" * 60)
        print("  FINE-TUNING TERMINÉ")
        print("=" * 60)
        print(f"  Modèle déployé : {deployed}")
        print(f"  mAP50          : {metrics.get('mAP50', 0):.4f}")
        print(f"  mAP50-95       : {metrics.get('mAP50_95', 0):.4f}")
        print(f"  Précision      : {metrics.get('precision', 0):.4f}")
        print(f"  Rappel         : {metrics.get('recall', 0):.4f}")
        print("=" * 60)
        print("\nL'API Rugby IA utilisera automatiquement ce modèle.")
        print(f"  → Configuré dans config.yaml : detection.fine_tuned_weights")


if __name__ == "__main__":
    main()
