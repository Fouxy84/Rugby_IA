"""Run lightweight pretrained YOLO inference on an image, video, webcam, or URL."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Détection YOLO légère (personnes et ballons COCO) sur image ou vidéo."
    )
    parser.add_argument("source", help="Image, vidéo, URL vidéo ou index webcam (ex. 0).")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Poids YOLO pré-entraînés (défaut : yolov8n.pt, téléchargés automatiquement si absents).",
    )
    parser.add_argument("--device", default="cpu", help="Périphérique : cpu, 0, 0,1, etc. (défaut : cpu).")
    parser.add_argument("--conf", type=float, default=0.35, help="Seuil de confiance entre 0 et 1.")
    parser.add_argument("--imgsz", type=int, default=640, help="Taille d'inférence (défaut : 640).")
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Détecter toutes les classes COCO au lieu des personnes et ballons uniquement.",
    )
    parser.add_argument("--output-dir", default="runs/detect_yolo", help="Dossier de sortie.")
    parser.add_argument("--name", default="rugby", help="Nom du sous-dossier de résultats.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.conf <= 1.0:
        raise SystemExit("--conf doit être compris entre 0 et 1.")
    if args.imgsz <= 0:
        raise SystemExit("--imgsz doit être supérieur à 0.")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics est requis : installez les dépendances avec pip install -r requirements.txt.") from exc

    model = YOLO(args.model)
    results = model.predict(
        source=args.source,
        device=args.device,
        conf=args.conf,
        imgsz=args.imgsz,
        classes=None if args.all_classes else [0, 32],
        save=True,
        project=args.output_dir,
        name=args.name,
        exist_ok=True,
        verbose=False,
    )

    output_path = Path(args.output_dir) / args.name
    print(f"{len(results)} image(s) traitée(s). Résultats : {output_path}")


if __name__ == "__main__":
    main()