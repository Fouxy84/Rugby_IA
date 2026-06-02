#!/usr/bin/env python3
"""
Test de dépendances et diagnose CPU pour Rugby IA.

Usage:
    python scripts/test_cpu_setup.py
"""

import sys
from pathlib import Path

def test_imports():
    """Tester que toutes les dépendances sont disponibles."""
    print("=" * 60)
    print("  TEST DES DÉPENDANCES")
    print("=" * 60)
    
    deps = [
        ("yaml", "PyYAML"),
        ("cv2", "OpenCV"),
        ("fastapi", "FastAPI"),
        ("streamlit", "Streamlit"),
        ("mlflow", "MLflow"),
        ("requests", "Requests"),
    ]
    
    failed = []
    for module, name in deps:
        try:
            __import__(module)
            print(f"OK   {name:<20}")
        except ImportError as e:
            print(f"FAIL {name:<20} {e}")
            failed.append((module, name))
    
    return len(failed) == 0, failed

def test_pytorch():
    """Vérifier PyTorch et ses capacités."""
    print("\n" + "=" * 60)
    print("  TEST PYTORCH")
    print("=" * 60)
    
    try:
        import torch
        print(f"PyTorch version  : {torch.__version__}")
        print(f"CUDA available   : {torch.cuda.is_available()}")
        print(f"Device count     : {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            print(f"Current device   : {torch.cuda.current_device()}")
            print(f"Device name      : {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("✓ CPU-only mode (pas de GPU détecté)")
            return True
    except Exception as e:
        print(f"✗ ERREUR: {e}")
        return False

def test_yolo():
    """Tester YOLOv8 avec le modèle small."""
    print("\n" + "=" * 60)
    print("  TEST YOLOV8 (modèle small)")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
        
        print("Chargement yolov8s.pt (light)...")
        model = YOLO("yolov8s.pt")
        print(f"✓ Modèle yolov8s chargé")
        
        # Vérifier que c'est bien le small
        print(f"  Paramètres: {sum(p.numel() for p in model.model.parameters()):,}")
        
        print("✓ YOLO OK")
        return True
    except Exception as e:
        print(f"✗ YOLO ERREUR: {e}")
        return False

def test_data():
    """Vérifier que les données sont présentes."""
    print("\n" + "=" * 60)
    print("  TEST DONNÉES")
    print("=" * 60)
    
    data_yaml = Path("data/roboflow/merged/data.yaml")
    if data_yaml.exists():
        print(f"✓ data.yaml trouvé : {data_yaml}")
        return True
    else:
        print(f"✗ data.yaml manquant : {data_yaml}")
        print("  Exécutez : python scripts/download_roboflow_dataset.py --api-key <KEY>")
        return False

def test_config():
    """Vérifier la configuration CPU."""
    print("\n" + "=" * 60)
    print("  TEST CONFIG (CPU)")
    print("=" * 60)
    
    import yaml
    
    cfg_path = Path("config/config.yaml")
    if not cfg_path.exists():
        print(f"✗ config.yaml manquant")
        return False
    
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    detection = cfg["detection"]
    finetune = cfg["finetune"]
    
    print(f"Modèle détection   : {detection['model_name']}")
    print(f"Device             : {detection['device']}")
    print(f"Résolution         : {cfg['video']['resize_width']}x{cfg['video']['resize_height']}")
    print()
    print(f"Modèle fine-tune   : {finetune['base_model']}")
    print(f"Epochs             : {finetune['epochs']}")
    print(f"Imgsz              : {finetune['imgsz']}")
    print(f"Batch size         : {finetune['batch']}")
    print(f"AMP                : {finetune['amp']}")
    
    # Vérifications
    ok = True
    if detection['model_name'] != "yolov8s.pt":
        print(f"⚠ Conseil: utiliser yolov8s pour CPU (actuellement {detection['model_name']})")
        ok = False
    
    if detection['device'] != "cpu":
        print(f"⚠ Avertissement: device n'est pas 'cpu' (actuellement {detection['device']})")
    
    if finetune['amp']:
        print(f"⚠ ERREUR: AMP ne fonctionne pas sur CPU, mettez-le à false")
        ok = False
    
    if finetune['batch'] > 8:
        print(f"⚠ Conseil: réduire batch_size (actuellement {finetune['batch']})")
    
    return ok

def main():
    print("\n[TEST] DIAGNOSTIC RUGBY IA CPU\n")
    
    checks = [
        ("Imports", test_imports()),
        ("PyTorch", test_pytorch()),
        ("YOLOv8", test_yolo()),
        ("Donnees", test_data()),
        ("Config", test_config()),
    ]
    
    print("\n" + "=" * 60)
    print("  RÉSUMÉ")
    print("=" * 60)
    
    for name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:<20} {status}")
    
    all_ok = all(r for _, r in checks)
    
    print("\n" + "=" * 60)
    if all_ok:
        print("  ✓ ENV PRÊT - Vous pouvez lancer le training !")
        print("  Commande :")
        print("  python scripts/finetune_yolo_rugby.py --data data/roboflow/merged/data.yaml")
    else:
        print("  ✗ ERREURS - Voir ci-dessus")
    print("=" * 60 + "\n")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
