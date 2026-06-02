"""Quick probe of Roboflow API endpoints to understand available data."""
import json
import requests

API_KEY = "fFkApOIskzteRyI5rvwm"
BASE = "https://api.roboflow.com"


def get(path, **params):
    params["api_key"] = API_KEY
    r = requests.get(f"{BASE}/{path}", params=params, timeout=20)
    return r.json()


# 1. Version metadata
print("=== rugby-analysis/rugby-player-detection/1 (version metadata) ===")
data = get("rugby-analysis/rugby-player-detection/1")
print(json.dumps({k: str(type(v).__name__) for k, v in data.items()}, indent=2))

# 2. Try images endpoint
print("\n=== /images endpoint ===")
try:
    imgs = get("rugby-analysis/rugby-player-detection/1/images", split="train", offset=0, limit=3)
    print(json.dumps(imgs, indent=2)[:3000])
except Exception as e:
    print(f"Error: {e}")

# 3. Try COCO format export (different format may have valid GCS)
print("\n=== COCO format export link ===")
try:
    coco = get("rugby-analysis/rugby-player-detection/1/coco")
    print(json.dumps(coco, indent=2)[:1000])
except Exception as e:
    print(f"Error: {e}")

# 4. Try VOC format
print("\n=== Pascal VOC format export link ===")
try:
    voc = get("rugby-analysis/rugby-player-detection/1/voc")
    print(json.dumps(voc, indent=2)[:1000])
except Exception as e:
    print(f"Error: {e}")
