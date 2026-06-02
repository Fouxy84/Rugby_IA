"""Test multiple Roboflow export formats for v1 and v2."""
import time
import requests

API_KEY = "fFkApOIskzteRyI5rvwm"
FORMATS = ["yolov8", "coco", "yolov5pytorch", "createml", "darknet"]


def test_format(workspace, project, version, fmt):
    r = requests.get(
        f"https://api.roboflow.com/{workspace}/{project}/{version}/{fmt}",
        params={"api_key": API_KEY},
        timeout=20,
    )
    d = r.json()
    link = (d.get("export") or {}).get("link", "")
    progress = d.get("progress")
    if not link:
        print(f"  [{fmt}] no link (progress={progress}, keys={list(d.keys())})")
        return None
    # Test actual download
    resp = requests.get(link, timeout=30, stream=True)
    chunk = next(resp.iter_content(chunk_size=1024), b"")
    is_zip = chunk[:2] == b"PK"
    clen = resp.headers.get("content-length", "?")
    print(f"  [{fmt}] HTTP {resp.status_code}  zip={is_zip}  len={clen}  url={link[:60]}")
    return link if is_zip else None


for version in [1, 2, 3, 4, 5]:
    print(f"\n=== Version {version} ===")
    for fmt in FORMATS:
        result = test_format("rugby-analysis", "rugby-player-detection", version, fmt)
        if result:
            print(f"  *** FOUND valid download: {result}")
            break

