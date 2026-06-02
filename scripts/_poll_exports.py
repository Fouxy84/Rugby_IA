"""
Poll freshly-generating Roboflow exports and download the first working one.
Darknet format = same .txt labels as YOLOv8, so we can use it directly.
"""
import time
import requests

API_KEY = "fFkApOIskzteRyI5rvwm"
VERSIONS = [1, 2, 3, 4, 5]
# Formats to try in order; darknet = same label format as yolov8
POLL_FORMATS = ["darknet", "createml", "yolov8", "coco", "yolov5pytorch"]


def try_format(workspace, project, version, fmt, attempts=20, delay=15):
    print(f"\n[{workspace}/{project} v{version}] Polling '{fmt}' export…")
    for i in range(1, attempts + 1):
        r = requests.get(
            f"https://api.roboflow.com/{workspace}/{project}/{version}/{fmt}",
            params={"api_key": API_KEY},
            timeout=20,
        )
        d = r.json()
        link = (d.get("export") or {}).get("link", "")
        progress = d.get("progress")

        if not link:
            print(f"  attempt {i:2d}: progress={progress}")
            time.sleep(delay)
            continue

        # Got a link — check if GCS file actually exists
        resp = requests.get(link, timeout=60, stream=True)
        chunk = next(resp.iter_content(chunk_size=1024), b"")
        is_zip = chunk[:2] == b"PK"
        clen = resp.headers.get("content-length", "?")
        print(f"  attempt {i:2d}: HTTP {resp.status_code}  zip={is_zip}  len={clen}")

        if is_zip:
            return link, resp
        time.sleep(delay)

    return None, None


for version in VERSIONS:
    for fmt in POLL_FORMATS:
        link, resp = try_format(
            "rugby-analysis", "rugby-player-detection", version, fmt,
            attempts=12, delay=10,
        )
        if link:
            print(f"\n✓ FOUND: rugby-analysis/rugby-player-detection v{version} fmt={fmt}")
            print(f"  URL: {link}")
            break
    else:
        continue
    break
else:
    print("\n✗ No working export found for any version/format of rugby-analysis/rugby-player-detection")
