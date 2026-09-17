#!/usr/bin/env python3
"""OCR sidecar for desub: reads one JSON request from stdin and writes one
JSON response to stdout. Logs and library noise go to stderr only.

Request:  {"input": str, "band_y": int, "band_h": int, "frames": [int]}
Response: {"boxes": [{"frame": int, "rect": {"X","Y","W","H"}, "score": float}],
           "error": str  # present only on failure
          }
Rect coordinates are relative to the band (the caller adds band_y back).
"""
import json
import sys


def fail(msg):
    json.dump({"boxes": [], "error": msg}, sys.stdout)
    sys.exit(0)  # protocol errors are reported in-band, not via exit code


def main():
    try:
        req = json.load(sys.stdin)
    except Exception as e:
        fail(f"bad request: {e}")
    try:
        import cv2
        import easyocr
    except ImportError as e:
        fail(f"import: {e}")

    cap = cv2.VideoCapture(req["input"])
    if not cap.isOpened():
        fail("cannot open input")
    try:
        reader = easyocr.Reader(["ch_sim", "en"], gpu=True, verbose=False)
    except Exception as e:
        cap.release()
        fail(f"easyocr init: {e}")

    by, bh = int(req["band_y"]), int(req["band_h"])
    out = []
    for f in req["frames"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        band = img[by:by + bh, :]
        try:
            results = reader.readtext(band)
        except Exception as e:
            print(f"warn: frame {f}: {e}", file=sys.stderr)
            continue
        for quad, _text, conf in results:
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            x0, x1 = int(min(xs)), int(max(xs))
            y0, y1 = int(min(ys)), int(max(ys))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            out.append({
                "frame": int(f),
                "rect": {"X": x0, "Y": y0, "W": x1 - x0, "H": y1 - y0},
                "score": float(conf),
            })
    cap.release()
    json.dump({"boxes": out}, sys.stdout)


if __name__ == "__main__":
    main()
