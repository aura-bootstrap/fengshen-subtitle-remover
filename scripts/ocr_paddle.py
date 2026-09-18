#!/usr/bin/env python3
"""PaddleOCR sidecar for desub: drop-in replacement for ocr_boxes.py with the
same stdin/stdout JSON contract. Select it with `--ocr-script
scripts/ocr_paddle.py` (and DESUB_IMAGE=desub:cu124-paddle).

Request:  {"input": str, "band_y": int, "band_h": int, "frames": [int]}
Response: {"boxes": [{"frame": int, "rect": {"X","Y","W","H"}, "score": float}],
           "error": str  # present only on failure
          }
Rect coordinates are relative to the band (the caller adds band_y back).

Handles both paddleocr 2.x (`ocr.ocr(img, cls=False)`) and 3.x
(`ocr.predict(img)`) result shapes.
"""
import json
import os
import sys

# paddle 3.3.1 oneDNN backend crashes on this CPU (NotImplementedError
# ConvertPirAttribute2RuntimeAttribute); must be set before paddle imports.
os.environ.setdefault("FLAGS_use_mkldnn", "0")


def fail(msg):
    json.dump({"boxes": [], "error": msg}, sys.stdout)
    sys.exit(0)  # protocol errors are reported in-band, not via exit code


def make_reader():
    from paddleocr import PaddleOCR
    try:
        return PaddleOCR(use_angle_cls=False, lang="ch", use_gpu=False)
    except Exception:
        return PaddleOCR(lang="ch", device="cpu", enable_mkldnn=False)


def detect(reader, img):
    """Yield (quad, conf) pairs. Prefer the 3.x predict() API — its ocr()
    alias rejects the 2.x cls kwarg, so hasattr checks are unreliable."""
    predict = getattr(reader, "predict", None)
    if callable(predict):
        try:
            results = predict(img)
        except TypeError:
            results = None
        if results is not None:
            for res in results or []:
                polys = res.get("rec_polys", res.get("dt_polys", []))
                scores = res.get("rec_scores", [])
                for quad, conf in zip(polys, scores):
                    yield quad, conf
            return
    for line in reader.ocr(img, cls=False) or []:
        for quad, (_text, conf) in line or []:
            yield quad, conf


def main():
    try:
        req = json.load(sys.stdin)
    except Exception as e:
        fail(f"bad request: {e}")
    try:
        import cv2  # noqa: F401
        import numpy as np  # noqa: F401
    except ImportError as e:
        fail(f"import: {e}")

    cap = cv2.VideoCapture(req["input"])
    if not cap.isOpened():
        fail("cannot open input")
    try:
        reader = make_reader()
    except Exception as e:
        cap.release()
        fail(f"paddleocr init: {e}")

    by, bh = int(req["band_y"]), int(req["band_h"])
    out = []
    for f in req["frames"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        band = img[by:by + bh, :]
        try:
            for quad, conf in detect(reader, band):
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
        except Exception as e:
            print(f"warn: frame {f}: {e}", file=sys.stderr)
    cap.release()
    json.dump({"boxes": out}, sys.stdout)


if __name__ == "__main__":
    main()
