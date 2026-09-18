#!/usr/bin/env python3
"""VLM QC sidecar for desub: re-judges verify-stage residue boxes.

Input:
  --video PATH     the repaired output video (residue coords are absolute)
  --residue PATH   JSON list of {"frame","time","x","y","w","h"}
  --out PATH       verdict JSON out: same entries plus
                   {"residue": bool, "confidence": 0..1, "note": str?}
  --endpoint URL   OpenAI-compatible base URL (default: local ollama)
  --model NAME     VLM model name (default qwen2.5vl:7b)
  --api-key KEY    default "ollama" (local servers ignore it)
  --pad PX         crop padding around the box (default 16)
  --max N          cap on reviewed boxes (default 200)
  --timeout SEC    per-request timeout (default 60)

The VLM itself is NOT bundled: point --endpoint at any OpenAI-compatible
server (ollama / vLLM / LM Studio) hosting a vision model. If the endpoint
is unreachable on the first request the script exits 3 so the caller
degrades with a warning instead of blocking the pipeline.
"""
import argparse
import base64
import json
import os
import sys
import urllib.request
import urllib.error


def fail(msg):
    print(f"vlm-qc: {msg}", file=sys.stderr)
    sys.exit(3)


PROMPT = (
    "This crop comes from a video frame where hard subtitles were removed. "
    "Decide whether it still contains visible leftover subtitle text "
    "(full or partial glyphs, stroke fragments, ghost text edges). "
    "Background textures (ropes, fabric, hands, faces) are NOT residue. "
    'Reply with JSON only: {"residue": true|false, "confidence": 0.0-1.0}'
)


def post_chat(args, png_b64):
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a strict video QC inspector."},
            {"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + png_b64}},
            ]},
        ],
        "temperature": 0,
        "max_tokens": 64,
    }
    req = urllib.request.Request(
        args.endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + (args.api_key or "ollama")},
        method="POST")
    with urllib.request.urlopen(req, timeout=args.timeout) as r:
        payload = json.loads(r.read().decode())
    return payload["choices"][0]["message"]["content"]


def parse_verdict(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:]
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None, None
    try:
        d = json.loads(t[i:j + 1])
    except json.JSONDecodeError:
        return None, None
    return bool(d.get("residue")), float(d.get("confidence", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--residue", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoint",
                    default="http://host.docker.internal:11434/v1")
    ap.add_argument("--model", default="qwen2.5vl:7b")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--pad", type=int, default=16)
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--timeout", type=float, default=60)
    args = ap.parse_args()

    try:
        import cv2
    except ImportError as e:
        fail(f"import: {e}")

    with open(args.residue) as f:
        boxes = json.load(f)
    if not isinstance(boxes, list):
        fail("residue JSON must be a list")
    boxes = boxes[: args.max]
    if not boxes:
        with open(args.out, "w") as f:
            json.dump([], f)
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        fail(f"cannot open {args.video}")
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = []
    for k, b in enumerate(boxes):
        verdict = dict(b)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(b["frame"]))
            ok, img = cap.read()
            if not ok:
                raise RuntimeError("frame %s unreadable" % b["frame"])
            x0 = max(0, int(b["x"]) - args.pad)
            y0 = max(0, int(b["y"]) - args.pad)
            x1 = min(fw, int(b["x"]) + int(b["w"]) + args.pad)
            y1 = min(fh, int(b["y"]) + int(b["h"]) + args.pad)
            crop = img[y0:y1, x0:x1]
            ok, png = cv2.imencode(".png", crop)
            if not ok:
                raise RuntimeError("png encode failed")
            text = post_chat(args, base64.b64encode(png.tobytes()).decode())
            res, conf = parse_verdict(text)
            if res is None:
                verdict["residue"] = None
                verdict["note"] = "unparseable VLM reply"
            else:
                verdict["residue"] = res
                verdict["confidence"] = conf
        except urllib.error.URLError as e:
            cap.release()
            if k == 0:
                fail(f"endpoint unreachable: {e}")
            verdict["residue"] = None
            verdict["note"] = f"endpoint: {e}"
        except Exception as e:  # per-box errors must not abort the batch
            verdict["residue"] = None
            verdict["note"] = str(e)
        out.append(verdict)
    cap.release()

    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
