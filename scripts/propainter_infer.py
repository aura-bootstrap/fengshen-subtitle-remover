#!/usr/bin/env python3
"""ProPainter sidecar for desub (R5): inpaints one chunk of the band strip.

Input/output exchange is lossless PNG directories written by the Go caller:
  --strip DIR   RGB band frames, %05d.png numbered from 0 over the whole job
  --masks DIR   gray masks (255 = inpaint), same numbering
  --out DIR     repaired frames for the requested range, same numbering
  --start N     first frame index (relative to the job, 0-based)
  --count M     number of frames to process

Only the mask bbox plus vertical context (~1.75x the mask height) reaches the
model (R5.2); the rest of each output frame is the untouched input, so the
caller may composite the whole frame and still affect only masked pixels.

ProPainter itself is NOT bundled: set PROPAINTER_HOME to a checkout of
https://github.com/sczhou/ProPainter with its weights under weights/. Any
setup or inference failure exits non-zero (3) with the reason on stderr, so
the caller degrades the event to the motion tier (R5.6).
"""
import argparse
import os
import subprocess
import sys


def fail(msg):
    print(f"propainter: {msg}", file=sys.stderr)
    sys.exit(3)


def load_frame(path):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        fail(f"cannot read {path}")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    args = ap.parse_args()

    home = os.environ.get("PROPAINTER_HOME", "")
    if not home or not os.path.isdir(home):
        fail("PROPAINTER_HOME not set or not a directory; sidecar unavailable")
    infer_py = os.path.join(home, "inference.py")
    if not os.path.isfile(infer_py):
        fail(f"{infer_py} missing; sidecar unavailable")
    try:
        import cv2  # noqa: F401
        import numpy as np
    except ImportError as e:
        fail(f"import: {e}")

    n = args.count
    frames, masks = [], []
    for k in range(n):
        name = "%05d.png" % (args.start + k)
        frames.append(load_frame(os.path.join(args.strip, name)))
        m = cv2.imread(os.path.join(args.masks, name), cv2.IMREAD_GRAYSCALE)
        if m is None:
            fail(f"cannot read mask {name}")
        masks.append(m)
    h, w = frames[0].shape[:2]

    # Vertical context crop around the union mask bbox (R5.2): the model sees
    # the strip plus ~1.75x the mask height of context on each side, never the
    # whole band. Output frames are pasted back at full size.
    ys = np.zeros(h, dtype=bool)
    for m in masks:
        ys |= m.max(axis=1) > 0
    rows = np.nonzero(ys)[0]
    if len(rows) == 0:
        os.makedirs(args.out, exist_ok=True)
        for k in range(n):
            cv2.imwrite(os.path.join(args.out, "%05d.png" % (args.start + k)), frames[k])
        return
    mh = rows[-1] - rows[0] + 1
    pad = int(mh * 1.75 + 0.5)
    y0 = max(0, rows[0] - pad)
    y1 = min(h, rows[-1] + pad + 1)

    work = os.path.join(args.out, ".work")
    os.makedirs(work, exist_ok=True)
    sub_video = os.path.join(work, "chunk.mp4")
    frames_dir_in = os.path.join(work, "frames")
    mask_dir = os.path.join(work, "mask")
    os.makedirs(frames_dir_in, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    for k in range(n):
        cv2.imwrite(os.path.join(frames_dir_in, "%05d.png" % k), frames[k][y0:y1, :])
        cv2.imwrite(os.path.join(mask_dir, "%05d.png" % k), masks[k][y0:y1, :])
    # Lossless intermediate (R5.1): x264rgb qp 0 keeps every pixel intact.
    ff = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
                         "-framerate", "%.6f" % args.fps,
                         "-i", os.path.join(frames_dir_in, "%05d.png"),
                         "-c:v", "libx264rgb", "-qp", "0", sub_video],
                        capture_output=True, text=True)
    if ff.returncode != 0:
        fail(f"ffmpeg: {ff.stderr[-300:]}")

    model_out = os.path.join(work, "result")
    env = dict(os.environ)
    cmd = [sys.executable, infer_py,
           "--mode", "video_inpainting",
           "--video", sub_video,
           "--mask", mask_dir,
           "--output", model_out]
    if len(frames) > 0:
        cmd += ["--save_frames"]
    r = subprocess.run(cmd, cwd=home, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        fail(f"inference exit {r.returncode}: {(r.stderr or r.stdout)[-400:]}")

    frames_dir = None
    for root, _dirs, files in os.walk(model_out):
        pngs = [f for f in files if f.endswith(".png")]
        if pngs:
            frames_dir = root
            break
    if frames_dir is None:
        fail("inference produced no frames")
    produced = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    if len(produced) < n:
        fail(f"inference produced {len(produced)} frames, want {n}")

    os.makedirs(args.out, exist_ok=True)
    for k in range(n):
        sub = load_frame(os.path.join(frames_dir, produced[k]))
        out = frames[k].copy()
        sh = sub.shape[0]
        out[y0:y0 + sh, :] = sub
        cv2.imwrite(os.path.join(args.out, "%05d.png" % (args.start + k)), out)


if __name__ == "__main__":
    main()
