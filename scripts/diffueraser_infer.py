#!/usr/bin/env python3
"""DiffuEraser sidecar for desub: diffusion-tier inpainting of one chunk of
the band strip, same PNG-directory contract as propainter_infer.py, so the Go
caller selects it with `--propainter-script scripts/diffueraser_infer.py`.

  --strip DIR   RGB band frames, %05d.png numbered from 0 over the whole job
  --masks DIR   gray masks (255 = inpaint), same numbering
  --out DIR     repaired frames for the requested range, same numbering
  --start N     first frame index (relative to the job, 0-based)
  --count M     number of frames to process
  --fps F       frame rate (chunk and mask videos must share it exactly)

DiffuEraser itself is NOT bundled: set DIFFUERASER_HOME to a checkout of
https://github.com/lixiaowen-xw/DiffuEraser with weights under weights/
(HF lixiaowen/diffuEraser + sd-1.5 / PCM_Weights / propainter / sd-vae-ft-mse).

Env knobs: DIFFUERASER_HOME (required), DIFFUERASER_ENTRY (default
run_diffueraser.py), DIFFUERASER_EXTRA_ARGS (appended verbatim).

STATUS: written against the upstream README but NOT integration-tested (needs
GPU: 12G@640x360 … 33G@1280x720). Bring-up notes:
  - upstream run_diffueraser.py has no argparse: it reads module-level
    input_video / input_mask variables. This sidecar calls ENTRY with
    --input_video/--input_mask/--output_dir; if your checkout lacks them,
    add the 6-line argparse shim shown in doc/bringup notes or set
    DIFFUERASER_ENTRY to your own wrapper.
  - mask video polarity: white = region to erase (same as ProPainter);
    verify on the first run.
  - video and mask must be mp4 with identical fps (upstream misalignment
    guard); we encode both from the same frame count with -r == --fps.
"""
import argparse
import os
import subprocess
import sys


def fail(msg):
    print(f"diffueraser: {msg}", file=sys.stderr)
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

    home = os.environ.get("DIFFUERASER_HOME", "")
    if not home or not os.path.isdir(home):
        fail("DIFFUERASER_HOME not set or not a directory; sidecar unavailable")
    entry = os.path.join(home, os.environ.get("DIFFUERASER_ENTRY", "run_diffueraser.py"))
    if not os.path.isfile(entry):
        fail(f"{entry} missing; sidecar unavailable")
    try:
        import cv2
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

    # Same vertical-context crop as the ProPainter sidecar (R5.2).
    ys = np.zeros(h, dtype=bool)
    for m in masks:
        ys |= m.max(axis=1) > 0
    rows = np.nonzero(ys)[0]
    os.makedirs(args.out, exist_ok=True)
    if len(rows) == 0:
        for k in range(n):
            cv2.imwrite(os.path.join(args.out, "%05d.png" % (args.start + k)), frames[k])
        return
    mh = rows[-1] - rows[0] + 1
    pad = int(mh * 1.75 + 0.5)
    y0 = max(0, rows[0] - pad)
    y1 = min(h, rows[-1] + pad + 1)

    work = os.path.join(args.out, ".work-diffueraser")
    frames_dir_in = os.path.join(work, "frames")
    mask_dir = os.path.join(work, "mask")
    os.makedirs(frames_dir_in, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    for k in range(n):
        cv2.imwrite(os.path.join(frames_dir_in, "%05d.png" % k), frames[k][y0:y1, :])
        mk = np.where(masks[k][y0:y1] > 127, 255, 0).astype(np.uint8)
        cv2.imwrite(os.path.join(mask_dir, "%05d.png" % k), mk)

    fps = "%.6f" % args.fps
    sub_video = os.path.join(work, "chunk.mp4")
    ff = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
                         "-framerate", fps, "-i", os.path.join(frames_dir_in, "%05d.png"),
                         "-c:v", "libx264rgb", "-qp", "0", "-r", fps, sub_video],
                        capture_output=True, text=True)
    if ff.returncode != 0:
        fail(f"ffmpeg: {ff.stderr[-300:]}")
    mask_video = os.path.join(work, "mask.mp4")
    ff = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
                         "-framerate", fps, "-i", os.path.join(mask_dir, "%05d.png"),
                         "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p",
                         "-r", fps, mask_video],
                        capture_output=True, text=True)
    if ff.returncode != 0:
        fail(f"ffmpeg mask: {ff.stderr[-300:]}")

    result_dir = os.path.join(work, "results")
    os.makedirs(result_dir, exist_ok=True)
    cmd = [sys.executable, entry,
           "--input_video", sub_video,
           "--input_mask", mask_video,
           "--output_dir", result_dir]
    extra = os.environ.get("DIFFUERASER_EXTRA_ARGS", "")
    if extra:
        cmd += extra.split()
    r = subprocess.run(cmd, cwd=home, capture_output=True, text=True)
    if r.returncode != 0:
        fail(f"inference exit {r.returncode}: {(r.stderr or r.stdout)[-400:]}")
    produced_video = None
    for root, _dirs, files in os.walk(result_dir):
        mp4s = [f for f in files if f.endswith(".mp4")]
        if mp4s:
            produced_video = os.path.join(root, sorted(mp4s)[0])
            break
    if produced_video is None:
        fail("inference produced no mp4; if upstream ENTRY has no argparse, "
             "add --input_video/--input_mask/--output_dir or set DIFFUERASER_ENTRY")

    out_dir = os.path.join(work, "out_frames")
    os.makedirs(out_dir, exist_ok=True)
    ff = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
                         "-i", produced_video, "-start_number", "0",
                         os.path.join(out_dir, "%05d.png")],
                        capture_output=True, text=True)
    if ff.returncode != 0:
        fail(f"ffmpeg decode: {ff.stderr[-300:]}")
    produced = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
    if len(produced) < n:
        fail(f"result has {len(produced)} frames, want {n}")

    crop_h = y1 - y0
    for k in range(n):
        sub = load_frame(os.path.join(out_dir, produced[k]))
        if sub.shape[0] != crop_h or sub.shape[1] != w:
            sub = cv2.resize(sub, (w, crop_h), interpolation=cv2.INTER_LINEAR)
        out = frames[k].copy()
        out[y0:y1, :] = sub
        cv2.imwrite(os.path.join(args.out, "%05d.png" % (args.start + k)), out)


if __name__ == "__main__":
    main()
