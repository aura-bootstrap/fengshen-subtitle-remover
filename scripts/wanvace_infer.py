#!/usr/bin/env python3
"""Wan-VACE sidecar for desub: generative inpainting of one chunk of the band
strip, using the same PNG-directory contract as propainter_infer.py, so the Go
caller selects it with `--propainter-script scripts/wanvace_infer.py`.

  --strip DIR   RGB band frames, %05d.png numbered from 0 over the whole job
  --masks DIR   gray masks (255 = inpaint), same numbering
  --out DIR     repaired frames for the requested range, same numbering
  --start N     first frame index (relative to the job, 0-based)
  --count M     number of frames to process
  --fps F       frame rate (for the lossless intermediate video)

Wan-VACE itself is NOT bundled: set WANVACE_HOME to a checkout of
https://github.com/Wan-Video/Wan2.1 containing generate.py and the VACE
checkpoint directory. Any setup or inference failure exits 3 with the reason
on stderr, so the caller degrades the event to the motion tier.

Differences from the ProPainter sidecar (fixed-size diffusion model):
  - the vertical context crop is resized to WANVACE_SIZE (default 832*480)
    before inference and the result is resized back;
  - the frame count is padded (last frame repeated, black masks) up to the
    next 4k+1 (Wan causal-VAE constraint), >= WANVACE_FRAME_NUM (default 81);
  - VACE needs a mask VIDEO, not a directory: white = regenerate, black = keep.

Env knobs: WANVACE_HOME (required), WANVACE_TASK (default vace-1.3B),
WANVACE_CKPT (default $WANVACE_HOME/Wan2.1-VACE-1.3B), WANVACE_SIZE,
WANVACE_FRAME_NUM, WANVACE_STEPS, WANVACE_PROMPT, WANVACE_NEG_PROMPT,
WANVACE_OFFLOAD (non-empty -> --offload_model True),
WANVACE_T5_CPU (non-empty -> --t5_cpu, T5 text encoder stays on CPU).

STATUS: written against the Wan2.1 VACE CLI contract but NOT integration-tested
(no local GPU). Treat the first real run as a bring-up: check the VACE arg
names and the mask polarity (white = regenerate) against your checkout.
"""
import argparse
import math
import os
import subprocess
import sys


def fail(msg):
    print(f"wanvace: {msg}", file=sys.stderr)
    sys.exit(3)


def load_frame(path):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        fail(f"cannot read {path}")
    return img


def parse_size(s):
    try:
        w, h = s.replace("x", "*").split("*")
        return int(w), int(h)
    except ValueError:
        fail(f"bad WANVACE_SIZE {s!r}, want e.g. 832*480")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    args = ap.parse_args()

    home = os.environ.get("WANVACE_HOME", "")
    if not home or not os.path.isdir(home):
        fail("WANVACE_HOME not set or not a directory; sidecar unavailable")
    gen_py = os.path.join(home, "generate.py")
    if not os.path.isfile(gen_py):
        fail(f"{gen_py} missing; sidecar unavailable")
    task = os.environ.get("WANVACE_TASK", "vace-1.3B")
    ckpt = os.environ.get("WANVACE_CKPT", os.path.join(home, "Wan2.1-VACE-1.3B"))
    if not os.path.isdir(ckpt):
        fail(f"checkpoint dir {ckpt} missing (WANVACE_CKPT)")
    size_w, size_h = parse_size(os.environ.get("WANVACE_SIZE", "832*480"))
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
    crop_h = y1 - y0

    # Wan causal VAE consumes 4k+1 frames; pad by repeating the last frame
    # with a black (keep-everything) mask and drop the padding afterwards.
    min_fn = int(os.environ.get("WANVACE_FRAME_NUM", "81"))
    fn = max(min_fn, 4 * math.ceil(max(n - 1, 0) / 4) + 1)

    work = os.path.join(args.out, ".work-vace")
    frames_dir_in = os.path.join(work, "frames")
    mask_dir = os.path.join(work, "mask")
    os.makedirs(frames_dir_in, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    for k in range(fn):
        src = min(k, n - 1)
        crop = cv2.resize(frames[src][y0:y1, :], (size_w, size_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(frames_dir_in, "%05d.png" % k), crop)
        if k < n:
            mk = cv2.resize(masks[src][y0:y1], (size_w, size_h), interpolation=cv2.INTER_NEAREST)
            mk = np.where(mk > 127, 255, 0).astype(np.uint8)
        else:
            mk = np.zeros((size_h, size_w), dtype=np.uint8)
        cv2.imwrite(os.path.join(mask_dir, "%05d.png" % k), mk)

    sub_video = os.path.join(work, "chunk.mp4")
    ff = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
                         "-framerate", "%.6f" % args.fps,
                         "-i", os.path.join(frames_dir_in, "%05d.png"),
                         "-c:v", "libx264rgb", "-qp", "0", sub_video],
                        capture_output=True, text=True)
    if ff.returncode != 0:
        fail(f"ffmpeg: {ff.stderr[-300:]}")
    mask_video = os.path.join(work, "mask.mp4")
    ff = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
                         "-framerate", "%.6f" % args.fps,
                         "-i", os.path.join(mask_dir, "%05d.png"),
                         "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", mask_video],
                        capture_output=True, text=True)
    if ff.returncode != 0:
        fail(f"ffmpeg mask: {ff.stderr[-300:]}")

    result_video = os.path.join(work, "result.mp4")
    cmd = [sys.executable, gen_py,
           "--task", task,
           "--size", "%d*%d" % (size_w, size_h),
           "--frame_num", str(fn),
           "--ckpt_dir", ckpt,
           "--src_video", sub_video,
           "--src_mask", mask_video,
           "--prompt", os.environ.get("WANVACE_PROMPT", "clean background, no text"),
           "--save_file", result_video]
    if os.environ.get("WANVACE_NEG_PROMPT"):
        cmd += ["--sample_neg_prompt", os.environ["WANVACE_NEG_PROMPT"]]
    if os.environ.get("WANVACE_STEPS"):
        cmd += ["--sample_steps", os.environ["WANVACE_STEPS"]]
    if os.environ.get("WANVACE_OFFLOAD"):
        cmd += ["--offload_model", "True"]
    if os.environ.get("WANVACE_T5_CPU"):
        cmd += ["--t5_cpu"]
    r = subprocess.run(cmd, cwd=home, capture_output=True, text=True)
    if r.returncode != 0:
        fail(f"generate exit {r.returncode}: {(r.stderr or r.stdout)[-400:]}")
    if not os.path.isfile(result_video):
        fail("generate produced no result video")

    out_dir = os.path.join(work, "out_frames")
    os.makedirs(out_dir, exist_ok=True)
    ff = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
                         "-i", result_video, "-start_number", "0",
                         os.path.join(out_dir, "%05d.png")],
                        capture_output=True, text=True)
    if ff.returncode != 0:
        fail(f"ffmpeg decode: {ff.stderr[-300:]}")
    produced = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
    if len(produced) < n:
        fail(f"result has {len(produced)} frames, want {n}")

    for k in range(n):
        sub = load_frame(os.path.join(out_dir, produced[k]))
        if sub.shape[0] != crop_h or sub.shape[1] != w:
            sub = cv2.resize(sub, (w, crop_h), interpolation=cv2.INTER_LINEAR)
        out = frames[k].copy()
        out[y0:y1, :] = sub
        cv2.imwrite(os.path.join(args.out, "%05d.png" % (args.start + k)), out)


if __name__ == "__main__":
    main()
