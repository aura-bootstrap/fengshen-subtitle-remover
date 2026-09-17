package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/engine"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/events"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/ffx"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/manifest"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/propainter"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/route"
)

// cmdRerun re-repairs one manifest segment and splices it back into the
// existing output (R8.2, R8.3), then appends the rerun record (R8.5).
func cmdRerun(args []string) error {
	fs := flag.NewFlagSet("rerun", flag.ExitOnError)
	fs.Usage = usage
	manifestPath := fs.String("manifest", "", "run manifest written by remove --manifest (required)")
	segment := fs.Int("segment", -1, "segment index to re-run (required)")
	final := fs.String("final", "", "existing output to splice into (required)")
	out := fs.String("o", "", "output file (required)")
	crf := fs.Int("crf", 17, "x264 CRF for the re-encoded segment")
	preset := fs.String("preset", "medium", "x264 preset")
	neighbors := fs.Int("neighbors", 6, "temporal radius for pixel fill")
	noMotion := fs.Bool("no-motion", false, "disable motion-compensated pixel transfer")
	alphaOn := fs.Bool("alpha", true, "unmix semi-transparent bars")
	grainOn := fs.Bool("grain", false, "texture-match the repaired area")
	painterOn := fs.Bool("propainter", false, "enable the ProPainter sidecar")
	painterScript := fs.String("propainter-script", "scripts/propainter_infer.py", "path to the ProPainter sidecar script")
	keepWork := fs.Bool("keep-work", false, "keep the temporary segment videos")
	pos, err := parseArgs(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 1 || *manifestPath == "" || *segment < 0 || *final == "" || *out == "" {
		return fmt.Errorf("usage: desub rerun <video> --manifest m.json --segment K --final existing.mp4 -o out.mp4")
	}
	input := pos[0]

	m, err := manifest.Read(*manifestPath)
	if err != nil {
		return err
	}
	info, err := ffx.Probe(input)
	if err != nil {
		return err
	}
	if err := m.Verify(input, info); err != nil {
		return err
	}
	seg, err := m.SegmentOf(*segment)
	if err != nil {
		return err
	}
	n := seg.EndF - seg.StartF + 1
	if len(seg.Masks) != n {
		return fmt.Errorf("manifest: segment %d has %d masks for %d frames", *segment, len(seg.Masks), n)
	}

	work, err := os.MkdirTemp("", "desub-rerun-*")
	if err != nil {
		return err
	}
	if !*keepWork {
		defer os.RemoveAll(work)
	}
	segIn := filepath.Join(work, "seg_in.mp4")
	segOut := filepath.Join(work, "seg_out.mp4")

	// Lossless segment cut from the original input.
	if out1, err := exec.Command(ffx.FFmpeg(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
		"-ss", fmt.Sprintf("%.6f", seg.Start),
		"-i", input,
		"-frames:v", fmt.Sprint(n),
		"-an", "-c:v", "libx264", "-qp", "0", "-preset", "veryfast", segIn).CombinedOutput(); err != nil {
		return fmt.Errorf("segment cut: %v: %s", err, tail(out1, 300))
	}
	cuts, err := ffx.SceneCuts(segIn, 0.35)
	if err != nil {
		return err
	}

	overrides := map[string]any{}
	var painter engine.Painter
	if *painterOn {
		cl, cerr := propainter.NewClient(*painterScript, 0)
		if cerr != nil {
			fmt.Fprintf(os.Stderr, "warn: propainter unavailable: %v\n", cerr)
		} else {
			painter = cl
		}
	}
	// Re-dispatch only this segment from the recorded verdict: the engine is
	// taken from the manifest, tier kept for the report.
	decisions := []route.Decision{{
		Event:  events.Event{StartF: 0, EndF: n - 1, Start: 0, End: float64(n) / info.FPS},
		Tier:   route.Tier(seg.Tier),
		Engine: seg.Engine,
		Risk:   seg.Risk,
	}}
	frep, err := engine.RunFill(engine.FillOptions{
		TemporalOptions: engine.TemporalOptions{
			Input: segIn, Output: segOut, W: info.W, H: info.H,
			BandY: m.Band.Y, BandH: m.Band.H, FPS: info.FPS,
			Masks: seg.Masks, Cuts: cuts,
			CRF: *crf, Preset: *preset, Neighbors: *neighbors,
			Motion: !*noMotion, Alpha: *alphaOn, CharH: m.CharH,
			Grain:    *grainOn,
			EncColor: info.ColorEncodeArgs(),
		},
		Events:    []events.Event{decisions[0].Event},
		Decisions: decisions,
		Painter:   painter,
		Log:       os.Stderr,
	})
	if err != nil {
		return err
	}
	overrides["crf"] = *crf
	overrides["motion"] = !*noMotion
	overrides["alpha"] = *alphaOn
	overrides["grain"] = *grainOn
	if len(frep.Fallback) > 0 {
		overrides["fallback"] = frep.Fallback
	}

	// Splice the re-repaired segment over the existing output; every other
	// frame stays as it was (R8.3).
	splice := []string{"-hide_banner", "-nostdin", "-y", "-loglevel", "error",
		"-i", *final, "-i", segOut,
		"-filter_complex", fmt.Sprintf("[0:v][1:v]overlay=0:0:enable='between(n,%d,%d)'[v]", seg.StartF, seg.EndF),
		"-map", "[v]", "-map", "0:a?"}
	splice = append(splice, info.ColorEncodeArgs()...)
	splice = append(splice,
		"-c:v", "libx264", "-crf", fmt.Sprint(*crf), "-preset", *preset,
		"-c:a", "copy", "-movflags", "+faststart", *out)
	if out1, err := exec.Command(ffx.FFmpeg(), splice...).CombinedOutput(); err != nil {
		return fmt.Errorf("splice: %v: %s", err, tail(out1, 300))
	}

	m.Reruns = append(m.Reruns, manifest.Rerun{Segment: *segment, Time: time.Now().UTC(), Params: overrides})
	if err := manifest.Write(*manifestPath, m); err != nil {
		return err
	}
	rep := map[string]any{
		"segment":   *segment,
		"start_f":   seg.StartF,
		"end_f":     seg.EndF,
		"frames":    n,
		"output":    *out,
		"reruns":    len(m.Reruns),
		"fallback":  frep.Fallback,
		"keep_work": *keepWork,
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(rep)
}

func tail(b []byte, n int) string {
	s := string(b)
	if len(s) > n {
		return s[len(s)-n:]
	}
	return s
}
