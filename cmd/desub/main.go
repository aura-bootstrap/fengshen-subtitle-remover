// Command desub removes hardcoded subtitles from a video (L0 baseline:
// band detection + temporal/delogo repair + single re-encode).
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/detect"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/ffx"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/pipeline"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/subs"
)

const version = "0.1.0"

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "probe":
		err = cmdProbe(os.Args[2:])
	case "detect":
		err = cmdDetect(os.Args[2:])
	case "remove":
		err = cmdRemove(os.Args[2:])
	case "rerun":
		err = cmdRerun(os.Args[2:])
	case "risk":
		err = cmdRisk(os.Args[2:])
	case "synth":
		err = cmdSynth(os.Args[2:])
	case "eval":
		err = cmdEval(os.Args[2:])
	case "verify":
		err = cmdVerify(os.Args[2:])
	case "version":
		fmt.Println("desub " + version)
		return
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `desub `+version+` — hardcoded subtitle remover (L0)

usage:
  desub probe   <video>
  desub detect  <video> [--band 0.58] [--dump DIR] [--dump-limit 12] [--events] [--json FILE]
  desub remove  <video> -o <out.mp4> [--engine temporal|delogo] [--band 0.58] [--crf 17]
                [--preset medium] [--max-gap 3] [--scene-thr 0.35] [--neighbors 6]
                [--pad 4] [--no-motion] [--alpha] [--propainter] [--grain]
                [--manifest FILE] [--risk-list FILE] [--risk-coverage 0.35]
                [--dump DIR] [--verify=false]
  desub rerun   <video> --manifest m.json --segment K --final existing.mp4 -o out.mp4
                [--crf 17] [--no-motion] [--alpha=true] [--grain] [--propainter]
  desub risk    --manifest m.json [--timecodes] [--risk-coverage 0.35]
  desub synth   --src clean.mp4 -o burned.mp4 --gt-mask mask.mkv [--cues cues.json] [--box]
  desub eval    --out repaired.mp4 --gt clean.mp4 --gt-mask mask.mkv [--burned burned.mp4]
  desub verify  <video> [--band 0.58]

environment:
  DESUB_FFMPEG / DESUB_FFPROBE  override binary paths
`)
}

func addBandFlags(fs *flag.FlagSet) *float64 {
	return fs.Float64("band", 0.58, "subtitle band start as fraction of frame height")
}

// parseArgs parses flags wherever they appear, returning positional args,
// so "desub remove in.mp4 -o out.mp4" works like users expect.
func parseArgs(fs *flag.FlagSet, args []string) ([]string, error) {
	var positional, flags []string
	for i := 0; i < len(args); i++ {
		a := args[i]
		if len(a) > 1 && strings.HasPrefix(a, "-") {
			flags = append(flags, a)
			if !strings.Contains(a, "=") {
				if f := fs.Lookup(strings.TrimLeft(a, "-")); f != nil {
					if _, isBool := f.Value.(interface{ IsBoolFlag() bool }); !isBool && i+1 < len(args) {
						i++
						flags = append(flags, args[i])
					}
				}
			}
			continue
		}
		positional = append(positional, a)
	}
	if err := fs.Parse(flags); err != nil {
		return nil, err
	}
	return positional, nil
}

func cmdProbe(args []string) error {
	fs := flag.NewFlagSet("probe", flag.ExitOnError)
	fs.Usage = usage
	pos, err := parseArgs(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 1 {
		return fmt.Errorf("usage: desub probe <video>")
	}
	info, err := ffx.Probe(pos[0])
	if err != nil {
		return err
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(info); err != nil {
		return err
	}
	if info.SubtitleStreams > 0 {
		fmt.Fprintf(os.Stderr, "hint: %d soft subtitle stream(s) — `ffmpeg -i in -c copy -sn out.mp4` removes them losslessly\n", info.SubtitleStreams)
	}
	return nil
}

func cmdDetect(args []string) error {
	fs := flag.NewFlagSet("detect", flag.ExitOnError)
	fs.Usage = usage
	bandFrac := addBandFlags(fs)
	dump := fs.String("dump", "", "dir for preview PNGs")
	dumpLimit := fs.Int("dump-limit", 12, "max preview PNGs")
	dumpStride := fs.Int("dump-stride", 1, "dump every Nth detected frame")
	showEvents := fs.Bool("events", false, "include event list in output")
	jsonOut := fs.String("json", "", "write full report JSON to file")
	maxGap := fs.Int("max-gap", 3, "max gap frames inside one event")
	sceneThr := fs.Float64("scene-thr", 0.35, "scene cut threshold (0 disables)")
	closeGap := fs.Int("close-gap", 24, "temporal closing: merge same-position events closer than this (frames)")
	closeOvr := fs.Float64("close-overlap", 0.5, "temporal closing: min intersection-over-smaller-box to merge adjacent events")
	edgePad := fs.Int("edge-pad", 2, "pad each event by N frames to cover fade-in/out residue")
	ocrOn := fs.Bool("ocr", false, "fuse easyocr sidecar detections into the masks")
	ocrScript := fs.String("ocr-script", "scripts/ocr_boxes.py", "path to the OCR sidecar script")
	ocrStride := fs.Int("ocr-stride", 12, "OCR every Nth frame")
	alphaOn := fs.Bool("alpha", false, "detect semi-transparent backdrop bars and report bars[]")
	pos, err := parseArgs(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 1 {
		return fmt.Errorf("usage: desub detect <video>")
	}
	input := pos[0]
	if *dump != "" {
		if err := os.MkdirAll(*dump, 0o755); err != nil {
			return err
		}
	}
	rep, err := pipeline.Run(pipeline.Options{
		Input: input, Engine: "temporal", BandStart: *bandFrac,
		MaxGap: *maxGap, SceneThreshold: *sceneThr,
		CloseGap: *closeGap, CloseOverlap: *closeOvr, EdgePad: *edgePad,
		OCR: *ocrOn, OCRScript: *ocrScript, OCRStride: *ocrStride,
		Alpha:   *alphaOn,
		DumpDir: *dump, DumpLimit: *dumpLimit,
		DumpStride: *dumpStride, Log: os.Stderr,
	})
	if err != nil {
		return err
	}
	if !*showEvents {
		rep.Events = nil
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(rep); err != nil {
		return err
	}
	if *jsonOut != "" {
		f, err := os.Create(*jsonOut)
		if err != nil {
			return err
		}
		defer f.Close()
		e2 := json.NewEncoder(f)
		e2.SetIndent("", "  ")
		return e2.Encode(rep)
	}
	return nil
}

func cmdRemove(args []string) error {
	fs := flag.NewFlagSet("remove", flag.ExitOnError)
	fs.Usage = usage
	out := fs.String("o", "", "output file (required)")
	engine := fs.String("engine", "temporal", "temporal|delogo")
	bandFrac := addBandFlags(fs)
	crf := fs.Int("crf", 17, "x264 CRF")
	preset := fs.String("preset", "medium", "x264 preset")
	maxGap := fs.Int("max-gap", 3, "max gap frames inside one event")
	sceneThr := fs.Float64("scene-thr", 0.35, "scene cut threshold (0 disables)")
	neighbors := fs.Int("neighbors", 6, "temporal radius for pixel fill")
	pad := fs.Int("pad", 4, "extra padding for delogo boxes (px)")
	noMotion := fs.Bool("no-motion", false, "disable motion-compensated pixel transfer (temporal engine)")
	closeGap := fs.Int("close-gap", 24, "temporal closing: merge same-position events closer than this (frames)")
	closeOvr := fs.Float64("close-overlap", 0.5, "temporal closing: min intersection-over-smaller-box to merge adjacent events")
	edgePad := fs.Int("edge-pad", 2, "pad each event by N frames to cover fade-in/out residue")
	ocrOn := fs.Bool("ocr", false, "fuse easyocr sidecar detections into the masks")
	ocrScript := fs.String("ocr-script", "scripts/ocr_boxes.py", "path to the OCR sidecar script")
	ocrStride := fs.Int("ocr-stride", 12, "OCR every Nth frame")
	alphaOn := fs.Bool("alpha", false, "unmix semi-transparent backdrop bars instead of inpainting them")
	painterOn := fs.Bool("propainter", false, "enable the ProPainter sidecar for generative-tier events")
	painterScript := fs.String("propainter-script", "scripts/propainter_infer.py", "path to the ProPainter sidecar script")
	painterHome := fs.String("propainter-home", "", "PROPAINTER_HOME checkout dir for the sidecar (empty: inherit env)")
	ppMaskDilation := fs.Int("pp-mask-dilation", 8, "ProPainter mask dilation px (validated: 8 removes stroke-halo bleed)")
	ppRaftIter := fs.Int("pp-raft-iter", 32, "ProPainter RAFT iterations (validated: 32)")
	ppNeighbor := fs.Int("pp-neighbor-length", 20, "ProPainter local neighbor length (validated: 20)")
	grainOn := fs.Bool("grain", false, "match repaired-area texture (noise, chroma, blockiness) to the source")
	forceEngine := fs.String("force-engine", "", "override routing for every event: motion|propainter")
	manifestOut := fs.String("manifest", "", "write a run manifest for segment reruns")
	riskList := fs.String("risk-list", "", "write the high-risk segment list as JSON")
	riskCov := fs.Float64("risk-coverage", 0.35, "coverage threshold below which an event is flagged high-risk")
	dump := fs.String("dump", "", "dir for detection preview PNGs")
	dumpLimit := fs.Int("dump-limit", 12, "max preview PNGs")
	dumpStride := fs.Int("dump-stride", 1, "dump every Nth detected frame")
	verify := fs.Bool("verify", true, "re-run detection on output")
	pos, err := parseArgs(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 1 || *out == "" {
		return fmt.Errorf("usage: desub remove <video> -o <output.mp4> [flags]")
	}
	switch *forceEngine {
	case "", "motion", "propainter":
	default:
		return fmt.Errorf("force-engine: unknown engine %q (want motion|propainter)", *forceEngine)
	}
	if *dump != "" {
		if err := os.MkdirAll(*dump, 0o755); err != nil {
			return err
		}
	}
	rep, err := pipeline.Run(pipeline.Options{
		Input: pos[0], Output: *out, Engine: *engine, BandStart: *bandFrac,
		CRF: *crf, Preset: *preset, MaxGap: *maxGap, SceneThreshold: *sceneThr,
		Neighbors: *neighbors, Pad: *pad, Motion: !*noMotion,
		CloseGap: *closeGap, CloseOverlap: *closeOvr, EdgePad: *edgePad,
		OCR: *ocrOn, OCRScript: *ocrScript, OCRStride: *ocrStride,
		Alpha:      *alphaOn,
		ProPainter: *painterOn, PainterScript: *painterScript,
		PainterHome: *painterHome, PainterMaskDilation: *ppMaskDilation,
		PainterRaftIter: *ppRaftIter, PainterNeighborLength: *ppNeighbor,
		Grain: *grainOn, ForceEngine: *forceEngine, ManifestPath: *manifestOut,
		RiskListPath: *riskList, RiskCoverage: *riskCov,
		DumpDir: *dump, DumpLimit: *dumpLimit,
		DumpStride: *dumpStride, Verify: *verify, Log: os.Stderr,
	})
	if err != nil {
		return err
	}
	rep.Events = nil
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(rep)
}

func cmdVerify(args []string) error {
	fs := flag.NewFlagSet("verify", flag.ExitOnError)
	fs.Usage = usage
	bandFrac := addBandFlags(fs)
	pos, err := parseArgs(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 1 {
		return fmt.Errorf("usage: desub verify <video>")
	}
	input := pos[0]
	info, err := ffx.Probe(input)
	if err != nil {
		return err
	}
	b := pipeline.ComputeBand(info.H, *bandFrac)
	params := detect.DefaultParams(info.H)
	_, _, det, err := subs.Measure(input, info.W, b, params, false, "", 0, 1)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(det); err != nil {
		return err
	}
	if det.TextFrames > 0 {
		fmt.Fprintf(os.Stderr, "residual text in %d/%d frames (%d boxes)\n", det.TextFrames, det.Frames, det.Boxes)
		os.Exit(3)
	}
	return nil
}
