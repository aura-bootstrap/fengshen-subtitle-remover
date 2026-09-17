package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/detect"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/eval"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/ffx"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/pipeline"
)

// cmdSynth burns programmatic subtitles into a clean source with an exact GT
// mask (R11.1): desub synth --src clean.mp4 -o burned.mp4 --gt-mask mask.mkv
func cmdSynth(args []string) error {
	fs := flag.NewFlagSet("synth", flag.ExitOnError)
	fs.Usage = usage
	src := fs.String("src", "", "clean source video (required)")
	out := fs.String("o", "", "burned output video (required)")
	maskOut := fs.String("gt-mask", "", "ground-truth mask video, FFV1 in .mkv (required)")
	cuesFile := fs.String("cues", "", "JSON file with [{start,end,text}]; default: one full-duration cue")
	text := fs.String("text", "这是一句测试字幕", "cue text when --cues is not given")
	font := fs.String("font", "", "font file (default: first Noto CJK / msyh found)")
	size := fs.Int("size", 0, "font size (default: height/15)")
	y := fs.Int("y", 0, "text block top (default: 82% of height)")
	box := fs.Bool("box", false, "draw a semi-transparent backdrop bar")
	pos, err := parseArgs(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 0 || *src == "" || *out == "" || *maskOut == "" {
		return fmt.Errorf("usage: desub synth --src clean.mp4 -o burned.mp4 --gt-mask mask.mkv [--cues cues.json] [--box]")
	}
	var cues []eval.Cue
	if *cuesFile != "" {
		b, err := os.ReadFile(*cuesFile)
		if err != nil {
			return err
		}
		if err := json.Unmarshal(b, &cues); err != nil {
			return fmt.Errorf("cues: %v", err)
		}
	} else {
		info, err := ffx.Probe(*src)
		if err != nil {
			return err
		}
		cues = []eval.Cue{{Start: 0, End: info.Duration, Text: *text}}
	}
	return eval.BurnIn(*src, *out, *maskOut, cues, eval.Style{
		FontFile: *font, FontSize: *size, Y: *y, Box: *box,
	})
}

// cmdEval scores a removal output against the clean source (R11.2..R11.4):
// desub eval --out repaired.mp4 --gt clean.mp4 --gt-mask mask.mkv
func cmdEval(args []string) error {
	fs := flag.NewFlagSet("eval", flag.ExitOnError)
	fs.Usage = usage
	out := fs.String("out", "", "removal output (required)")
	gt := fs.String("gt", "", "clean source (required)")
	maskIn := fs.String("gt-mask", "", "ground-truth mask video (required)")
	burned := fs.String("burned", "", "burned input; enables mask recall against the detector")
	bandFrac := addBandFlags(fs)
	jsonOut := fs.String("json", "", "write metrics JSON to file")
	pos, err := parseArgs(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 0 || *out == "" || *gt == "" || *maskIn == "" {
		return fmt.Errorf("usage: desub eval --out repaired.mp4 --gt clean.mp4 --gt-mask mask.mkv [--burned burned.mp4] [--json m.json]")
	}
	info, err := ffx.Probe(*gt)
	if err != nil {
		return err
	}
	band := pipeline.ComputeBand(info.H, *bandFrac)
	m, err := eval.Compare(*out, *gt, *maskIn, band)
	if err != nil {
		return err
	}
	if *burned != "" {
		rec, err := eval.DetectionRecall(*burned, *maskIn, band, detect.DefaultParams(info.H))
		if err != nil {
			return err
		}
		m.MaskRecall = rec
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(m); err != nil {
		return err
	}
	if *jsonOut != "" {
		b, err := json.MarshalIndent(m, "", "  ")
		if err != nil {
			return err
		}
		if err := os.WriteFile(*jsonOut, b, 0o644); err != nil {
			return err
		}
	}
	return nil
}
