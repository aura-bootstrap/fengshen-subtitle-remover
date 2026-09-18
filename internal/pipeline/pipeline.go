// Package pipeline wires probing, the subtitle plan (internal/subs) and the
// removal engines together.
package pipeline

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/alpha"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/detect"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/engine"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/events"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/ffx"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/imgx"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/manifest"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/mask"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/ocr"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/propainter"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/report"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/route"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/subs"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/vlmqc"
)

type Options struct {
	Input                 string
	Output                string
	Engine                string // temporal | delogo
	BandStart             float64
	CRF                   int
	Preset                string
	MaxGap                int
	SceneThreshold        float64
	Neighbors             int
	Pad                   int
	Motion                bool // temporal engine: motion-compensated pixel transfer
	CloseGap              int  // temporal closing window in frames (subs)
	CloseOverlap          float64
	EdgePad               int // frames padded before/after each event
	OCR                   bool
	OCRScript             string
	OCRStride             int
	Alpha                 bool // semi-transparent bar detection + unmixing
	ProPainter            bool // enable the ProPainter sidecar for generative-tier events
	PainterScript         string
	PainterHome           string // PROPAINTER_HOME for the sidecar (empty: inherit env)
	PainterMaskDilation   int    // 0: model default 4
	PainterRaftIter       int    // 0: model default 20
	PainterNeighborLength int    // 0: model default 10
	Grain                 bool   // texture-match the repaired area (internal/grain)
	ForceEngine           string // R7.5: "motion"|"propainter" overrides the router for every event
	VLMQC                 bool   // re-judge verify-stage residue boxes with a VLM
	VLMScript             string // path to scripts/vlm_qc.py (empty: default relative path)
	VLMEndpoint           string // OpenAI-compatible base URL (empty: local ollama default)
	VLMModel              string // VLM model name (empty: qwen2.5vl:7b)
	VLMAPIKey             string
	ManifestPath          string
	RiskListPath          string  // write the high-risk segment list (R6) as JSON
	RiskCoverage          float64 // coverage threshold below which an event is high-risk (0 uses the default)
	DumpDir               string
	DumpLimit             int
	DumpStride            int
	Verify                bool
	Log                   io.Writer
}

// EventCoverage summarizes how much of an event's masked area was repaired
// with real pixels (temporal or warped) rather than spatial diffusion; low
// coverage marks the high-risk segments that need human review (roadmap L4).
type EventCoverage = engine.EventCoverage

type Detection = subs.Detection

type Band = subs.Band

// FrameBar records one accepted semi-transparent bar; Rect is in full-frame
// coordinates.
type FrameBar = alpha.FrameBar

// RiskItem is one high-risk span flagged for review (R6).
type RiskItem = report.RiskItem

var ComputeBand = subs.ComputeBand

// scanBars runs bar detection on the frames that carry a repair mask. It is
// the detect-mode counterpart of the engine's inline unmixing: bars are
// reported but no output pixels are written.
func scanBars(input string, w int, b Band, charH int, masks []mask.Frame) ([]FrameBar, error) {
	fr, err := ffx.NewFrameReader(input, fmt.Sprintf("crop=%d:%d:0:%d,format=rgb24", w, b.H, b.Y), w, b.H, "rgb24")
	if err != nil {
		return nil, err
	}
	defer fr.Close()
	raw := make([]byte, w*b.H*3)
	bits := make([]uint8, w*b.H)
	var out []FrameBar
	for i := 0; i < len(masks); i++ {
		ok, err := fr.Next(raw)
		if err != nil || !ok {
			return out, err
		}
		if masks[i].Empty() {
			continue
		}
		masks[i].Decode(bits, w)
		for _, fb := range alpha.UnmixFrame(raw, w, b.H, charH, bits, i) {
			fb.Rect.Y += b.Y
			out = append(out, fb)
		}
	}
	return out, nil
}

// ResidueBox locates one detection in the output that overlaps where the
// original had subtitles (true removal misses, not prop text).
type ResidueBox struct {
	Frame int     `json:"frame"`
	Time  float64 `json:"time"`
	X     int     `json:"x"`
	Y     int     `json:"y"`
	W     int     `json:"w"`
	H     int     `json:"h"`
}

type Report struct {
	Input       string           `json:"input"`
	Output      string           `json:"output,omitempty"`
	Engine      string           `json:"engine"`
	Width       int              `json:"width"`
	Height      int              `json:"height"`
	FPS         float64          `json:"fps"`
	Duration    float64          `json:"duration"`
	BandY       int              `json:"band_y"`
	BandH       int              `json:"band_h"`
	CharH       int              `json:"char_h"`
	Stroke      int              `json:"stroke"`
	Cuts        int              `json:"scene_cuts"`
	Detection   Detection        `json:"detection"`
	Events      []events.Event   `json:"events,omitempty"`
	Bars        []FrameBar       `json:"bars,omitempty"`
	Routing     []route.Decision `json:"routing,omitempty"`
	Motion      []EventCoverage  `json:"motion,omitempty"`
	Fallback    []int            `json:"fallback,omitempty"`
	Residual    *Detection       `json:"residual,omitempty"`
	ResidueList []ResidueBox     `json:"residue_list,omitempty"`
	VLMQC       []vlmqc.Verdict  `json:"vlm_qc,omitempty"`
	Risk        []RiskItem       `json:"risk,omitempty"`
	// CoverageTotal is Σreal/Σmasked over all events (R4.7 acceptance).
	CoverageTotal float64 `json:"coverage_total,omitempty"`
	ElapsedSec    float64 `json:"elapsed_sec"`
}

func Run(o Options) (*Report, error) {
	if o.Log == nil {
		o.Log = io.Discard
	}
	start := time.Now()
	info, err := ffx.Probe(o.Input)
	if err != nil {
		return nil, err
	}
	if info.SubtitleStreams > 0 {
		fmt.Fprintf(o.Log, "note: %d soft subtitle stream(s) present; `ffmpeg -c copy -sn` would remove them losslessly\n", info.SubtitleStreams)
	}
	b := subs.ComputeBand(info.H, o.BandStart)
	params := detect.DefaultParams(info.H)
	fmt.Fprintf(o.Log, "video %dx%d %.3gfps %.1fs; band y=%d h=%d; charH=%d stroke=%d dilate=%d\n",
		info.W, info.H, info.FPS, info.Duration, b.Y, b.H, params.CharH, params.Stroke, params.MaskDilate)

	var cuts []float64
	if o.SceneThreshold > 0 {
		t0 := time.Now()
		cuts, err = ffx.SceneCuts(o.Input, o.SceneThreshold)
		if err != nil {
			fmt.Fprintf(o.Log, "warn: scene detection failed: %v\n", err)
			cuts = nil
		} else {
			fmt.Fprintf(o.Log, "scene cuts: %d (%.1fs)\n", len(cuts), time.Since(t0).Seconds())
		}
	}

	t0 := time.Now()
	var ocrClient *ocr.Client
	if o.OCR {
		script := o.OCRScript
		if script == "" {
			script = "scripts/ocr_boxes.py"
		}
		// CPU OCR costs seconds per frame; the fixed 10-minute default kills
		// long videos mid-request (sidecar EOF), so scale the cap with the
		// estimated workload.
		estFrames := int(info.Duration*info.FPS)/max(o.OCRStride, 1) + 1
		timeout := 10 * time.Minute
		if d := time.Duration(estFrames) * 10 * time.Second; d > timeout {
			timeout = d
		}
		c, err := ocr.NewClient(script, timeout)
		if err != nil {
			fmt.Fprintf(o.Log, "warn: ocr disabled: %v\n", err)
		} else {
			ocrClient = c
		}
	}
	plan, err := subs.Build(subs.Options{
		Input: o.Input, W: info.W, FPS: info.FPS, Band: b, Params: params,
		Cuts: cuts, MaxGap: o.MaxGap,
		CloseGap: o.CloseGap, CloseOverlap: o.CloseOverlap, EdgePad: o.EdgePad,
		OCR: ocrClient, OCRStride: o.OCRStride,
		DumpDir: o.DumpDir, DumpLimit: o.DumpLimit, DumpStride: o.DumpStride,
		Log: o.Log,
	})
	if err != nil {
		return nil, err
	}
	evs, masks, det := plan.Events, plan.Masks, plan.Detection
	params = plan.Params
	fmt.Fprintf(o.Log, "detect: %d frames, %d with text (%d boxes, %d mask px), %d events, %.1fs\n",
		det.Frames, det.TextFrames, det.Boxes, det.TextPix, len(evs), time.Since(t0).Seconds())

	rep := &Report{
		Input: o.Input, Output: o.Output, Engine: o.Engine,
		Width: info.W, Height: info.H, FPS: info.FPS, Duration: info.Duration,
		BandY: b.Y, BandH: b.H, CharH: params.CharH, Stroke: params.Stroke,
		Cuts: len(cuts), Detection: det, Events: evs,
	}

	if o.Alpha && o.Output == "" {
		t0 = time.Now()
		bars, err := scanBars(o.Input, info.W, b, params.CharH, masks)
		if err != nil {
			fmt.Fprintf(o.Log, "warn: bar scan failed: %v\n", err)
		} else {
			rep.Bars = bars
			fmt.Fprintf(o.Log, "alpha: %d bars accepted (%.1fs)\n", len(bars), time.Since(t0).Seconds())
		}
	}

	if o.Output != "" {
		t0 = time.Now()
		switch o.Engine {
		case "delogo":
			full := make([]events.Event, len(evs))
			for i, ev := range evs {
				ev.Box.Y += b.Y
				full[i] = ev
			}
			err = engine.RunDelogo(engine.DelogoOptions{
				Input: o.Input, Output: o.Output, W: info.W, H: info.H,
				Events: full, Pad: o.Pad, CRF: o.CRF, Preset: o.Preset,
				EncColor: info.ColorEncodeArgs(),
			})
		case "temporal":
			base := engine.TemporalOptions{
				Input: o.Input, Output: o.Output, W: info.W, H: info.H,
				BandY: b.Y, BandH: b.H, FPS: info.FPS, Masks: masks, Cuts: cuts,
				CRF: o.CRF, Preset: o.Preset, Neighbors: o.Neighbors,
				RegionPad: params.MaskDilate * 2, Motion: o.Motion,
				Alpha: o.Alpha, CharH: params.CharH, Grain: o.Grain,
				EncColor: info.ColorEncodeArgs(),
				Progress: func(f, t int) { fmt.Fprintf(o.Log, "repair %d/%d\r", f, t) },
			}
			mags, merr := engine.EstimateMags(base)
			if merr != nil {
				return nil, merr
			}
			if os.Getenv("DESUB_DBG") != "" && len(mags) > 0 {
				lo, hi, sum := mags[0], mags[0], 0.0
				for _, m := range mags {
					lo, hi = min(lo, m), max(hi, m)
					sum += m
				}
				fmt.Fprintf(o.Log, "mags: n=%d min=%.2f max=%.2f mean=%.2f\n", len(mags), lo, hi, sum/float64(len(mags)))
			}
			rep.Routing = route.Dispatch(plan, info.W, mags, nil, o.ForceEngine)
			var painter engine.Painter
			if o.ProPainter {
				if cl, cerr := propainter.NewClient(o.PainterScript, 0); cerr != nil {
					fmt.Fprintf(o.Log, "warn: propainter unavailable: %v; generative events fall back to motion\n", cerr)
				} else {
					cl.Home = o.PainterHome
					cl.MaskDilation = o.PainterMaskDilation
					cl.RaftIter = o.PainterRaftIter
					cl.NeighborLength = o.PainterNeighborLength
					painter = cl
				}
			}
			var barList []alpha.FrameBar
			base.Bars = &barList
			var frep *engine.FillReport
			frep, err = engine.RunFill(engine.FillOptions{
				TemporalOptions: base,
				Events:          evs,
				Decisions:       rep.Routing,
				Painter:         painter,
				Log:             o.Log,
			})
			fmt.Fprintln(o.Log)
			for i := range barList {
				barList[i].Rect.Y += b.Y
			}
			rep.Bars = barList
			if err == nil && frep != nil {
				rep.Motion = frep.Coverage
				var masked, real int
				for _, c := range frep.Coverage {
					masked += c.Masked
					real += c.Real
				}
				if masked > 0 {
					rep.CoverageTotal = float64(real) / float64(masked)
				}
				rep.Fallback = frep.Fallback
			}
		default:
			return nil, fmt.Errorf("unknown engine %q", o.Engine)
		}
		if err != nil {
			return nil, err
		}
		engDetail := ""
		if len(rep.Routing) > 0 {
			counts := map[string]int{}
			order := []string{}
			for _, d := range rep.Routing {
				e := d.Engine
				if _, seen := counts[e]; !seen {
					order = append(order, e)
				}
				counts[e]++
			}
			parts := make([]string, 0, len(order))
			for _, e := range order {
				parts = append(parts, fmt.Sprintf("%s x%d", e, counts[e]))
			}
			engDetail = " (" + strings.Join(parts, ", ") + ")"
		}
		fmt.Fprintf(o.Log, "engine %s%s: %.1fs -> %s\n", o.Engine, engDetail, time.Since(t0).Seconds(), o.Output)
	}

	if o.Verify && o.Output != "" {
		t0 = time.Now()
		outFrames, _, res, err := subs.Measure(o.Output, info.W, b, params, false, "", 0, 1)
		if err != nil {
			return nil, err
		}
		for i := range outFrames {
			if i >= len(plan.Frames) {
				break
			}
			region := subs.EventBoxAt(evs, i)
			if region == nil {
				continue
			}
			reg := region.Expand(params.CharH / 2)
			residBoxes := 0
			for _, ob := range outFrames[i].Boxes {
				if inter := ob.Intersect(reg); inter.W*inter.H*2 < ob.W*ob.H {
					continue
				}
				for _, ib := range plan.Frames[i].Boxes {
					inter := ob.Intersect(ib)
					if inter.W*inter.H*2 >= ob.W*ob.H {
						residBoxes++
						rep.ResidueList = append(rep.ResidueList, ResidueBox{
							Frame: i, Time: float64(i) / info.FPS,
							X: ob.X, Y: ob.Y + b.Y, W: ob.W, H: ob.H,
						})
						break
					}
				}
			}
			if residBoxes > 0 {
				res.ResidueFrames++
				res.ResidueBoxes += residBoxes
			}
		}
		rep.Residual = &res
		fmt.Fprintf(o.Log, "verify: %d/%d frames still detected, %d boxes, %d overlapping original subtitle regions (%.1fs)\n",
			res.TextFrames, res.Frames, res.Boxes, res.ResidueBoxes, time.Since(t0).Seconds())
	}
	if o.VLMQC && len(rep.ResidueList) > 0 {
		t0 := time.Now()
		cl, err := vlmqc.NewClient(o.VLMScript, o.VLMEndpoint, o.VLMModel, o.VLMAPIKey)
		if err != nil {
			fmt.Fprintf(o.Log, "warn: vlm-qc unavailable: %v\n", err)
		} else {
			boxes := make([]vlmqc.Box, len(rep.ResidueList))
			for i, r := range rep.ResidueList {
				boxes[i] = vlmqc.Box{Frame: r.Frame, Time: r.Time, X: r.X, Y: r.Y, W: r.W, H: r.H}
			}
			verdicts, rerr := cl.Review(o.Output, boxes)
			if rerr != nil {
				fmt.Fprintf(o.Log, "warn: vlm-qc failed, threshold counts kept: %v\n", rerr)
			} else {
				rep.VLMQC = verdicts
				confirmed, unclear := 0, 0
				for _, v := range verdicts {
					switch {
					case v.Residue == nil:
						unclear++
					case *v.Residue:
						confirmed++
					}
				}
				fmt.Fprintf(o.Log, "vlm-qc: %d boxes reviewed, %d confirmed residue, %d texture, %d unclear (%.1fs)\n",
					len(verdicts), confirmed, len(verdicts)-confirmed-unclear, unclear, time.Since(t0).Seconds())
			}
		}
	}
	rep.ElapsedSec = time.Since(start).Seconds()
	if o.Output != "" {
		m, err := buildManifest(o, rep, info, evs, masks, b, params)
		if err != nil {
			fmt.Fprintf(o.Log, "warn: manifest: %v\n", err)
		} else {
			if o.ManifestPath != "" {
				if err := manifest.Write(o.ManifestPath, m); err != nil {
					fmt.Fprintf(o.Log, "warn: manifest: %v\n", err)
				}
			}
			covThr := o.RiskCoverage
			if covThr <= 0 {
				covThr = report.DefaultCoverage
			}
			var residues []report.Residue
			for _, r := range rep.ResidueList {
				residues = append(residues, report.Residue{
					Frame: r.Frame, Time: r.Time,
					Box: imgx.Rect{X: r.X, Y: r.Y, W: r.W, H: r.H},
				})
			}
			rep.Risk = report.RiskList(m, residues, covThr)
			if o.RiskListPath != "" {
				if err := writeRiskList(o.RiskListPath, rep.Risk); err != nil {
					fmt.Fprintf(o.Log, "warn: risk list: %v\n", err)
				}
			}
		}
	}
	return rep, nil
}

// writeRiskList exports the risk list as indented JSON.
func writeRiskList(path string, items []RiskItem) error {
	if items == nil {
		items = []RiskItem{}
	}
	b, err := json.MarshalIndent(items, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

// buildManifest records the run so individual segments can be re-run later
// (R8.1). Segment routing comes from rep.Routing (aligned with events);
// coverage comes from rep.Motion keyed by frame range.
func buildManifest(o Options, rep *Report, info *ffx.MediaInfo, evs []events.Event, masks []mask.Frame, b Band, params detect.Params) (*manifest.Manifest, error) {
	fp, err := manifest.FingerprintInput(o.Input, info)
	if err != nil {
		return nil, err
	}
	m := manifest.Manifest{
		Version: manifest.Version,
		Input:   fp,
		Params: map[string]any{
			"engine": o.Engine, "crf": o.CRF, "preset": o.Preset,
			"motion": o.Motion, "alpha": o.Alpha, "grain": o.Grain,
			"neighbors": o.Neighbors, "band_start": b,
		},
		Band:  manifest.Band{Y: b.Y, H: b.H},
		CharH: params.CharH,
	}
	covOf := func(startF, endF int) float64 {
		for _, c := range rep.Motion {
			if c.StartF == startF && c.EndF <= endF {
				return c.Coverage
			}
		}
		return 0
	}
	for i, ev := range evs {
		tier, eng := 0, "motion"
		var risk bool
		var reason []string
		if i < len(rep.Routing) {
			d := rep.Routing[i]
			tier, eng, risk, reason = int(d.Tier), d.Engine, d.Risk, d.Reason
		}
		m.Segments = append(m.Segments, manifest.NewSegment(i, ev, masks, tier, eng, covOf(ev.StartF, ev.EndF), risk, reason))
	}
	return &m, nil
}
