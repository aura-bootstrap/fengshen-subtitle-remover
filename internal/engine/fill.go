package engine

import (
	"fmt"
	"io"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/events"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/mask"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/route"
)

// PaintJob describes one event's generative-tier repair: the frame range
// [StartF, EndF] of the band plus its repair masks. Shot cuts inside the
// range force chunk splits in the sidecar.
type PaintJob struct {
	Input   string
	W       int
	BandY   int
	BandH   int
	FPS     float64
	StartF  int
	EndF    int
	Masks   []mask.Frame // one per frame in [StartF, EndF], band coords
	Cuts    []float64    // cut times (seconds) inside the range
	WorkDir string       // scratch space for the lossless intermediate
}

// Painter is the generative fill tier (implemented by
// internal/propainter.Client). Inpaint returns one repaired RGB band frame
// (W*BandH*3 bytes) per frame in [StartF, EndF]; only the masked pixels are
// composited back.
type Painter interface {
	Inpaint(j PaintJob) ([][]byte, error)
}

// FillOptions describes one arbitration run. Decisions pick the tier per
// event; events without a decision, and decisions the Painter cannot serve,
// use the motion tier (priority ①), whose own unfilled pixels diffuse (③).
type FillOptions struct {
	TemporalOptions
	Events    []events.Event
	Decisions []route.Decision
	Painter   Painter // nil: propainter decisions fall back to motion
	WorkDir   string
	Log       io.Writer // fallback reasons land here (R5.6)
}

// EventCoverage summarizes how much of an event's masked area was repaired
// with real pixels (temporal or warped) rather than spatial diffusion or
// generative fill.
type EventCoverage struct {
	StartF   int     `json:"start_f"`
	EndF     int     `json:"end_f"`
	Start    float64 `json:"start"`
	End      float64 `json:"end"`
	Masked   int     `json:"masked_px"`
	Real     int     `json:"real_px"`
	Coverage float64 `json:"coverage"`
}

// FillReport is the arbitration outcome.
type FillReport struct {
	Stats    []FrameStat
	Coverage []EventCoverage
	Fallback []int // event indices whose propainter tier was unavailable
}

// CoveragePerEvent aggregates per-frame fill stats into per-event coverage.
func CoveragePerEvent(evs []events.Event, stats []FrameStat) []EventCoverage {
	out := make([]EventCoverage, 0, len(evs))
	for _, ev := range evs {
		end := ev.EndF
		if end >= len(stats) {
			end = len(stats) - 1
		}
		var masked, real int
		for f := ev.StartF; f <= end; f++ {
			masked += stats[f].Masked
			real += stats[f].Real
		}
		cov := EventCoverage{
			StartF: ev.StartF, EndF: end, Start: ev.Start, End: ev.End,
			Masked: masked, Real: real,
		}
		if masked > 0 {
			cov.Coverage = float64(real) / float64(masked)
		}
		out = append(out, cov)
	}
	return out
}

// framePainted runs the Painter over every event routed to the generative
// tier and returns the repaired frames indexed by absolute frame number.
// Events the Painter cannot serve (nil client or an Inpaint error) are
// collected in fallback and their frames are left to the motion tier.
func framePainted(o FillOptions) (map[int][]byte, []int, error) {
	painted := map[int][]byte{}
	var fallback []int
	logf := func(format string, a ...any) {
		if o.Log != nil {
			fmt.Fprintf(o.Log, format+"\n", a...)
		}
	}
	for k, d := range o.Decisions {
		if d.Engine != "propainter" {
			continue
		}
		ev := d.Event
		if o.Painter == nil {
			logf("fill: event %d (%d-%d): no painter configured, using motion tier", k, ev.StartF, ev.EndF)
			fallback = append(fallback, k)
			continue
		}
		end := ev.EndF
		if end >= len(o.Masks) {
			end = len(o.Masks) - 1
		}
		if end < ev.StartF || ev.StartF < 0 {
			fallback = append(fallback, k)
			continue
		}
		var cuts []float64
		t0, t1 := float64(ev.StartF)/o.FPS, float64(end+1)/o.FPS
		for _, c := range o.Cuts {
			if c > t0 && c < t1 {
				cuts = append(cuts, c)
			}
		}
		frames, err := o.Painter.Inpaint(PaintJob{
			Input: o.Input, W: o.W, BandY: o.BandY, BandH: o.BandH,
			FPS: o.FPS, StartF: ev.StartF, EndF: end,
			Masks: o.Masks[ev.StartF : end+1], Cuts: cuts, WorkDir: o.WorkDir,
		})
		if err != nil || len(frames) != end-ev.StartF+1 {
			if err != nil {
				logf("fill: event %d (%d-%d): painter failed, using motion tier: %v", k, ev.StartF, end, err)
			} else {
				logf("fill: event %d (%d-%d): painter returned %d frames, using motion tier", k, ev.StartF, end, len(frames))
			}
			fallback = append(fallback, k)
			continue
		}
		for j, fr := range frames {
			if len(fr) != o.W*o.BandH*3 {
				return nil, nil, fmt.Errorf("fill: painter frame %d has %d bytes, want %d", ev.StartF+j, len(fr), o.W*o.BandH*3)
			}
			painted[ev.StartF+j] = fr
		}
	}
	return painted, fallback, nil
}

// RunFill executes the fill arbitration: the generative tier (priority ②)
// repairs its events up front, then every remaining frame goes through the
// temporal core — motion-compensated reconstruction (①) with spatial
// diffusion as the last resort (③).
func RunFill(o FillOptions) (*FillReport, error) {
	painted, fallback, err := framePainted(o)
	if err != nil {
		return nil, err
	}
	stats := make([]FrameStat, len(o.Masks))
	to := o.TemporalOptions
	to.Stats = stats
	if err := runCore(to, painted); err != nil {
		return nil, err
	}
	return &FillReport{
		Stats:    stats,
		Coverage: CoveragePerEvent(o.Events, stats),
		Fallback: fallback,
	}, nil
}
