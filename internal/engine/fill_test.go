package engine

import (
	"errors"
	"testing"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/events"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/mask"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/route"
)

type stubPainter struct {
	jobs   []PaintJob
	frames [][]byte
	err    error
}

func (s *stubPainter) Inpaint(j PaintJob) ([][]byte, error) {
	s.jobs = append(s.jobs, j)
	if s.err != nil {
		return nil, s.err
	}
	out := make([][]byte, j.EndF-j.StartF+1)
	for i := range out {
		out[i] = make([]byte, j.W*j.BandH*3)
		for k := range out[i] {
			out[i][k] = byte(i + 1)
		}
	}
	return out, nil
}

func fillOpts(nFrames int, decisions ...route.Decision) FillOptions {
	return FillOptions{
		TemporalOptions: TemporalOptions{
			Input: "in.mp4", W: 8, BandY: 100, BandH: 4, FPS: 25,
			Masks: make([]mask.Frame, nFrames),
			Cuts:  []float64{0.2, 0.5, 1.5},
		},
		Decisions: decisions,
	}
}

func propDecision(startF, endF int) route.Decision {
	return route.Decision{Event: events.Event{StartF: startF, EndF: endF}, Engine: "propainter"}
}

// TestFramePaintedNilPainter: generative-tier decisions with no Painter must
// fall back to the motion tier and be reported.
func TestFramePaintedNilPainter(t *testing.T) {
	o := fillOpts(10,
		propDecision(2, 4),
		route.Decision{Event: events.Event{StartF: 6, EndF: 8}, Engine: "motion"},
		propDecision(6, 8),
	)
	painted, fallback, err := framePainted(o)
	if err != nil {
		t.Fatal(err)
	}
	if len(painted) != 0 {
		t.Fatalf("painted %d frames, want 0", len(painted))
	}
	if len(fallback) != 2 || fallback[0] != 0 || fallback[1] != 2 {
		t.Fatalf("fallback = %v, want [0 2]", fallback)
	}
}

// TestFramePaintedPainterError: an Inpaint error degrades that event to the
// motion tier; other events still go through the Painter.
func TestFramePaintedPainterError(t *testing.T) {
	p := &stubPainter{err: errors.New("boom")}
	o := fillOpts(10, propDecision(2, 4))
	o.Painter = p
	painted, fallback, err := framePainted(o)
	if err != nil {
		t.Fatal(err)
	}
	if len(painted) != 0 {
		t.Fatalf("painted %d frames, want 0", len(painted))
	}
	if len(fallback) != 1 || fallback[0] != 0 {
		t.Fatalf("fallback = %v, want [0]", fallback)
	}
	if len(p.jobs) != 1 {
		t.Fatalf("painter called %d times, want 1", len(p.jobs))
	}
}

// TestFramePaintedSuccess: the Painter receives the event's frame range with
// only the in-range cuts, and its frames land under absolute frame numbers.
func TestFramePaintedSuccess(t *testing.T) {
	p := &stubPainter{}
	o := fillOpts(10, propDecision(5, 7))
	o.Painter = p
	painted, fallback, err := framePainted(o)
	if err != nil {
		t.Fatal(err)
	}
	if len(fallback) != 0 {
		t.Fatalf("fallback = %v, want empty", fallback)
	}
	if len(painted) != 3 {
		t.Fatalf("painted %d frames, want 3", len(painted))
	}
	for f := 5; f <= 7; f++ {
		fr, ok := painted[f]
		if !ok {
			t.Fatalf("frame %d missing", f)
		}
		if len(fr) != 8*4*3 || fr[0] != byte(f-5+1) {
			t.Fatalf("frame %d content wrong", f)
		}
	}
	j := p.jobs[0]
	if j.StartF != 5 || j.EndF != 7 || len(j.Masks) != 3 {
		t.Fatalf("job range = %d..%d masks %d", j.StartF, j.EndF, len(j.Masks))
	}
	// cuts: t0=0.2, t1=0.32 → only 0.2 excluded (not strictly inside); 0.5
	// and 1.5 outside. 0.2 < t0? t0 = 5/25 = 0.2, cut must be > t0.
	if len(j.Cuts) != 0 {
		t.Fatalf("cuts = %v, want none strictly inside (0.2, 0.32)", j.Cuts)
	}
	if j.W != 8 || j.BandY != 100 || j.BandH != 4 {
		t.Fatalf("job geometry = %+v", j)
	}
}

// TestFramePaintedShortReturn: a Painter returning the wrong frame count is
// treated as a failure and the event falls back.
func TestFramePaintedShortReturn(t *testing.T) {
	o := fillOpts(10, propDecision(2, 4))
	o.Painter = &stubPainter2{}
	painted, fallback, err := framePainted(o)
	if err != nil {
		t.Fatal(err)
	}
	if len(painted) != 0 || len(fallback) != 1 {
		t.Fatalf("painted %d fallback %v, want 0/[0]", len(painted), fallback)
	}
}

// stubPainter2 returns a single frame regardless of the job range.
type stubPainter2 struct{}

func (s *stubPainter2) Inpaint(j PaintJob) ([][]byte, error) {
	return [][]byte{make([]byte, j.W*j.BandH*3)}, nil
}

// TestCoveragePerEvent: stats aggregate per event with clamped end frames.
func TestCoveragePerEvent(t *testing.T) {
	stats := []FrameStat{
		{Masked: 10, Real: 5}, {Masked: 20, Real: 20}, {Masked: 30, Real: 0},
	}
	evs := []events.Event{
		{StartF: 0, EndF: 1, Start: 0, End: 0.08},
		{StartF: 2, EndF: 9, Start: 0.08, End: 0.4}, // end beyond stats
	}
	cov := CoveragePerEvent(evs, stats)
	if len(cov) != 2 {
		t.Fatalf("%d coverages", len(cov))
	}
	if cov[0].Masked != 30 || cov[0].Real != 25 {
		t.Fatalf("ev0 = %+v", cov[0])
	}
	if got := cov[0].Coverage; got < 0.833 || got > 0.834 {
		t.Fatalf("ev0 coverage %v", got)
	}
	if cov[1].EndF != 2 || cov[1].Masked != 30 || cov[1].Coverage != 0 {
		t.Fatalf("ev1 = %+v", cov[1])
	}
}
