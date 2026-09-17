package subs

import (
	"context"
	"errors"
	"testing"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/events"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/imgx"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/ocr"
)

func ev(startF, endF int, box imgx.Rect) events.Event {
	return events.Event{
		StartF: startF, EndF: endF,
		Start: float64(startF) / 25, End: float64(endF+1) / 25,
		Box: box, Frames: endF - startF + 1,
	}
}

func TestCloseEventsMergesSamePositionGap(t *testing.T) {
	// Two subtitle lines at the same y with a 10-frame undetected gap
	// (the fade between lines) must merge into one event covering the gap.
	box := imgx.Rect{X: 200, Y: 90, W: 300, H: 60}
	box2 := imgx.Rect{X: 210, Y: 92, W: 280, H: 58}
	evs := []events.Event{ev(100, 200, box), ev(211, 300, box2)}
	shotID := make([]int, 400)
	out := closeEvents(evs, shotID, 24, 0.5)
	if len(out) != 1 {
		t.Fatalf("want 1 merged event, got %d", len(out))
	}
	if out[0].StartF != 100 || out[0].EndF != 300 {
		t.Fatalf("merged range = %d-%d, want 100-300", out[0].StartF, out[0].EndF)
	}
	if out[0].Frames != 201 {
		t.Fatalf("Frames = %d, want 201", out[0].Frames)
	}
	want := imgx.Rect{X: 200, Y: 90, W: 300, H: 60} // box2 is fully contained in box
	if out[0].Box != want {
		t.Fatalf("merged box = %+v, want %+v", out[0].Box, want)
	}
}

func TestCloseEventsChainMerges(t *testing.T) {
	box := imgx.Rect{X: 200, Y: 90, W: 300, H: 60}
	evs := []events.Event{ev(0, 10, box), ev(15, 25, box), ev(30, 40, box)}
	shotID := make([]int, 100)
	out := closeEvents(evs, shotID, 24, 0.5)
	if len(out) != 1 || out[0].StartF != 0 || out[0].EndF != 40 {
		t.Fatalf("chain merge failed: %+v", out)
	}
}

func TestCloseEventsRejectsAcrossCut(t *testing.T) {
	box := imgx.Rect{X: 200, Y: 90, W: 300, H: 60}
	evs := []events.Event{ev(100, 200, box), ev(205, 300, box)}
	shotID := make([]int, 400)
	for f := 202; f < 400; f++ { // cut between the two events
		shotID[f] = 1
	}
	out := closeEvents(evs, shotID, 24, 0.5)
	if len(out) != 2 {
		t.Fatalf("cut must block the merge, got %d events", len(out))
	}
}

func TestCloseEventsRejectsLowIoU(t *testing.T) {
	a := imgx.Rect{X: 100, Y: 90, W: 200, H: 60}
	b := imgx.Rect{X: 400, Y: 300, W: 200, H: 60} // different position
	evs := []events.Event{ev(100, 200, a), ev(205, 300, b)}
	shotID := make([]int, 400)
	out := closeEvents(evs, shotID, 24, 0.5)
	if len(out) != 2 {
		t.Fatalf("low IoU must block the merge, got %d events", len(out))
	}
}

func TestCloseEventsRejectsLongGap(t *testing.T) {
	box := imgx.Rect{X: 200, Y: 90, W: 300, H: 60}
	evs := []events.Event{ev(100, 200, box), ev(300, 400, box)} // gap 99 > 24
	shotID := make([]int, 500)
	out := closeEvents(evs, shotID, 24, 0.5)
	if len(out) != 2 {
		t.Fatalf("long gap must block the merge, got %d events", len(out))
	}
}

func TestPadEvents(t *testing.T) {
	box := imgx.Rect{X: 200, Y: 90, W: 300, H: 60}
	evs := []events.Event{ev(50, 100, box), ev(150, 200, box)}
	out := padEvents(evs, 2, 300, 25)
	if out[0].StartF != 48 || out[0].EndF != 102 {
		t.Fatalf("first = %d-%d, want 48-102", out[0].StartF, out[0].EndF)
	}
	if out[1].StartF != 148 || out[1].EndF != 202 {
		t.Fatalf("second = %d-%d, want 148-202", out[1].StartF, out[1].EndF)
	}
	if out[0].Frames != 55 {
		t.Fatalf("Frames = %d, want 55", out[0].Frames)
	}
}

func TestPadEventsClampsBounds(t *testing.T) {
	box := imgx.Rect{X: 200, Y: 90, W: 300, H: 60}
	evs := []events.Event{ev(0, 10, box), ev(90, 99, box)}
	out := padEvents(evs, 5, 100, 25)
	if out[0].StartF != 0 {
		t.Fatalf("first start = %d, want 0 (clamped)", out[0].StartF)
	}
	if out[1].EndF != 99 {
		t.Fatalf("last end = %d, want 99 (clamped)", out[1].EndF)
	}
}

func TestPadEventsNoOverlap(t *testing.T) {
	box := imgx.Rect{X: 200, Y: 90, W: 300, H: 60}
	// Gap of 3 between events, pad 5 on both sides must not overlap.
	evs := []events.Event{ev(10, 20, box), ev(24, 30, box)}
	out := padEvents(evs, 5, 100, 25)
	if out[0].EndF >= out[1].StartF {
		t.Fatalf("overlap: first ends %d, second starts %d", out[0].EndF, out[1].StartF)
	}
	if out[0].EndF != 23 {
		t.Fatalf("first end = %d, want 23 (clamped to neighbour start-1)", out[0].EndF)
	}
}

func TestCloseThenPadKeepsGapCovered(t *testing.T) {
	// The 2445-2552 / 2563-2568 / 2573-2659 pattern from the ep1 regression:
	// three lines at the same y with short undetected gaps must end up one
	// continuous event so the gap frames are repaired and never sampled.
	box := imgx.Rect{X: 148, Y: 89, W: 423, H: 63}
	box2 := imgx.Rect{X: 282, Y: 90, W: 156, H: 64}
	evs := []events.Event{ev(2445, 2552, box), ev(2563, 2568, box2), ev(2573, 2659, box)}
	shotID := make([]int, 3349)
	closed := closeEvents(evs, shotID, 24, 0.5)
	if len(closed) != 1 {
		t.Fatalf("want 1 merged event, got %d", len(closed))
	}
	padded := padEvents(closed, 2, 3349, 25)
	if padded[0].StartF != 2443 || padded[0].EndF != 2661 {
		t.Fatalf("final range = %d-%d, want 2443-2661", padded[0].StartF, padded[0].EndF)
	}
}

func TestFuseOCRUnionAndDedup(t *testing.T) {
	frames := []events.Frame{
		{Boxes: []imgx.Rect{{X: 100, Y: 90, W: 200, H: 50}}},
		{},
		{},
	}
	stub := ocr.NewStubClient(func(_ context.Context, req ocr.Request) (ocr.Response, error) {
		if len(req.Frames) != 1 || req.Frames[0] != 0 {
			t.Errorf("sampled frames = %v, want [0]", req.Frames)
		}
		return ocr.Response{Boxes: []ocr.Box{
			// Fully inside the CV box: adds nothing.
			{Frame: 0, Rect: imgx.Rect{X: 120, Y: 95, W: 150, H: 40}, Score: 0.9},
			// New text the CV pass missed.
			{Frame: 0, Rect: imgx.Rect{X: 300, Y: 300, W: 120, H: 40}, Score: 0.8},
		}}, nil
	})
	o := Options{Input: "/in.mp4", W: 720, Band: Band{Y: 742, H: 538}, OCR: stub, OCRStride: 12}
	added := fuseOCR(o, frames, nil)
	if added != 1 {
		t.Fatalf("added = %d, want 1", added)
	}
	if len(frames[0].Boxes) != 2 {
		t.Fatalf("frame 0 boxes = %d, want 2", len(frames[0].Boxes))
	}
	got := frames[0].Boxes[1]
	if got.X != 300 || got.Y != 300 {
		t.Fatalf("fused box = %+v, want (300,300)", got)
	}
}

func TestFuseOCRDegradesOnError(t *testing.T) {
	frames := []events.Frame{{}, {}}
	stub := ocr.NewStubClient(func(_ context.Context, _ ocr.Request) (ocr.Response, error) {
		return ocr.Response{}, errors.New("exit status 2")
	})
	o := Options{Input: "/in.mp4", W: 720, Band: Band{Y: 742, H: 538}, OCR: stub, OCRStride: 1}
	if added := fuseOCR(o, frames, nil); added != 0 {
		t.Fatalf("added = %d, want 0 on sidecar failure", added)
	}
}
