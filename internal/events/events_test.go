package events

import (
	"testing"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/imgx"
)

func r(x, y, w, h int) imgx.Rect { return imgx.Rect{X: x, Y: y, W: w, H: h} }

func TestRobustUnionKeepsSecondLine(t *testing.T) {
	// Two subtitle lines ~2 estimated char heights apart (the T1 geometry:
	// 44px glyphs with charH estimated at 29). Both must survive.
	var rs []imgx.Rect
	for i := 0; i < 30; i++ {
		rs = append(rs, r(400, 110, 392, 42), r(410, 112, 390, 42))
		rs = append(rs, r(420, 168, 397, 42))
	}
	u := robustUnion(rs, 29)
	if u.Y > 110 || u.Bottom() < 210 {
		t.Errorf("second line dropped: union = %+v", u)
	}
}

func TestRobustUnionRejectsFarPropText(t *testing.T) {
	// A sporadic prop detection far from the subtitle cluster must not
	// inflate the union.
	var rs []imgx.Rect
	for i := 0; i < 30; i++ {
		rs = append(rs, r(400, 110, 392, 42))
	}
	rs = append(rs, r(100, 20, 200, 40)) // 1/31 < 1/3 of modal count
	u := robustUnion(rs, 29)
	if u.Y < 100 {
		t.Errorf("prop text inflated the union: %+v", u)
	}
}

func TestRobustUnionTinyInput(t *testing.T) {
	u := robustUnion([]imgx.Rect{r(0, 0, 10, 10), r(100, 500, 10, 10)}, 29)
	if u.W != 110 || u.H != 510 {
		t.Errorf("two boxes are unioned verbatim, got %+v", u)
	}
	if got := robustUnion(nil, 29); got != (imgx.Rect{}) {
		t.Errorf("empty input = %+v", got)
	}
}

func TestBuildSplitsOnLongGap(t *testing.T) {
	frames := make([]Frame, 20)
	for i := 0; i < 20; i++ {
		if i >= 8 && i <= 11 {
			continue // 4-frame gap > maxGap 3
		}
		frames[i] = Frame{Boxes: []imgx.Rect{r(10, 10, 50, 20)}}
	}
	evs := Build(frames, 25, 3, nil, 24)
	if len(evs) != 2 {
		t.Fatalf("got %d events, want 2", len(evs))
	}
	if evs[0].StartF != 0 || evs[0].EndF != 7 || evs[1].StartF != 12 || evs[1].EndF != 19 {
		t.Errorf("events = %+v", evs)
	}
}

func TestBuildCutForcesSplit(t *testing.T) {
	frames := make([]Frame, 10)
	for i := range frames {
		frames[i] = Frame{Boxes: []imgx.Rect{r(10, 10, 50, 20)}}
	}
	// Gap of empty frames 4-5 is within maxGap, but a cut at frame 5 must
	// block bridging across it.
	frames[4] = Frame{}
	frames[5] = Frame{}
	evs := Build(frames, 25, 3, []float64{5.0 / 25}, 24)
	if len(evs) != 2 {
		t.Fatalf("got %d events, want 2 (cut split)", len(evs))
	}
	if evs[0].EndF != 3 || evs[1].StartF != 6 {
		t.Errorf("events = %+v", evs)
	}
}
