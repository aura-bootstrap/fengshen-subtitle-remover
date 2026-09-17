package route

import (
	"testing"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/events"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/imgx"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/mask"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/subs"
)

func TestClassifyBoundaries(t *testing.T) {
	cases := []struct {
		name string
		f    Features
		want Tier
	}{
		{"static", Features{MotionMag: 0.1, DurFrames: 30}, T0},
		{"static edge", Features{MotionMag: T0Motion, DurFrames: 30}, T0},
		{"slow", Features{MotionMag: 1.0, DurFrames: 30}, T1},
		{"slow edge", Features{MotionMag: T1Motion, DurFrames: 30}, T1},
		{"general", Features{MotionMag: 5.0, DurFrames: 30}, T2},
		{"general edge", Features{MotionMag: T2Motion, DurFrames: 30}, T2},
		{"fast", Features{MotionMag: 12.0, DurFrames: 30}, T3},
		{"large area", Features{AreaFrac: T3Area + 0.01, DurFrames: 30}, T3},
		{"dense texture", Features{Texture: T4Texture + 1, DurFrames: 30}, T4},
		{"texture beats area", Features{Texture: T4Texture + 1, AreaFrac: 0.5, MotionMag: 20, DurFrames: 30}, T4},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, reason := Classify(c.f)
			if got != c.want {
				t.Fatalf("Classify(%+v) = T%d, want T%d", c.f, got, c.want)
			}
			if len(reason) == 0 {
				t.Fatal("no reason recorded")
			}
		})
	}
}

func testPlan() *subs.Plan {
	// 100-frame video, one 30-frame event masking 2% of a 640x152 band.
	bits := make([]uint8, 640*152)
	for y := 70; y < 80; y++ {
		for x := 200; x < 240; x++ {
			bits[y*640+x] = 1
		}
	}
	m := mask.Encode(bits, 640, 152)
	masks := make([]mask.Frame, 100)
	for i := 10; i < 40; i++ {
		masks[i] = m
	}
	return &subs.Plan{
		Events: []events.Event{{StartF: 10, EndF: 39, Box: imgx.Rect{X: 200, Y: 70, W: 40, H: 10}}},
		Masks:  masks,
		Band:   subs.Band{Y: 208, H: 152},
	}
}

func TestDispatchBasic(t *testing.T) {
	p := testPlan()
	mags := make([]float64, 99)
	for i := range mags {
		mags[i] = 1.0
	}
	ds := Dispatch(p, 640, mags, func(imgx.Rect) float64 { return 10 }, "")
	if len(ds) != 1 {
		t.Fatalf("decisions = %d, want 1", len(ds))
	}
	d := ds[0]
	if d.Tier != T1 || d.Engine != "motion" || d.Risk {
		t.Fatalf("decision = %+v, want T1/motion/no-risk", d)
	}
	wantArea := float64(40*10) / float64(640*152)
	if d.Reason == nil || d.Event.StartF != 10 {
		t.Fatalf("bad decision fields: %+v", d)
	}
	_ = wantArea // area fraction feeds Classify; boundary covered by Classify tests
}

func TestDispatchT5Override(t *testing.T) {
	p := testPlan()
	p.Events[0].StartF, p.Events[0].EndF = 0, 89 // 90 of 100 frames
	ds := Dispatch(p, 640, nil, nil, "")
	if ds[0].Tier != T5 || ds[0].Engine != "propainter" || !ds[0].Risk {
		t.Fatalf("persistent occlusion = %+v, want T5/propainter/risk", ds[0])
	}
}

func TestDispatchForcedShortCircuit(t *testing.T) {
	p := testPlan()
	ds := Dispatch(p, 640, nil, func(imgx.Rect) float64 { return 100 }, "delogo")
	d := ds[0]
	if d.Engine != "delogo" {
		t.Fatalf("forced engine = %q, want delogo", d.Engine)
	}
	if d.Tier != T4 || !d.Risk { // classification still runs for the report
		t.Fatalf("tier = T%d risk=%v, want T4/risk", d.Tier, d.Risk)
	}
}
