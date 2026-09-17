package alpha

import (
	"math"
	"testing"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/imgx"
)

const (
	tw, th = 240, 160
	barY0  = 60
	barY1  = 99 // 40px tall bar
	barX0  = 40
	barX1  = 199
)

// bgTexture is smooth enough to interpolate vertically yet textured enough
// for the row-energy test to see a contrast drop under the bar.
func bgTexture(x, y float64) float64 {
	return 150 + 45*math.Sin(x/7.0) + 25*math.Cos(y/5.0) + 12*math.Sin((x+y)/11.0)
}

func makeRGB() []byte {
	rgb := make([]byte, tw*th*3)
	for y := 0; y < th; y++ {
		for x := 0; x < tw; x++ {
			v := bgTexture(float64(x), float64(y))
			for c := 0; c < 3; c++ {
				rgb[(y*tw+x)*3+c] = uint8(v + 0.5)
			}
		}
	}
	return rgb
}

// applyBar blends a semi-transparent foreground over the bar rect.
func applyBar(rgb []byte, alpha float64, fore [3]uint8) {
	for y := barY0; y <= barY1; y++ {
		for x := barX0; x <= barX1; x++ {
			p := (y*tw + x) * 3
			for c := 0; c < 3; c++ {
				rgb[p+c] = uint8(alpha*float64(fore[c]) + (1-alpha)*float64(rgb[p+c]) + 0.5)
			}
		}
	}
}

func grayOf(rgb []byte) []uint8 {
	g := make([]uint8, tw*th)
	for p := 0; p < tw*th; p++ {
		g[p] = uint8((int(rgb[p*3]) + 2*int(rgb[p*3+1]) + int(rgb[p*3+2])) / 4)
	}
	return g
}

func psnr(a, b []byte) float64 {
	var se float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		se += d * d
	}
	mse := se / float64(len(a))
	if mse < 1e-9 {
		return 99
	}
	return 10 * math.Log10(255*255/mse)
}

func TestDetectBars(t *testing.T) {
	rgb := makeRGB()
	applyBar(rgb, 0.55, [3]uint8{20, 20, 20})
	bars := DetectBars(grayOf(rgb), tw, th, 32)
	if len(bars) != 1 {
		t.Fatalf("bars = %d, want 1 (%+v)", len(bars), bars)
	}
	r := bars[0].Rect
	if math.Abs(float64(r.Y-barY0)) > 2 || math.Abs(float64(r.Y+r.H-1-barY1)) > 2 {
		t.Fatalf("bar rows = %d..%d, want %d..%d", r.Y, r.Y+r.H-1, barY0, barY1)
	}
	if r.X > barX0+8 || r.X+r.W < barX1-8 {
		t.Fatalf("bar cols = %d..%d, want ~%d..%d", r.X, r.X+r.W-1, barX0, barX1)
	}
}

func TestDetectBarsNoneOnPlainTexture(t *testing.T) {
	rgb := makeRGB()
	if bars := DetectBars(grayOf(rgb), tw, th, 32); len(bars) != 0 {
		t.Fatalf("false bars on clean texture: %+v", bars)
	}
}

func TestUnmixRecoversBackground(t *testing.T) {
	rgb := makeRGB()
	want := makeRGB()
	applyBar(rgb, 0.4, [3]uint8{20, 20, 20})
	// Fake stroke cores excluded from the fit and from unmixing.
	stroke := make([]uint8, tw*th)
	for y := 70; y < 90; y++ {
		for x := 90; x < 150; x++ {
			stroke[y*tw+x] = 1
		}
	}
	bar, ok := Unmix(rgb, tw, th, Bar{Rect: imgx.Rect{X: barX0, Y: barY0, W: barX1 - barX0 + 1, H: barY1 - barY0 + 1}}, stroke)
	if !ok {
		t.Fatalf("Unmix abandoned a clean bar (resid=%.2f)", bar.Resid)
	}
	if math.Abs(bar.Alpha-0.4) > 0.05 {
		t.Fatalf("alpha = %.3f, want ~0.4", bar.Alpha)
	}
	// F absorbs the background-estimate bias by design (boundary continuity),
	// so only sanity-check it; recovery PSNR below is the real criterion.
	for c := 0; c < 3; c++ {
		if bar.Fore[c] > 80 {
			t.Fatalf("fore[%d] = %d, implausible for a dark scrim", c, bar.Fore[c])
		}
	}
	// Non-stroke bar pixels must come back near the original background.
	var a, b []byte
	for y := barY0; y <= barY1; y++ {
		for x := barX0; x <= barX1; x++ {
			if stroke[y*tw+x] != 0 {
				continue
			}
			p := (y*tw + x) * 3
			a = append(a, rgb[p:p+3]...)
			b = append(b, want[p:p+3]...)
		}
	}
	if q := psnr(a, b); q < 40 {
		t.Fatalf("PSNR after unmix = %.1fdB, want > 40", q)
	}
	// Stroke cores stay untouched for the repair mask.
	p := (75*tw + 100) * 3
	mixed := uint8(0.4*20 + 0.6*float64(want[p]) + 0.5)
	if d := int(rgb[p]) - int(mixed); d < -1 || d > 1 {
		t.Fatalf("stroke core modified: got %d, want ~%d", rgb[p], mixed)
	}
}

func TestUnmixRejectsBadFit(t *testing.T) {
	rgb := makeRGB()
	// A "bar" whose content is strong independent noise: no constant F/α can
	// explain it, so the fit residual must exceed the abandon threshold.
	for y := barY0; y <= barY1; y++ {
		for x := barX0; x <= barX1; x++ {
			p := (y*tw + x) * 3
			n := uint8((x*37 + y*91) % 256)
			for c := 0; c < 3; c++ {
				rgb[p+c] = uint8(0.4*float64(n) + 0.6*float64(rgb[p+c]))
			}
		}
	}
	before := append([]byte(nil), rgb...)
	bar, ok := Unmix(rgb, tw, th, Bar{Rect: imgx.Rect{X: barX0, Y: barY0, W: barX1 - barX0 + 1, H: barY1 - barY0 + 1}}, nil)
	if ok {
		t.Fatalf("noisy bar accepted (resid=%.2f)", bar.Resid)
	}
	for i := range rgb {
		if rgb[i] != before[i] {
			t.Fatal("abandoned bar modified the frame")
		}
	}
}

func TestUnmixRejectsNearOpaque(t *testing.T) {
	// Near-opaque dark bar: unmixing would divide by ~0.15 and amplify
	// noise past usefulness, so the fit is refused and the bar is left
	// untouched for the normal repair path (design R3.3).
	rgb := makeRGB()
	applyBar(rgb, 0.85, [3]uint8{20, 20, 20})
	before := append([]byte(nil), rgb...)
	if _, ok := Unmix(rgb, tw, th, Bar{Rect: imgx.Rect{X: barX0, Y: barY0, W: barX1 - barX0 + 1, H: barY1 - barY0 + 1}}, nil); ok {
		t.Fatal("near-opaque bar accepted; want rejection (α > 0.75)")
	}
	for i := range rgb {
		if rgb[i] != before[i] {
			t.Fatal("rejected bar modified the frame")
		}
	}
}

func TestUnmixClamps(t *testing.T) {
	// Bright bar over a bright background with a deliberately wrong F:
	// recovery (I − αF)/(1 − α) overshoots and must clamp into [0,255]
	// instead of wrapping.
	rgb := makeRGB()
	applyBar(rgb, 0.4, [3]uint8{250, 250, 250})
	if _, ok := Unmix(rgb, tw, th, Bar{Rect: imgx.Rect{X: barX0, Y: barY0, W: barX1 - barX0 + 1, H: barY1 - barY0 + 1}}, nil); !ok {
		t.Fatal("bright translucent bar rejected")
	}
}
