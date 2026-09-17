package eval

import (
	"math"
	"testing"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/imgx"
)

func TestPSNRKnownValues(t *testing.T) {
	if got := PSNR(0); got != 99 {
		t.Errorf("PSNR(0) = %v, want 99 (identical)", got)
	}
	// MSE = 255² → 0 dB by definition.
	if got := PSNR(255 * 255); math.Abs(got) > 1e-9 {
		t.Errorf("PSNR(255^2) = %v, want 0", got)
	}
	// MSE = 1 → 48.13 dB.
	if got := PSNR(1); math.Abs(got-48.1308) > 0.001 {
		t.Errorf("PSNR(1) = %v, want ~48.13", got)
	}
}

func TestSSIMKnownValues(t *testing.T) {
	x := []uint8{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160}
	if got := SSIM(x, x); math.Abs(got-1) > 1e-9 {
		t.Errorf("SSIM(x, x) = %v, want 1", got)
	}
	// Constant blocks compared against a shifted constant: luminance term
	// dominates; c1=(2.55)^2 → 2*100*110/(100²+110²+c1) ≈ 0.9997.
	a := make([]uint8, 64)
	b := make([]uint8, 64)
	for i := range a {
		a[i] = 100
		b[i] = 110
	}
	want := (2.0*100*110 + 6.5025) / (100.0*100.0 + 110.0*110.0 + 6.5025)
	if got := SSIM(a, b); math.Abs(got-want) > 1e-6 {
		t.Errorf("SSIM(const100, const110) = %v, want %v", got, want)
	}
	// Uncorrelated noise must score well below 1.
	n := 64
	p := make([]uint8, n)
	q := make([]uint8, n)
	for i := 0; i < n; i++ {
		p[i] = uint8((i * 37) % 256)
		q[i] = uint8((i*91 + 13) % 256)
	}
	if got := SSIM(p, q); got > 0.9 {
		t.Errorf("SSIM(shuffled) = %v, want well below 1", got)
	}
	if SSIM(nil, nil) != 0 || SSIM(x, x[:8]) != 0 {
		t.Error("degenerate inputs must score 0")
	}
}

func TestRecall(t *testing.T) {
	if got := Recall(75, 100); got != 0.75 {
		t.Errorf("Recall(75,100) = %v, want 0.75", got)
	}
	if got := Recall(0, 0); got != 1 {
		t.Errorf("Recall(0,0) = %v, want 1 (nothing to recover)", got)
	}
	if got := Recall(5, 1000); math.Abs(got-0.005) > 1e-12 {
		t.Errorf("Recall(5,1000) = %v", got)
	}
}

func TestDilate(t *testing.T) {
	w, h := 8, 8
	bits := make([]uint8, w*h)
	bits[3*w+3] = 1
	out := dilate(bits, w, h, 1)
	on := 0
	for _, b := range out {
		if b != 0 {
			on++
		}
	}
	if on != 9 {
		t.Errorf("dilate r=1 of a single pixel = %d px, want 9", on)
	}
	if out[0] != 0 {
		t.Error("far corner affected")
	}
}

func TestGrowTo(t *testing.T) {
	r := growTo(imgx.Rect{X: 10, Y: 10, W: 1, H: 1}, 15, 12)
	if r.X != 10 || r.Y != 10 || r.W != 6 || r.H != 3 {
		t.Errorf("growTo = %+v", r)
	}
	r = growTo(r, 5, 20)
	if r.X != 5 || r.Y != 10 || r.W != 11 || r.H != 11 {
		t.Errorf("growTo left/down = %+v", r)
	}
}
