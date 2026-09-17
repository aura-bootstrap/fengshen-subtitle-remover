package motion

import (
	"math"
	"testing"
)

const (
	tw  = 320
	th  = 240
	tdx = 1.6
	tdy = -0.9
)

// tex is a smooth crossing-gradient texture: sin(x/6)*sin(y/6) makes gradient
// direction rotate within a 3x3 window (real Shi-Tomasi corner strength) while
// the ~38px wavelength keeps the single-level LK basin far larger than tdx/tdy.
func tex(x, y float64) float64 {
	return 127 + 60*math.Sin(x/6.0)*math.Sin(y/6.0) + 30*math.Sin((x+y)/9.0)
}

func grayAt(fn func(x, y float64) float64) []uint8 {
	g := make([]uint8, tw*th)
	for y := 0; y < th; y++ {
		for x := 0; x < tw; x++ {
			v := fn(float64(x), float64(y)) + 0.5
			if v < 0 {
				v = 0
			}
			if v > 255 {
				v = 255
			}
			g[y*tw+x] = uint8(v)
		}
	}
	return g
}

func trackAll(t *testing.T) [9]float64 {
	t.Helper()
	prev := grayAt(func(x, y float64) float64 { return tex(x, y) })
	// Feature at (x,y) in prev sits at (x+tdx, y+tdy) in cur.
	cur := grayAt(func(x, y float64) float64 { return tex(x-tdx, y-tdy) })

	pts := Corners(prev, tw, th, nil, 24, 400, 120)
	if len(pts) < 40 {
		t.Fatalf("Corners found only %d points, need >= 40", len(pts))
	}
	nxt, ok := TrackLK(prev, cur, tw, th, pts, 7, 30)
	nOK := 0
	for _, o := range ok {
		if o {
			nOK++
		}
	}
	if nOK < 30 {
		t.Fatalf("TrackLK converged on only %d/%d points", nOK, len(pts))
	}
	for i := range pts {
		if !ok[i] {
			continue
		}
		if math.Hypot(nxt[i].X-pts[i].X-tdx, nxt[i].Y-pts[i].Y-tdy) > 0.5 {
			t.Fatalf("point %d tracked to (%.2f,%.2f), expected (%.2f,%.2f)",
				i, nxt[i].X, nxt[i].Y, pts[i].X+tdx, pts[i].Y+tdy)
		}
	}

	h, inl, fit := FitHomography(pts, nxt, ok, 1.0)
	if !fit {
		t.Fatalf("FitHomography rejected a clean translation (inliers=%d)", inl)
	}
	return h
}

func TestTrackAndHomography(t *testing.T) {
	h := trackAll(t)
	fx, fy := Apply(h, 160, 120)
	if math.Abs(fx-160-tdx) > 0.5 || math.Abs(fy-120-tdy) > 0.5 {
		t.Fatalf("H maps (160,120) to (%.2f,%.2f), want (%.2f,%.2f)", fx, fy, 160+tdx, 120+tdy)
	}
	if math.Abs(h[0]-1) > 0.01 || math.Abs(h[4]-1) > 0.01 || math.Abs(h[1]) > 0.01 || math.Abs(h[3]) > 0.01 {
		t.Fatalf("H is not a pure translation: %v", h)
	}
}

func TestInvertCompose(t *testing.T) {
	h := trackAll(t)
	round := Compose(h, Invert(h))
	for x := 20.0; x < 300; x += 40 {
		for y := 20.0; y < 220; y += 40 {
			fx, fy := Apply(round, x, y)
			if math.Abs(fx-x) > 1e-6 || math.Abs(fy-y) > 1e-6 {
				t.Fatalf("H*inv(H) maps (%v,%v) to (%.6f,%.6f)", x, y, fx, fy)
			}
		}
	}
	inv := Invert(h)
	mx, my := Apply(h, 160, 120)
	fx, fy := Apply(inv, mx, my)
	if math.Abs(fx-160) > 1e-6 || math.Abs(fy-120) > 1e-6 {
		t.Fatalf("inverse maps H(160,120)=(%.3f,%.3f) back to (%.6f,%.6f)", mx, my, fx, fy)
	}
}

func TestProjectiveRoundtrip(t *testing.T) {
	h := [9]float64{1.05, 0.002, 4, -0.004, 0.97, -3, 0.00004, -0.00002, 1}
	round := Compose(h, Invert(h))
	for x := 10.0; x < 310; x += 50 {
		for y := 10.0; y < 230; y += 50 {
			fx, fy := Apply(round, x, y)
			if math.Abs(fx-x) > 1e-6 || math.Abs(fy-y) > 1e-6 {
				t.Fatalf("roundtrip maps (%v,%v) to (%.6f,%.6f)", x, y, fx, fy)
			}
		}
	}
}

func TestBilinearRGB(t *testing.T) {
	rgb := make([]byte, 4*3)
	for i := 0; i < 4; i++ {
		rgb[i*3] = uint8(i * 40)
		rgb[i*3+1] = uint8(i * 20)
		rgb[i*3+2] = uint8(i * 10)
	}
	// 2x2 image: pixels (0,0,0), (40,20,10), (80,40,20), (120,60,30).
	r, g, b, inb := BilinearRGB(rgb, 2, 2, 0.5, 0.5)
	if !inb {
		t.Fatal("BilinearRGB reported out of bounds inside the image")
	}
	if r != 60 || g != 30 || b != 15 {
		t.Fatalf("BilinearRGB(0.5,0.5) = (%d,%d,%d), want (60,30,15)", r, g, b)
	}
	r, g, b, inb = BilinearRGB(rgb, 2, 2, 0, 0)
	if !inb || r != 0 || g != 0 || b != 0 {
		t.Fatalf("BilinearRGB(0,0) = (%d,%d,%d,%v), want (0,0,0,true)", r, g, b, inb)
	}
	if _, _, _, inb := BilinearRGB(rgb, 2, 2, 1.5, 0.5); inb {
		t.Fatal("BilinearRGB accepted x beyond the last bilinear cell")
	}
}

// coarseTex mimics the synth T1 background: smooth multi-octave value noise
// (feature period ~15-20px) plus a few sharper edges, at low contrast.
func coarseTex(x, y float64) float64 {
	return 127 + 45*math.Sin(x/11.0)*math.Cos(y/13.0) +
		25*math.Sin((x+2*y)/6.5) + 15*math.Sin(x/3.1)*math.Sin(y/2.7)
}

// TestTrackLKLargeShiftCoarseTexture pins the aperture limit: an 8.6px hop on
// coarse texture exceeds the capture range of a 7px half-window, so seeded
// tracking from a coarse model must bring the seed within range first.
func TestTrackLKLargeShiftCoarseTexture(t *testing.T) {
	const dx, dy = 8.6, 2.4
	prev := grayAt(func(x, y float64) float64 { return coarseTex(x, y) })
	cur := grayAt(func(x, y float64) float64 { return coarseTex(x-dx, y-dy) })
	pts := Corners(prev, tw, th, nil, 24, 400, 120)
	if len(pts) < 40 {
		t.Fatalf("Corners found only %d points", len(pts))
	}

	// Unseeded from the source position: expected to fall short.
	nxt, ok := TrackLK(prev, cur, tw, th, pts, 7, 25)
	var sum, n float64
	for i := range pts {
		if !ok[i] {
			continue
		}
		sum += math.Hypot(nxt[i].X-pts[i].X-dx, nxt[i].Y-pts[i].Y-dy)
		n++
	}
	t.Logf("unseeded: %d/%d tracked, mean err %.2f", int(n), len(pts), sum/n)

	// Seeded within 3px of the truth (what a coarse pyramid level delivers):
	// must converge tightly.
	seeds := make([]Pt, len(pts))
	for i, p := range pts {
		seeds[i] = Pt{p.X + dx - 3, p.Y + dy - 3}
	}
	nxt, ok = TrackLKSeeded(prev, cur, tw, th, pts, seeds, 7, 25)
	sum, n = 0, 0
	var worst float64
	for i := range pts {
		if !ok[i] {
			continue
		}
		e := math.Hypot(nxt[i].X-pts[i].X-dx, nxt[i].Y-pts[i].Y-dy)
		if e > worst {
			worst = e
		}
		sum += e
		n++
	}
	if n < 30 {
		t.Fatalf("seeded: only %d/%d tracked", int(n), len(pts))
	}
	if mean := sum / n; mean > 0.5 {
		t.Fatalf("seeded mean err %.2f exceeds 0.5px (worst %.2f)", mean, worst)
	}
}
