package engine

import (
	"math"
	"testing"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/motion"
)

func transH(dx, dy float64) [9]float64 {
	return [9]float64{1, 0, dx, 0, 1, dy, 0, 0, 1}
}

// TestVerifyFullRes replays the quarter-resolution fallback on a large
// synthetic shift: the lifted homography must pass the full-res residual
// check, while one biased by 5px must be rejected.
func TestVerifyFullRes(t *testing.T) {
	const w, h = 320, 240
	const dx, dy = 8.0, -4.0
	// Smoother texture than motion_test's: after 4x box downsampling the
	// dominant wavelength stays ~19 quarter-px, keeping the LK basin (~±4.7)
	// comfortably wider than the quarter-res shift (2.0, -1.0).
	tex := func(x, y float64) float64 {
		return 127 + 60*math.Sin(x/12)*math.Sin(y/12) + 30*math.Sin((x+y)/18)
	}
	gray := func(fn func(x, y float64) float64) []uint8 {
		g := make([]uint8, w*h)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				v := fn(float64(x), float64(y)) + 0.5
				if v < 0 {
					v = 0
				}
				if v > 255 {
					v = 255
				}
				g[y*w+x] = uint8(v)
			}
		}
		return g
	}
	prev := gray(tex)
	cur := gray(func(x, y float64) float64 { return tex(x-dx, y-dy) })

	pq, wq, hq := motion.DownsampleGray(prev, w, h, 4)
	cq, _, _ := motion.DownsampleGray(cur, w, h, 4)
	pts := motion.Corners(pq, wq, hq, nil, 6, 300, 8)
	if len(pts) < 16 {
		t.Fatalf("only %d corners", len(pts))
	}
	nxt, okf := motion.TrackLK(pq, cq, wq, hq, pts, 7, 25)
	hq4, _, fit := motion.FitHomography(pts, nxt, okf, 2.0)
	if !fit {
		t.Fatal("quarter-res fit failed on a clean shift")
	}
	S := [9]float64{4, 0, 0, 0, 4, 0, 0, 0, 1}
	Si := [9]float64{0.25, 0, 0, 0, 0.25, 0, 0, 0, 1}
	lifted := motion.Compose(motion.Compose(S, hq4), Si)

	// Mirror fallbackFit: refine at full res from the coarse seeds, then gate.
	src, seed := liftSeeds(lifted, pts)
	tr, ok := motion.TrackLKSeeded(prev, cur, w, h, src, seed, 7, 25)
	refined, _, fit2 := motion.FitHomography(src, tr, ok, 1.5)
	if !fit2 {
		t.Fatal("full-res refinement failed on a clean shift")
	}

	fx, fy := motion.Apply(refined, 160, 120)
	if math.Abs(fx-160-dx) > 1.0 || math.Abs(fy-120-dy) > 1.0 {
		t.Fatalf("refined H maps to (%.2f,%.2f), want (%.1f,%.1f)", fx, fy, 160+dx, 120+dy)
	}
	if !verifyFullRes(refined, pts, prev, cur, w, h) {
		t.Fatal("verifyFullRes rejected an accurate homography")
	}
	bad := motion.Compose(transH(5, 0), refined)
	if verifyFullRes(bad, pts, prev, cur, w, h) {
		t.Fatal("verifyFullRes accepted a 5px-off homography")
	}
}

func TestBuildChains(t *testing.T) {
	// Camera pans left 1px/frame: pairs[j] maps frame j -> j+1 by (-1, 0).
	const n = 8
	pairs := make([]motionPair, n)
	for j := range pairs {
		pairs[j] = motionPair{transH(-1, 0), true}
	}
	shotID := make([]int, n)
	chains := buildChains(pairs, 4, 1, 3, 4, n, shotID)
	if len(chains) == 0 {
		t.Fatal("no chains built from valid pairs")
	}
	for _, ch := range chains {
		want := -(float64(ch.f) - 4)
		gx, gy := motion.Apply(ch.h, 10, 10)
		if math.Abs(gx-(10+want)) > 1e-9 || math.Abs(gy-10) > 1e-9 {
			t.Fatalf("chain to f=%d maps (10,10) to (%.3f,%.3f), want (%.0f,10)", ch.f, gx, gy, 10+want)
		}
	}
	// An invalid pair breaks the chain beyond it.
	pairs[4].ok = false
	chains = buildChains(pairs, 4, 1, 3, 4, n, shotID)
	for _, ch := range chains {
		if ch.f > 4 {
			t.Fatalf("chain crossed invalid pair: f=%d", ch.f)
		}
	}
	// A cut breaks it too.
	pairs[4].ok = true
	shotID[4] = 1
	chains = buildChains(pairs, 4, 1, 3, 4, n, shotID)
	for _, ch := range chains {
		if ch.f > 4 {
			t.Fatalf("chain crossed shot cut: f=%d", ch.f)
		}
	}
}

// TestWarpFill fills a masked pixel by tracking the background through a
// left-panning camera: frame f shows world coordinate W at x = W - f, so the
// neighbour samples must all land on the same world value.
func TestWarpFill(t *testing.T) {
	const (
		w  = 20
		h  = 10
		i  = 2
		K  = 2
		LR = 2
		n  = 5
	)
	ringN := 2*LR + 1
	ring := make([]ringSlot, ringN)
	pairs := make([]motionPair, n)
	shotID := make([]int, n)
	for f := 0; f < n; f++ {
		ring[f%ringN].raw = make([]byte, w*h*3)
		ring[f%ringN].bits = make([]uint8, w*h)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				v := uint8((x + f) * 10)
				p := (y*w + x) * 3
				ring[f%ringN].raw[p] = v
				ring[f%ringN].raw[p+1] = v
				ring[f%ringN].raw[p+2] = v
			}
		}
		if f+1 < n {
			pairs[f] = motionPair{transH(-1, 0), true}
		}
	}
	rep := make([]uint8, w*h)
	fill := make([]byte, w*h*3)
	copy(fill, ring[i%ringN].raw)

	px := 5*w + 5 // repair target at (x=5,y=5)
	rep[px] = 1
	// A second pixel at (x=8,y=5) whose sampled positions are masked in
	// every neighbour.
	px2 := 5*w + 8
	rep[px2] = 1
	for f := 0; f < n; f++ {
		if f == i {
			continue
		}
		// sample position of px2 under chain i->f is x = 8-(f-i)
		for d := -LR; d <= LR; d++ {
			if d == 0 {
				continue
			}
			sx := 8 - d
			if sx >= 0 && sx < w {
				ring[(i+d)%ringN].bits[5*w+sx] = 1
			}
		}
	}

	fb, nReal := warpFill(fill, ring, ringN, pairs, i, K, LR, n, shotID, rep, w, h, nil)
	if nReal != 1 {
		t.Fatalf("nReal = %d, want 1", nReal)
	}
	want := uint8((5 + i) * 10)
	got := fill[px*3]
	if got != want {
		t.Fatalf("filled value = %d, want %d", got, want)
	}
	found := false
	for _, p := range fb {
		if int(p) == px {
			found = true
		}
	}
	if found {
		t.Fatal("repairable pixel ended up in fallback")
	}
	found = false
	for _, p := range fb {
		if int(p) == px2 {
			found = true
		}
	}
	if !found {
		t.Fatal("fully occluded pixel should fall back to diffusion")
	}
}

func TestMadOK(t *testing.T) {
	if !madOK([]uint8{10, 10, 10, 11}, 10) {
		t.Fatal("tight cluster rejected")
	}
	// A single outlier is fine: the median ignores it.
	if !madOK([]uint8{10, 60, 10, 12}, 12) {
		t.Fatal("cluster with one outlier rejected")
	}
	// No majority agreement must be rejected.
	if madOK([]uint8{10, 40, 45, 12}, 40) {
		t.Fatal("spread cluster accepted")
	}
}

func TestEncoderArgsCarryColorMetadata(t *testing.T) {
	o := TemporalOptions{
		Input: "in.mp4", Output: "out.mp4", FPS: 25, CRF: 17, Preset: "medium",
		EncColor: []string{"-colorspace", "bt709", "-color_trc", "bt709"},
	}
	args := temporalEncoderArgs(o, 640, 100, 260)
	joined := ""
	for i, a := range args {
		if a == "-colorspace" && i+1 < len(args) && args[i+1] == "bt709" {
			joined = "ok"
		}
	}
	if joined != "ok" {
		t.Fatalf("colorspace flag missing from encoder args: %v", args)
	}
	// Colour flags must precede the codec selection so ffmpeg treats them as
	// output options for the video stream.
	var ci, vi int = -1, -1
	for i, a := range args {
		switch a {
		case "-colorspace":
			ci = i
		case "-c:v":
			vi = i
		}
	}
	if ci < 0 || vi < 0 || ci > vi {
		t.Fatalf("colour flags misplaced: colorspace@%d c:v@%d", ci, vi)
	}
}
