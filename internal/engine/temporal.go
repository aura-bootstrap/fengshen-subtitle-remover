package engine

import (
	"fmt"
	"image"
	"image/png"
	"math"
	"os"
	"sort"
	"strconv"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/alpha"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/ffx"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/grain"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/imgx"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/mask"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/motion"
)

// debugBand writes band frames to $DESUB_DBG frames lo..hi as PNGs (quality
// diagnostics only; set DESUB_DBG_LO / DESUB_DBG_HI).
func debugBand(dir string, lo, hi, i int, fill, alpha []uint8, w, h int) {
	if dir == "" || i < lo || i > hi {
		return
	}
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for p := 0; p < w*h; p++ {
		a := alpha[p]
		q := p * 3
		img.Pix[p*4] = fill[q]
		img.Pix[p*4+1] = fill[q+1]
		img.Pix[p*4+2] = fill[q+2]
		img.Pix[p*4+3] = a
	}
	f, err := os.Create(fmt.Sprintf("%s/band_%05d.png", dir, i))
	if err != nil {
		return
	}
	_ = png.Encode(f, img)
	_ = f.Close()
}

func debugCfg() (string, int, int) {
	dir := os.Getenv("DESUB_DBG")
	if dir == "" {
		return "", 0, -1
	}
	lo, hi := 0, 1<<30
	if v := os.Getenv("DESUB_DBG_LO"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			lo = n
		}
	}
	if v := os.Getenv("DESUB_DBG_HI"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			hi = n
		}
	}
	return dir, lo, hi
}

type TemporalOptions struct {
	Input     string
	Output    string
	W, H      int
	BandY     int
	BandH     int
	FPS       float64
	Masks     []mask.Frame // band coordinates, one per frame (repair mask)
	Cuts      []float64
	CRF       int
	Preset    string
	Neighbors int
	RegionPad int // expansion around the mask bbox used by the static-shot test
	Progress  func(frame, total int)
	// Motion enables L2 compensation: adjacent-frame homographies are
	// estimated first and moving shots are repaired by warping real pixels
	// from same-shot neighbours instead of diffusing.
	Motion bool
	// Stats, when non-nil (len == len(Masks)), receives per-frame fill
	// counts: how many masked pixels got real pixels versus diffusion.
	Stats []FrameStat
	// Alpha enables semi-transparent bar unmixing: frames are bar-detected
	// and unmixed (repair-mask pixels excluded) right after decoding, so
	// both the output and the neighbour sampling see recovered background.
	Alpha bool
	CharH int
	// Bars, when non-nil, receives the accepted bars per frame.
	Bars *[]alpha.FrameBar
	// Mags, when non-nil, receives the per-pair mean displacement (px) the
	// estimated homography induces on the four band corners; len == len(Masks).
	Mags *[]float64
	// EncColor carries ffx.MediaInfo.ColorEncodeArgs() output so the encode
	// preserves the input's colour space/transfer/primaries (R9.3).
	EncColor []string
	// Grain enables texture matching (internal/grain) inside the feathered
	// mask: noise injection, chroma alignment and, when the source measures
	// blocky, 8x8 boundary steps.
	Grain bool
}

// FrameStat records how a frame's masked pixels were repaired.
type FrameStat struct {
	Masked int
	Real   int
}

// motionPair is the homography from frame j to j+1; ok is false across shot
// cuts or when the estimate is unstable.
type motionPair struct {
	h  [9]float64
	ok bool
}

type ringSlot struct {
	raw  []byte
	bits []uint8
	bars []alpha.FrameBar
}

// RunTemporal repairs the band in Go. With Motion enabled, adjacent-frame
// homographies are estimated first and masked pixels are filled with real
// pixels warped from same-shot neighbours (median + consistency gate), long-
// range chains reaching outside the subtitle event; whatever cannot be filled
// falls back to spatial diffusion. Without Motion, static shots take direct
// medians of neighbouring frames and moving shots diffuse. The repaired band
// is alpha-blended back over the original with a feathered edge, then encoded
// once.
func RunTemporal(o TemporalOptions) error {
	return runCore(o, nil)
}

// runCore is RunTemporal with an optional generative-tier overlay: painted
// holds pre-repaired band frames (indexed by absolute frame number) whose
// masked pixels composite as-is instead of running the motion/diffuse tiers.
func runCore(o TemporalOptions, painted map[int][]byte) error {
	w, hb, by := o.W, o.BandH, o.BandY
	K := o.Neighbors
	if K < 1 {
		K = 6
	}
	if K > 31 {
		K = 31
	}
	// With motion compensation the ring must hold the long-range search
	// window: samples come from frames outside the subtitle event, so the
	// reach must exceed a typical event length.
	LR := K
	if o.Motion {
		LR = max(64, K)
	}
	ringN := 2*LR + 1
	total := len(o.Masks)
	if total == 0 {
		return fmt.Errorf("temporal: no frames to process")
	}

	shotID := make([]int, total)
	{
		sid, ci := 0, 0
		for f := 0; f < total; f++ {
			t := float64(f) / o.FPS
			for ci < len(o.Cuts) && o.Cuts[ci] <= t+1e-9 {
				sid++
				ci++
			}
			shotID[f] = sid
		}
	}

	// Mask bbox per frame (expanded): the static-shot test only looks here,
	// where smear would actually be visible.
	rx0 := make([]int32, total)
	ry0 := make([]int32, total)
	rx1 := make([]int32, total)
	ry1 := make([]int32, total)
	for i, m := range o.Masks {
		const big = 1 << 30
		minY, minX, maxY, maxX := big, big, -1, -1
		for k := 0; k+2 < len(m.RLE); k += 3 {
			y := int(m.RLE[k])
			x0 := int(m.RLE[k+1])
			x1 := int(m.RLE[k+2])
			if y < minY {
				minY = y
			}
			if y > maxY {
				maxY = y
			}
			if x0 < minX {
				minX = x0
			}
			if x1 > maxX {
				maxX = x1
			}
		}
		if maxY < 0 {
			rx1[i] = -1
			continue
		}
		p := o.RegionPad
		rx0[i] = int32(max(0, minX-p))
		ry0[i] = int32(max(0, minY-p))
		rx1[i] = int32(min(w, maxX+p))
		ry1[i] = int32(min(hb, maxY+p))
	}

	fr, err := ffx.NewFrameReader(o.Input, fmt.Sprintf("crop=%d:%d:0:%d,format=rgb24", w, hb, by), w, hb, "rgb24")
	if err != nil {
		return err
	}
	defer fr.Close()

	pairs, err := estimateMotion(o, w, hb, shotID)
	if err != nil {
		return err
	}
	if o.Mags != nil {
		mags := make([]float64, len(pairs))
		for i, p := range pairs {
			if p.ok {
				mags[i] = pairMag(p.h, w, hb)
			}
		}
		*o.Mags = mags
	}
	motionOn := o.Motion && len(pairs) > 0

	enc, err := ffx.NewEncoder(temporalEncoderArgs(o, w, hb, by))
	if err != nil {
		return err
	}

	ring := make([]ringSlot, ringN)
	for i := range ring {
		ring[i].raw = make([]byte, w*hb*3)
		ring[i].bits = make([]uint8, w*hb)
	}
	nextRead := 0
	eof := false
	readInto := func(i int) (bool, error) {
		s := &ring[i%ringN]
		ok, err := fr.Next(s.raw)
		if err != nil || !ok {
			return false, err
		}
		if i < total {
			o.Masks[i].Decode(s.bits, w)
			s.bars = s.bars[:0]
			if o.Alpha && !o.Masks[i].Empty() {
				s.bars = alpha.UnmixFrame(s.raw, w, hb, o.CharH, s.bits, i)
				if len(s.bars) > 0 && o.Bars != nil {
					*o.Bars = append(*o.Bars, s.bars...)
				}
			}
		} else {
			clear(s.bits)
		}
		return true, nil
	}

	rgba := make([]byte, w*hb*4)
	fill := make([]byte, w*hb*3)
	core := make([]uint8, w*hb)
	alpha := make([]uint8, w*hb)
	ablur := make([]uint8, w*hb)
	repBits := make([]uint8, w*hb)
	idx := make([]int32, 0, w*hb/8)

	for i := 0; i < total; i++ {
		for !eof && nextRead <= i+LR {
			ok, err := readInto(nextRead)
			if err != nil {
				return err
			}
			if !ok {
				eof = true
				break
			}
			nextRead++
		}
		if i >= nextRead {
			break
		}
		validEnd := nextRead
		if validEnd > total {
			validEnd = total
		}
		s := &ring[i%ringN]
		if o.Masks[i].Empty() {
			packRGBA(rgba, s.raw, nil, w*hb)
		} else if pf, okp := painted[i]; okp {
			// Generative tier: composite the sidecar frame through the same
			// feathered mask as the motion tiers; generated pixels do not
			// count as real. The sidecar sees the original (scrimped) strip,
			// so unmixed bar areas come from s.raw (already unmixed at
			// readInto) rather than the painted frame.
			o.Masks[i].Decode(repBits, w)
			imgx.MorphBin(repBits, core, w, hb, 3, 3, true)
			copy(fill, pf)
			for p, c := range core {
				if c != 0 {
					alpha[p] = 255
				} else {
					alpha[p] = 0
				}
			}
			for _, bar := range s.bars {
				r := bar.Rect
				y1 := r.Y + r.H
				if y1 > hb {
					y1 = hb
				}
				x1 := r.X + r.W
				if x1 > w {
					x1 = w
				}
				for y := r.Y; y < y1; y++ {
					for x := r.X; x < x1; x++ {
						p := y*w + x
						if core[p] == 0 {
							copy(fill[p*3:p*3+3], s.raw[p*3:p*3+3])
						}
						alpha[p] = 255
					}
				}
			}
			imgx.BoxBlur(alpha, w, hb, 2, ablur)
			if o.Grain {
				grain.Match(fill, w, hb, ablur, s.raw, true)
			}
			packRGBA(rgba, fill, ablur, w*hb)
			if o.Stats != nil {
				nMask := 0
				for _, b := range repBits {
					if b != 0 {
						nMask++
					}
				}
				o.Stats[i] = FrameStat{Masked: nMask}
			}
		} else {
			o.Masks[i].Decode(repBits, w)
			copy(fill, s.raw)
			idx = idx[:0]
			nReal := 0
			if motionOn {
				// Chains are ~identity on static shots, so the warped path
				// subsumes the direct one and adds long-range sampling.
				idx, nReal = warpFill(fill, ring, ringN, pairs, i, K, LR, validEnd, shotID, repBits, w, hb, idx)
			} else {
				static := false
				if i > 0 && shotID[i] == shotID[i-1] {
					prev := &ring[(i-1)%ringN]
					static = regionMatches(s.raw, prev.raw, s.bits, prev.bits, w,
						int(rx0[i]), int(ry0[i]), int(rx1[i]), int(ry1[i]))
				}
				if static && rx1[i] > 0 && rx1[i-1] > 0 {
					idx, nReal = temporalFill(fill, ring, ringN, i, K, validEnd, shotID, repBits, w*hb, idx)
				} else {
					for p, b := range repBits {
						if b != 0 {
							idx = append(idx, int32(p))
						}
					}
				}
			}
			if len(idx) > 0 {
				diffuse(fill, w, hb, idx, 64)
			}
			imgx.MorphBin(repBits, core, w, hb, 3, 3, true)
			for p, c := range core {
				if c != 0 {
					alpha[p] = 255
				} else {
					alpha[p] = 0
				}
			}
			// Unmixed bar pixels carry the recovered background in fill;
			// without this they would composite at alpha=0 and the original
			// scrim would show through.
			for _, bar := range s.bars {
				r := bar.Rect
				y1 := r.Y + r.H
				if y1 > hb {
					y1 = hb
				}
				x1 := r.X + r.W
				if x1 > w {
					x1 = w
				}
				for y := r.Y; y < y1; y++ {
					for x := r.X; x < x1; x++ {
						alpha[y*w+x] = 255
					}
				}
			}
			imgx.BoxBlur(alpha, w, hb, 2, ablur)
			if o.Grain {
				grain.Match(fill, w, hb, ablur, s.raw, true)
			}
			packRGBA(rgba, fill, ablur, w*hb)
			if o.Stats != nil {
				nMask := 0
				for _, b := range repBits {
					if b != 0 {
						nMask++
					}
				}
				o.Stats[i] = FrameStat{Masked: nMask, Real: nReal}
			}
		}
		if dbgDir, dbgLo, dbgHi := debugCfg(); dbgDir != "" {
			debugBand(dbgDir, dbgLo, dbgHi, i, fill, ablur, w, hb)
		}
		if _, err := enc.Write(rgba); err != nil {
			return err
		}
		if o.Progress != nil && (i%200 == 0 || i == total-1) {
			o.Progress(i, total)
		}
	}
	return enc.Close()
}

func temporalEncoderArgs(o TemporalOptions, w, hb, by int) []string {
	args := []string{"-hide_banner", "-nostdin", "-y", "-loglevel", "error",
		"-i", o.Input,
		"-f", "rawvideo", "-pix_fmt", "rgba", "-s", fmt.Sprintf("%dx%d", w, hb),
		"-r", fmt.Sprintf("%.6f", o.FPS), "-i", "pipe:0",
		"-filter_complex", fmt.Sprintf("[0:v][1:v]overlay=0:%d:eof_action=pass:format=auto,format=yuv420p[v]", by),
		"-map", "[v]", "-map", "0:a?"}
	args = append(args, o.EncColor...)
	return append(args,
		"-c:v", "libx264", "-crf", fmt.Sprint(o.CRF), "-preset", o.Preset,
		"-threads", fmt.Sprint(ffx.CPUWorkers()),
		"-c:a", "copy", "-movflags", "+faststart", o.Output)
}

func packRGBA(dst, rgb []byte, alpha []uint8, n int) {
	if alpha == nil {
		for p := 0; p < n; p++ {
			q := p * 3
			r := p * 4
			dst[r], dst[r+1], dst[r+2] = rgb[q], rgb[q+1], rgb[q+2]
			dst[r+3] = 0
		}
		return
	}
	for p := 0; p < n; p++ {
		q := p * 3
		r := p * 4
		dst[r], dst[r+1], dst[r+2] = rgb[q], rgb[q+1], rgb[q+2]
		dst[r+3] = alpha[p]
	}
}

// regionMatches reports whether two consecutive frames are effectively
// identical outside the masked areas inside the given rectangle. Judging
// locally (and by the fraction of clearly changed pixels rather than the
// mean) rejects sub-pixel drift that would otherwise streak the temporal
// median fill.
func regionMatches(cur, prev, curBits, prevBits []uint8, w, x0, y0, x1, y1 int) bool {
	const delta = 8
	const maxFrac = 0.01
	if x0 >= x1 || y0 >= y1 {
		return false
	}
	changed, cnt := 0, 0
	for y := y0; y < y1; y++ {
		row := y * w
		for x := x0; x < x1; x++ {
			p := row + x
			if curBits[p] != 0 || prevBits[p] != 0 {
				continue
			}
			q := p * 3
			lc := (int(cur[q]) + int(cur[q+1]) + int(cur[q+2])) / 3
			lp := (int(prev[q]) + int(prev[q+1]) + int(prev[q+2])) / 3
			d := lc - lp
			if d < 0 {
				d = -d
			}
			if d > delta {
				changed++
			}
			cnt++
		}
	}
	if cnt < 32 {
		return false
	}
	return float64(changed)/float64(cnt) <= maxFrac
}

// temporalFill fills masked pixels from the median of same-shot neighbour
// frames where that pixel is visible; pixels with fewer than 3 samples are
// appended to fallback instead.
func temporalFill(fill []byte, ring []ringSlot, ringN, i, K, validEnd int, shotID []int, curBits []uint8, n int, fallback []int32) ([]int32, int) {
	var rs, gs, bs [64]uint8
	nReal := 0
	for p := 0; p < n; p++ {
		if curBits[p] == 0 {
			continue
		}
		cnt := 0
		for d := -K; d <= K; d++ {
			f := i + d
			if d == 0 || f < 0 || f >= validEnd || shotID[f] != shotID[i] {
				continue
			}
			sl := &ring[f%ringN]
			if sl.bits[p] != 0 {
				continue
			}
			raw := sl.raw
			rs[cnt], gs[cnt], bs[cnt] = raw[p*3], raw[p*3+1], raw[p*3+2]
			cnt++
		}
		if cnt >= 3 {
			fill[p*3] = median(rs[:cnt])
			fill[p*3+1] = median(gs[:cnt])
			fill[p*3+2] = median(bs[:cnt])
			nReal++
		} else {
			fallback = append(fallback, int32(p))
		}
	}
	return fallback, nReal
}

// EstimateMags runs only the motion-estimation pass and returns the mean
// homography displacement per adjacent frame pair (0 for invalid pairs).
// Routing needs these magnitudes before the fill run to grade events.
func EstimateMags(o TemporalOptions) ([]float64, error) {
	total := len(o.Masks)
	shotID := make([]int, total)
	sid, ci := 0, 0
	for f := 0; f < total; f++ {
		t := float64(f) / o.FPS
		for ci < len(o.Cuts) && o.Cuts[ci] <= t+1e-9 {
			sid++
			ci++
		}
		shotID[f] = sid
	}
	pairs, err := estimateMotion(o, o.W, o.BandH, shotID)
	if err != nil {
		return nil, err
	}
	mags := make([]float64, len(pairs))
	for i, p := range pairs {
		if p.ok {
			mags[i] = pairMag(p.h, o.W, o.BandH)
		}
	}
	return mags, nil
}

// estimateMotion decodes the band once and estimates a homography for every
// adjacent frame pair (pass 1 of the motion-compensated repair). Corners are
// picked on the previous frame outside its subtitle mask so the burned-in
// text itself is never tracked. Pairs spanning a shot cut stay invalid.
func estimateMotion(o TemporalOptions, w, hb int, shotID []int) ([]motionPair, error) {
	n := len(o.Masks)
	pairs := make([]motionPair, n)
	if !o.Motion || n < 2 {
		return pairs, nil
	}
	fr, err := ffx.NewFrameReader(o.Input, fmt.Sprintf("crop=%d:%d:0:%d,format=rgb24", w, hb, o.BandY), w, hb, "rgb24")
	if err != nil {
		return nil, err
	}
	defer fr.Close()
	prevRaw := make([]byte, w*hb*3)
	curRaw := make([]byte, w*hb*3)
	prevGray := make([]uint8, w*hb)
	curGray := make([]uint8, w*hb)
	prevBits := make([]uint8, w*hb)
	// Quarter-resolution coarse-to-fine fit: fast camera moves exceed the
	// single-level LK basin at full resolution (and flat-region windows then
	// lock at zero displacement, letting RANSAC accept an identity H), so the
	// primary fit tracks at quarter resolution where displacement is divided
	// by four. The lifted homography must still prove itself at full
	// resolution before the pair is trusted.
	fallbackFit := func() ([9]float64, bool) {
		pqd, wq, hq := motion.DownsampleGray(prevGray, w, hb, 4)
		bqd, _, _ := motion.DownsampleBits(prevBits, w, hb, 4)
		cqd, _, _ := motion.DownsampleGray(curGray, w, hb, 4)
		pts := motion.Corners(pqd, wq, hq, bqd, 6, 300, 8)
		if len(pts) < 16 {
			return [9]float64{}, false
		}
		nxt, okf := motion.TrackLK(pqd, cqd, wq, hq, pts, 7, 25)
		hq4, _, fit := motion.FitHomography(pts, nxt, okf, 2.0)
		if !fit {
			return [9]float64{}, false
		}
		// Lift H back to full resolution: H = S · Hq · S⁻¹.
		S := [9]float64{4, 0, 0, 0, 4, 0, 0, 0, 1}
		Si := [9]float64{0.25, 0, 0, 0, 0.25, 0, 0, 0, 1}
		h := motion.Compose(motion.Compose(S, hq4), Si)
		dbg := os.Getenv("DESUB_DBG") != ""
		if dbg {
			fmt.Printf("  qres tx=%.2f ty=%.2f\n", h[2], h[5])
		}
		// Full-res refinement: re-track from the lifted predictions and refit.
		// Two rounds, wide gate first: a biased seed must not lock the fit —
		// flat-region windows barely move from their seed, so a tight gate on
		// round one would keep only those and preserve the quarter-res bias.
		// The wide round lets textured points pull H to the truth; the tight
		// round then polishes it.
		src, seed := liftSeeds(h, pts)
		for _, tol := range []float64{3.0, 2.0, 1.25, 0.9} {
			tr, ok := motion.TrackLKSeeded(prevGray, curGray, w, hb, src, seed, 7, 40)
			hr, nin, fit2 := motion.FitHomography(src, tr, ok, tol)
			if dbg {
				nok := 0
				for _, f := range ok {
					if f {
						nok++
					}
				}
				fmt.Printf("  refine tol=%.2f tracked=%d inliers=%d fit=%v tx=%.2f ty=%.2f\n", tol, nok, nin, fit2, hr[2], hr[5])
			}
			if !fit2 {
				break
			}
			h = hr
			for i, p := range src {
				sx, sy := motion.Apply(h, p.X, p.Y)
				seed[i] = motion.Pt{X: sx, Y: sy}
			}
		}
		if !verifyFullRes(h, pts, prevGray, curGray, w, hb) {
			return [9]float64{}, false
		}
		return h, true
	}
	ok1, err := fr.Next(prevRaw)
	if err != nil || !ok1 {
		return pairs, err
	}
	motion.Luma(prevRaw, prevGray)
	o.Masks[0].Decode(prevBits, w)
	for j := 1; j < n; j++ {
		ok2, err := fr.Next(curRaw)
		if err != nil {
			return nil, err
		}
		if !ok2 {
			break
		}
		motion.Luma(curRaw, curGray)
		if shotID[j] == shotID[j-1] {
			pts := motion.Corners(prevGray, w, hb, prevBits, 24, 500, 120)
			var mp motionPair
			if h, ok := fallbackFit(); ok {
				mp = motionPair{h, true}
			}
			fitPyramid := mp.ok
			nFull := 0
			if !mp.ok && len(pts) >= 16 {
				nxt, okf := motion.TrackLK(prevGray, curGray, w, hb, pts, 7, 25)
				for _, f := range okf {
					if f {
						nFull++
					}
				}
				h, _, fit := motion.FitHomography(pts, nxt, okf, 1.5)
				if fit {
					mp = motionPair{h, true}
				}
			}
			if dbg := os.Getenv("DESUB_DBG"); dbg != "" {
				fmt.Printf("motion %d->%d: corners=%d tracked=%d pyramid=%v fullres=%v tx=%.2f ty=%.2f\n",
					j-1, j, len(pts), nFull, fitPyramid, mp.ok && !fitPyramid, mp.h[2], mp.h[5])
			}
			pairs[j-1] = mp
		}
		prevRaw, curRaw = curRaw, prevRaw
		prevGray, curGray = curGray, prevGray
		o.Masks[j].Decode(prevBits, w)
	}
	return pairs, nil
}

// liftSeeds scales quarter-res corner points to full resolution and projects
// them through H; the projected positions seed a full-res LK pass.
func liftSeeds(h [9]float64, qpts []motion.Pt) (src, seed []motion.Pt) {
	for _, q := range qpts {
		p := motion.Pt{X: q.X*4 + 1.5, Y: q.Y*4 + 1.5}
		sx, sy := motion.Apply(h, p.X, p.Y)
		src = append(src, p)
		seed = append(seed, motion.Pt{X: sx, Y: sy})
	}
	return src, seed
}

// verifyFullRes checks a quarter-resolution homography at full resolution:
// the fallback corners are lifted, projected through H, and re-tracked from
// that seed. A median residual above 1.5px means the lifted H is wrong and
// the pair must stay invalid — an inaccurate H on a fast-moving shot is
// worse than none at all, because its warp samples land on wrong content
// that can still pass the agreement gate.
func verifyFullRes(h [9]float64, qpts []motion.Pt, prevGray, curGray []uint8, w, hb int) bool {
	src, seed := liftSeeds(h, qpts)
	tracked, ok := motion.TrackLKSeeded(prevGray, curGray, w, hb, src, seed, 7, 15)
	var res []float64
	for i := range src {
		if !ok[i] {
			continue
		}
		res = append(res, math.Hypot(tracked[i].X-seed[i].X, tracked[i].Y-seed[i].Y))
	}
	if len(res) < 8 {
		return false
	}
	sort.Float64s(res)
	return res[len(res)/2] <= 1.5
}

type warpChain struct {
	f int
	h [9]float64
}

// pairMag returns the mean displacement (px) the homography induces on the
// four band corners — the motion-magnitude feature used by routing.
func pairMag(h [9]float64, w, hb int) float64 {
	apply := func(x, y float64) (float64, float64) {
		d := h[6]*x + h[7]*y + h[8]
		if d == 0 {
			d = 1e-9
		}
		return (h[0]*x + h[1]*y + h[2]) / d, (h[3]*x + h[4]*y + h[5]) / d
	}
	var s float64
	for _, c := range [][2]float64{{0, 0}, {float64(w), 0}, {0, float64(hb)}, {float64(w), float64(hb)}} {
		u, v := apply(c[0], c[1])
		s += math.Hypot(u-c[0], v-c[1])
	}
	return s / 4
}

// snapChainH quantizes an almost-pure-translation composed homography to an
// integer shift. Pairwise LK estimates carry a small per-hop bias that
// accumulates linearly over long chains; when the composed matrix is a
// near-translation whose offset sits within 0.35px of an integer (digital
// pans and static shots land exactly on integers), snapping recovers the
// exact shift instead of the drifted estimate. Genuine sub-pixel camera
// motion fails the gate and keeps the estimate.
func snapChainH(h [9]float64) [9]float64 {
	const linTol, posTol = 0.01, 0.35
	if math.Abs(h[0]-1) > linTol || math.Abs(h[1]) > linTol ||
		math.Abs(h[3]) > linTol || math.Abs(h[4]-1) > linTol ||
		math.Abs(h[6]) > 1e-4 || math.Abs(h[7]) > 1e-4 {
		return h
	}
	rx, ry := math.Round(h[2]), math.Round(h[5])
	if math.Abs(h[2]-rx) > posTol || math.Abs(h[5]-ry) > posTol {
		return h
	}
	h[2], h[5] = rx, ry
	return h
}

// buildChains composes pairwise homographies from frame i out to each of
// its same-shot neighbours out to reach frames. Composition stops at invalid
// pairs or shot boundaries.
func buildChains(pairs []motionPair, i, K, reach, step, validEnd int, shotID []int) []warpChain {
	var out []warpChain
	add := func(f int, h [9]float64) {
		d := f - i
		if d < 0 {
			d = -d
		}
		if d <= K || (d-K)%step == 0 {
			out = append(out, warpChain{f, snapChainH(h)})
		}
	}
	c := motion.Identity
	for f := i + 1; f <= i+reach && f < validEnd; f++ {
		p := pairs[f-1]
		if !p.ok || shotID[f] != shotID[i] {
			break
		}
		c = motion.Compose(p.h, c)
		add(f, c)
	}
	c = motion.Identity
	for f := i - 1; f >= i-reach && f >= 0; f-- {
		p := pairs[f]
		if !p.ok || shotID[f] != shotID[i] {
			break
		}
		c = motion.Compose(motion.Invert(p.h), c)
		add(f, c)
	}
	sort.Slice(out, func(a, b int) bool {
		da, db := out[a].f-i, out[b].f-i
		if da < 0 {
			da = -da
		}
		if db < 0 {
			db = -db
		}
		return da < db
	})
	return out
}

// collectSamples gathers up to maxCnt warp samples for pixel p from chains.
// The background behind the subtitle at p has moved in the neighbour, so
// visibility is tested at the sampled position, not at p.
func collectSamples(cs []warpChain, ring []ringSlot, ringN, p, w, h int, x, y float64, rs, gs, bs *[24]uint8, maxCnt int) int {
	cnt := 0
	for _, ch := range cs {
		if cnt >= maxCnt {
			break
		}
		sl := &ring[ch.f%ringN]
		sx, sy := motion.Apply(ch.h, x, y)
		ix, iy := int(sx), int(sy)
		if ix < 0 || iy < 0 || ix >= w || iy >= h {
			continue
		}
		if sl.bits[iy*w+ix] != 0 {
			continue
		}
		r, g, b, inb := motion.BilinearRGB(sl.raw, w, h, sx, sy)
		if !inb {
			continue
		}
		rs[cnt], gs[cnt], bs[cnt] = r, g, b
		cnt++
	}
	return cnt
}

// madOK reports whether the samples sit tightly around their median; a large
// spread means the sources disagree (occlusion or a bad warp) and the pixel
// is better left to diffusion.
func madOK(v []uint8, med uint8) bool {
	const spread = 6
	var d [24]int
	n := 0
	for _, x := range v {
		dd := int(x) - int(med)
		if dd < 0 {
			dd = -dd
		}
		j := n
		for j > 0 && d[j-1] > dd {
			d[j] = d[j-1]
			j--
		}
		d[j] = dd
		n++
	}
	return d[n/2] <= spread
}

// warpFill repairs masked pixels of a moving shot by sampling same-shot
// neighbours through the chained homographies (nearest first), accepting the
// median when at least 3 samples agree. Pixels the near window cannot fill
// get a second chance against the long-range chains (background visible only
// well before/after the event); anything still unfilled returns in fallback
// for spatial diffusion.
func warpFill(fill []byte, ring []ringSlot, ringN int, pairs []motionPair, i, K, LR, validEnd int, shotID []int, curBits []uint8, w, h int, fallback []int32) ([]int32, int) {
	chains := buildChains(pairs, i, K, LR, 4, validEnd, shotID)
	var win, lng []warpChain
	for _, ch := range chains {
		d := ch.f - i
		if d < 0 {
			d = -d
		}
		if d <= K {
			win = append(win, ch)
		} else {
			lng = append(lng, ch)
		}
	}
	var rs, gs, bs [24]uint8
	nReal := 0
	fallback = fallback[:0]
	try := func(cs []warpChain, maxCnt int, p int, x, y float64) bool {
		cnt := collectSamples(cs, ring, ringN, p, w, h, x, y, &rs, &gs, &bs, maxCnt)
		if cnt < 3 {
			return false
		}
		mr, mg, mb := median(rs[:cnt]), median(gs[:cnt]), median(bs[:cnt])
		if madOK(rs[:cnt], mr) && madOK(gs[:cnt], mg) && madOK(bs[:cnt], mb) {
			fill[p*3], fill[p*3+1], fill[p*3+2] = mr, mg, mb
			return true
		}
		return false
	}
	for p := 0; p < len(curBits); p++ {
		if curBits[p] == 0 {
			continue
		}
		x, y := float64(p%w), float64(p/w)
		if try(win, 12, p, x, y) {
			nReal++
			continue
		}
		fallback = append(fallback, int32(p))
	}
	if len(fallback) > 0 && len(lng) > 0 {
		keep := fallback[:0]
		for _, p32 := range fallback {
			p := int(p32)
			x, y := float64(p%w), float64(p/w)
			if try(lng, 6, p, x, y) {
				nReal++
				continue
			}
			keep = append(keep, p32)
		}
		fallback = keep
	}
	return fallback, nReal
}

func median(v []uint8) uint8 {
	for i := 1; i < len(v); i++ {
		x := v[i]
		j := i - 1
		for j >= 0 && v[j] > x {
			v[j+1] = v[j]
			j--
		}
		v[j+1] = x
	}
	return v[len(v)/2]
}

// diffuse fills pixels by successive over-relaxation of Laplace's equation,
// with unmasked pixels acting as fixed boundary values.
func diffuse(fill []byte, w, h int, px []int32, iters int) {
	const omega = 1.7
	for it := 0; it < iters; it++ {
		maxd := 0.0
		for _, p32 := range px {
			p := int(p32)
			x, y := p%w, p/w
			var sr, sg, sb, cnt int
			if x > 0 {
				q := (p - 1) * 3
				sr += int(fill[q])
				sg += int(fill[q+1])
				sb += int(fill[q+2])
				cnt++
			}
			if x < w-1 {
				q := (p + 1) * 3
				sr += int(fill[q])
				sg += int(fill[q+1])
				sb += int(fill[q+2])
				cnt++
			}
			if y > 0 {
				q := (p - w) * 3
				sr += int(fill[q])
				sg += int(fill[q+1])
				sb += int(fill[q+2])
				cnt++
			}
			if y < h-1 {
				q := (p + w) * 3
				sr += int(fill[q])
				sg += int(fill[q+1])
				sb += int(fill[q+2])
				cnt++
			}
			if cnt == 0 {
				continue
			}
			o0 := float64(fill[p*3])
			o1 := float64(fill[p*3+1])
			o2 := float64(fill[p*3+2])
			n0 := o0 + omega*(float64(sr)/float64(cnt)-o0)
			n1 := o1 + omega*(float64(sg)/float64(cnt)-o1)
			n2 := o2 + omega*(float64(sb)/float64(cnt)-o2)
			d := math.Max(math.Abs(n0-o0), math.Max(math.Abs(n1-o1), math.Abs(n2-o2)))
			if d > maxd {
				maxd = d
			}
			fill[p*3], fill[p*3+1], fill[p*3+2] = clamp8(n0), clamp8(n1), clamp8(n2)
		}
		if maxd < 0.5 {
			break
		}
	}
}

func clamp8(v float64) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v + 0.5)
}
