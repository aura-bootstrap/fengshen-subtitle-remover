// Package alpha detects semi-transparent backdrop bars behind subtitles and
// unmixes them: with the bar's alpha and foreground colour estimated by
// least squares, the original background is recovered as (I − αF)/(1 − α).
// Stroke cores (α ≈ 1) are not unmixable and stay in the repair mask.
package alpha

import (
	"math"
	"sort"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/imgx"
)

// Bar is one detected backdrop bar in band coordinates.
type Bar struct {
	Rect  imgx.Rect
	Alpha float64
	Fore  [3]uint8
	Resid float64 // mean absolute fit residual; caller drops bars above threshold
}

// rowEnergy computes the median absolute gradient per row — a
// semi-transparent bar flattens the background texture, so its rows sit well
// below the band baseline. The median (not the mean) keeps subtitle strokes
// inside the bar from inflating the row: they occupy a minority of columns.
func rowEnergy(gray []uint8, w, h int) []float64 {
	e := make([]float64, h)
	row := make([]float64, 0, w-2)
	for y := 1; y < h-1; y++ {
		row = row[:0]
		for x := 1; x < w-1; x++ {
			p := y*w + x
			row = append(row, math.Abs(float64(gray[p+1])-float64(gray[p-1]))+
				math.Abs(float64(gray[p+w])-float64(gray[p-w])))
		}
		e[y] = medianF(row)
	}
	e[0], e[h-1] = e[1], e[h-2]
	return e
}

func medianF(v []float64) float64 {
	s := append([]float64(nil), v...)
	for i := 1; i < len(s); i++ {
		x := s[i]
		j := i - 1
		for j >= 0 && s[j] > x {
			s[j+1] = s[j]
			j--
		}
		s[j+1] = x
	}
	return s[len(s)/2]
}

// Scrims are commonly composited in YUV (subtitle renderers overlay on the
// YUV planes), where luma and chroma get different blend factors. In RGB such
// a scrim is NOT an affine blend per channel and any per-channel RGB fit is
// systematically biased — in practice the blue channel collapses. The fit and
// the unmixing therefore run in YUV; the affine map preserves each plane's
// blend factor, so RGB-composited scrims stay equally well modelled.
func rgbToYUV(r, g, b float64) (y, u, v float64) {
	y = 0.299*r + 0.587*g + 0.114*b
	u = -0.168736*r - 0.331264*g + 0.5*b + 128
	v = 0.5*r - 0.418688*g - 0.081312*b + 128
	return
}

func yuvToRGB(y, u, v float64) (r, g, b float64) {
	u -= 128
	v -= 128
	r = y + 1.402*v
	g = y - 0.344136*u - 0.714136*v
	b = y + 1.772*u
	return
}

// DetectBars finds horizontal bars bracketed by a pair of strong straight
// edges with a plausible height. The straight edge is the reliable scrim
// signature; interior contrast depression is only a secondary validation,
// because over an already-flat background a bar has nothing to depress.
// Candidates are deliberately loose — Unmix's fit residual is the strict
// gate that rejects regions the alpha model cannot explain.
func DetectBars(gray []uint8, w, h, charH int) []Bar {
	if charH < 8 {
		charH = 8
	}
	e := rowEnergy(gray, w, h)
	type edge struct {
		y, x0, x1, n int
	}
	var edges []edge
	for y := 0; y < h-1; y++ {
		x0, x1, n, ok := edgeSpan(gray, w, h, y, 1)
		if ok && n >= w/3 && x1-x0 >= 3*charH {
			edges = append(edges, edge{y, x0, x1, n})
		}
	}
	// Anti-aliasing can split one physical edge across adjacent row pairs:
	// within a ±2 row neighbourhood keep only the strongest.
	nms := make([]bool, len(edges))
	for i := range edges {
		for j := range edges {
			if i != j && !nms[i] && abs(edges[j].y-edges[i].y) <= 2 && edges[j].n > edges[i].n {
				nms[i] = true
			}
		}
	}
	var bars []Bar
	for i, t := range edges {
		if nms[i] {
			continue
		}
		for j := i + 1; j < len(edges); j++ {
			b := edges[j]
			if nms[j] {
				continue
			}
			ht := b.y - t.y
			if ht > 4*charH {
				break
			}
			if ht < charH-2 {
				continue
			}
			x0, x1 := t.x0, t.x1
			if b.x0 > x0 {
				x0 = b.x0
			}
			if b.x1 < x1 {
				x1 = b.x1
			}
			if x1-x0 < 3*charH {
				continue
			}
			// Interior = rows t.y+1 .. b.y. Depression is judged against the
			// rows just outside (skipping the blended edge rows).
			var is float64
			for k := t.y + 1; k <= b.y; k++ {
				is += e[k]
			}
			inMean := is / float64(ht)
			var flank []float64
			for k := t.y - charH/2; k <= t.y-2; k++ {
				if k >= 0 {
					flank = append(flank, e[k])
				}
			}
			for k := b.y + 2; k <= b.y+charH/2; k++ {
				if k < h {
					flank = append(flank, e[k])
				}
			}
			depressed := false
			if len(flank) > 0 {
				if ob := medianF(flank); ob > 1 && inMean/ob <= 0.65 {
					depressed = true
				}
			}
			strong := t.n >= w/2 && b.n >= w/2
			if !depressed && !strong {
				continue
			}
			bars = append(bars, Bar{Rect: imgx.Rect{X: x0, Y: t.y + 1, W: x1 - x0, H: ht}})
		}
	}
	return bars
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// edgeSpan returns the contiguous column span around the band centre where
// the vertical step between row y and its neighbour (dir=-1 above, +1 below)
// is significant — the signature of a straight horizontal bar edge — along
// with the count of significant columns.
func edgeSpan(gray []uint8, w, h, y, dir int) (int, int, int, bool) {
	y2 := y + dir
	if y2 < 0 || y2 >= h {
		return 0, 0, 0, false
	}
	const stepThr = 6
	sig := make([]bool, w)
	n := 0
	for x := 0; x < w; x++ {
		d := int(gray[y*w+x]) - int(gray[y2*w+x])
		if d < 0 {
			d = -d
		}
		if d > stepThr {
			sig[x] = true
			n++
		}
	}
	if n < w/6 {
		return 0, 0, 0, false
	}
	// Longest run of columns where the step density stays high (windowed
	// majority), centred spans preferred implicitly by length.
	best0, best1, cur0 := -1, -1, -1
	for x := 0; x <= w; x++ {
		on := x < w && sig[x]
		if on && cur0 < 0 {
			cur0 = x
		}
		if !on && cur0 >= 0 {
			// tolerate pinholes shorter than 4 px inside a run
			gap := 0
			for x+gap < w && !sig[x+gap] {
				gap++
			}
			if gap < 4 && x+gap < w {
				x += gap - 1
				continue
			}
			if runLen := x - cur0; runLen > best1-best0 {
				best0, best1 = cur0, x
			}
			cur0 = -1
		}
	}
	if best1-best0 < w/6 {
		return 0, 0, 0, false
	}
	return best0, best1, n, true
}

// fitBar estimates α and F for the bar in the YUV domain. α comes from the
// squared horizontal gradient energy ratio between in-bar and flank rows over
// the same x window (with rounding-noise correction); the blend scales
// gradient energy by (1−α)². F comes from boundary continuity at the rows
// just inside the top/bottom edges, against a background linearly
// extrapolated from the two untouched rows outside. strokeBits (band
// coordinates) excludes stroke cores from the fit.
func fitBar(rgb []byte, w, h int, r imgx.Rect, strokeBits []uint8) (alphaC [3]float64, fore [3]float64, ok bool) {
	y0, y1 := r.Y, r.Y+r.H-1
	if y0 < 2 || y1 > h-3 {
		return alphaC, fore, false
	}
	ny := r.H + 4 // rows y0-2 .. y1+2
	ybuf := make([]float64, ny*r.W*3)
	for yy := 0; yy < ny; yy++ {
		p := ((y0-2+yy)*w + r.X) * 3
		for x := 0; x < r.W; x++ {
			yv, uv, vv := rgbToYUV(float64(rgb[p]), float64(rgb[p+1]), float64(rgb[p+2]))
			q := (yy*r.W + x) * 3
			ybuf[q], ybuf[q+1], ybuf[q+2] = yv, uv, vv
			p += 3
		}
	}
	at := func(y, x, c int) float64 {
		return ybuf[((y-y0+2)*r.W+(x-r.X))*3+c]
	}
	// α from the squared-gradient energy ratio over the same x window:
	// dI = (1−α)·dBG per column, and rounding adds variance to any 2-pixel
	// difference (independent of sign, unlike |d| sums), so with s = (1−α)²:
	// M2in = s·M2out + qvar  →  s = (M2in − qvar)/M2out.
	// Strokes only ever inflate a row's energy, so the fit uses per-row
	// energies and keeps just the low tail (≤3× the first quartile) — the
	// repair mask never catches every stroke inside a scrim.
	rowE := make([][3]float64, 0, y1-y0+1)
	for y := y0; y <= y1; y++ {
		var s [3]float64
		var n int
		for x := r.X + 2; x < r.X+r.W-2; x++ {
			p := y*w + x
			if strokeBits != nil && (strokeBits[p] != 0 || strokeBits[p-1] != 0 || strokeBits[p+1] != 0) {
				continue
			}
			for c := 0; c < 3; c++ {
				d := at(y, x+1, c) - at(y, x-1, c)
				s[c] += d * d
			}
			n++
		}
		if n >= 64 {
			for c := 0; c < 3; c++ {
				s[c] /= float64(n)
			}
			rowE = append(rowE, s)
		}
	}
	var m2out [3]float64
	var nout int
	for _, y := range []int{y0 - 2, y0 - 1, y1 + 1, y1 + 2} {
		for x := r.X + 2; x < r.X+r.W-2; x++ {
			for c := 0; c < 3; c++ {
				d := at(y, x+1, c) - at(y, x-1, c)
				m2out[c] += d * d
			}
			nout++
		}
	}
	var totOut float64
	for c := 0; c < 3; c++ {
		m2out[c] /= float64(nout)
		totOut += m2out[c]
	}
	if len(rowE) < 8 || nout < 64 || totOut < 4 {
		return alphaC, fore, false // texture too flat to grade α
	}
	sort.Slice(rowE, func(i, j int) bool {
		return rowE[i][0]+rowE[i][1]+rowE[i][2] < rowE[j][0]+rowE[j][1]+rowE[j][2]
	})
	cut := rowE[len(rowE)/4][0] + rowE[len(rowE)/4][1] + rowE[len(rowE)/4][2]
	cut *= 3
	// Rounding variance of a 2-pixel RGB difference is 1/6 per channel; the
	// YUV transform scales it by each plane's squared coefficient sum.
	qvar := [3]float64{0.4469 / 6, 0.3882 / 6, 0.4319 / 6}
	var m2in [3]float64
	var nin int
	for _, e := range rowE {
		if e[0]+e[1]+e[2] <= cut {
			for c := 0; c < 3; c++ {
				m2in[c] += e[c]
			}
			nin++
		}
	}
	for c := 0; c < 3; c++ {
		m2in[c] /= float64(nin)
		if c == 0 && m2in[c] < 4*qvar[c] {
			// Luma carries no texture above the rounding floor: a flat patch
			// between textured flanks (scene content), not a scrim. Chroma is
			// exempt — scrims compress chroma hard even over real texture.
			return alphaC, fore, false
		}
		s := (m2in[c] - qvar[c]) / m2out[c]
		if s < 0.0025 {
			s = 0.0025
		}
		alphaC[c] = 1 - math.Sqrt(s)
		if alphaC[c] < 0.05 {
			alphaC[c] = 0.05
		}
	}
	if alphaC[0] > 0.75 {
		// Unmixing divides by (1−α): past this point noise amplification makes
		// luma recovery pointless, and effectively-opaque content (a flat dark
		// patch fits s≈0) is scene, not scrim. Per R3.3 the bar stays for the
		// normal fill path. Chroma may exceed 0.75 — Unmix falls back to flank
		// interpolation for those planes only.
		return alphaC, fore, false
	}
	// F per plane from boundary continuity against a linear-extrapolated
	// background: BG(y0) ≈ 2·I(y0−1) − I(y0−2). Plain row-to-row continuity
	// would import the background's vertical gradient as an F bias; the
	// extrapolation leaves only the (much smaller) curvature term.
	var fs [3][]float64
	for x := r.X; x < r.X+r.W; x++ {
		// {inside row, outside ref 1, outside ref 2}; ref rows step away
		// from the bar, d = distance of the inside row beyond ref1.
		for _, q := range [][3]int{{y0, y0 - 1, y0 - 2}, {y0 + 1, y0 - 1, y0 - 2}, {y1, y1 + 1, y1 + 2}, {y1 - 1, y1 + 1, y1 + 2}} {
			in := q[0]*w + x
			if strokeBits != nil && strokeBits[in] != 0 {
				continue
			}
			d := q[0] - q[1]
			if d < 0 {
				d = -d
			}
			for c := 0; c < 3; c++ {
				if alphaC[c] > 0.75 {
					continue // plane falls back to interpolation; F unused
				}
				bgEst := float64(1+d)*at(q[1], x, c) - float64(d)*at(q[2], x, c)
				fs[c] = append(fs[c], (at(q[0], x, c)-(1-alphaC[c])*bgEst)/alphaC[c])
			}
		}
	}
	for c := 0; c < 3; c++ {
		if alphaC[c] > 0.75 {
			fore[c] = 128 // neutral chroma; unused by Unmix
			continue
		}
		if len(fs[c]) < r.W/2 {
			return alphaC, fore, false
		}
		fore[c] = medianF(fs[c])
	}
	return alphaC, fore, true
}

// MaxResid is the fit residual above which a bar is left untouched: the
// alpha model does not describe its content. Quantization and background
// interpolation curvature put a well-behaved bar around ~13; unmodelled
// content (texture that is not a scrim) sits far higher.
const MaxResid = 16.0

// GrayOf converts an RGB frame to luma for DetectBars.
func GrayOf(rgb []byte, w, h int) []uint8 {
	g := make([]uint8, w*h)
	for p := 0; p < w*h; p++ {
		g[p] = uint8((int(rgb[p*3]) + 2*int(rgb[p*3+1]) + int(rgb[p*3+2])) / 4)
	}
	return g
}

// FrameBar is an accepted bar on one frame (band coordinates).
type FrameBar struct {
	Frame int       `json:"frame"`
	Rect  imgx.Rect `json:"rect"`
	Alpha float64   `json:"alpha"`
	Fore  [3]uint8  `json:"fore"`
	Resid float64   `json:"resid"`
}

// UnmixFrame detects bars on one RGB band frame and unmixes those whose fit
// residual is acceptable. strokeBits (the repair mask) keeps stroke cores out
// of both the fit and the unmixing — they stay for the fill stage. It returns
// the accepted bars.
func UnmixFrame(rgb []byte, w, h, charH int, strokeBits []uint8, frame int) []FrameBar {
	gray := GrayOf(rgb, w, h)
	var out []FrameBar
	for _, b := range DetectBars(gray, w, h, charH) {
		nb, ok := Unmix(rgb, w, h, b, strokeBits)
		if ok {
			out = append(out, FrameBar{Frame: frame, Rect: nb.Rect, Alpha: nb.Alpha, Fore: nb.Fore, Resid: nb.Resid})
		}
	}
	return out
}

// Unmix recovers the background inside the bar: B = (I − αF)/(1 − α), solved
// per YUV plane and converted back to RGB. Planes whose α exceeds 0.75
// (chroma of a YUV-composited scrim) would amplify noise 4×+, so they fall
// back to the flank-interpolated background — chroma is smooth in practice
// and the eye is insensitive to its texture. Stroke cores (strokeBits != 0)
// are kept for the repair mask instead. When the fit residual exceeds
// MaxResid the bar is abandoned and rgb is left unchanged.
func Unmix(rgb []byte, w, h int, b Bar, strokeBits []uint8) (Bar, bool) {
	alphaC, fore, ok := fitBar(rgb, w, h, b.Rect, strokeBits)
	b.Alpha = alphaC[0]
	y0, y1 := b.Rect.Y, b.Rect.Y+b.Rect.H-1
	ny := b.Rect.H + 2 // rows y0-1 .. y1+1
	ybuf := make([]float64, ny*b.Rect.W*3)
	for yy := 0; yy < ny; yy++ {
		p := ((y0-1+yy)*w + b.Rect.X) * 3
		for x := 0; x < b.Rect.W; x++ {
			yv, uv, vv := rgbToYUV(float64(rgb[p]), float64(rgb[p+1]), float64(rgb[p+2]))
			q := (yy*b.Rect.W + x) * 3
			ybuf[q], ybuf[q+1], ybuf[q+2] = yv, uv, vv
			p += 3
		}
	}
	at := func(y, x, c int) float64 {
		return ybuf[((y-y0+1)*b.Rect.W+(x-b.Rect.X))*3+c]
	}
	bg := func(x, y, c int) float64 {
		top := at(y0-1, x, c)
		bot := at(y1+1, x, c)
		t := float64(y-y0+1) / float64(b.Rect.H+1)
		return top*(1-t) + bot*t
	}
	if ok {
		// Mean absolute luma residual of the fit against the interpolated
		// background, computed per row and trimmed to the low tail: strokes
		// the repair mask missed inflate their row's residual, exactly as in
		// fitBar. Luma only — chroma is smooth and fits almost anything,
		// which would dilute a genuine luma misfit.
		rowR := make([]float64, 0, y1-y0+1)
		for y := y0; y <= y1; y++ {
			var s float64
			var n int
			for x := b.Rect.X + 2; x < b.Rect.X+b.Rect.W-2; x++ {
				p := y*w + x
				if strokeBits != nil && strokeBits[p] != 0 {
					continue
				}
				s += math.Abs(at(y, x, 0) - (alphaC[0]*fore[0] + (1-alphaC[0])*bg(x, y, 0)))
				n++
			}
			if n >= 64 {
				rowR = append(rowR, s/float64(n))
			}
		}
		if len(rowR) < 8 {
			return b, false
		}
		sort.Float64s(rowR)
		cut := 3 * rowR[len(rowR)/4]
		var s float64
		var n int
		for _, r := range rowR {
			if r <= cut {
				s += r
				n++
			}
		}
		b.Resid = s / float64(n)
	}
	fr, fg, fb := yuvToRGB(fore[0], fore[1], fore[2])
	b.Fore[0] = clamp8(fr)
	b.Fore[1] = clamp8(fg)
	b.Fore[2] = clamp8(fb)
	if !ok || b.Resid > MaxResid {
		return b, false
	}
	for y := y0; y <= y1; y++ {
		for x := b.Rect.X; x < b.Rect.X+b.Rect.W; x++ {
			p := y*w + x
			if strokeBits != nil && strokeBits[p] != 0 {
				continue
			}
			var rec [3]float64
			for c := 0; c < 3; c++ {
				if alphaC[c] > 0.75 {
					rec[c] = bg(x, y, c)
				} else {
					rec[c] = (at(y, x, c) - alphaC[c]*fore[c]) / (1 - alphaC[c])
				}
			}
			r8, g8, b8 := yuvToRGB(rec[0], rec[1], rec[2])
			rgb[p*3] = clamp8(r8)
			rgb[p*3+1] = clamp8(g8)
			rgb[p*3+2] = clamp8(b8)
		}
	}
	return b, true
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
