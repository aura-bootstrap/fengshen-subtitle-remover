// Package motion provides sparse Lucas-Kanade tracking and RANSAC
// homography estimation, used to transfer real pixels between frames of the
// same shot (roadmap L2).
package motion

import "math"

type Pt struct{ X, Y float64 }

// Luma fills dst (w*h) with (r + 2g + b) / 4 from an rgb24 buffer.
func Luma(rgb []byte, dst []uint8) {
	for p := 0; p < len(dst); p++ {
		q := p * 3
		dst[p] = uint8((int(rgb[q]) + 2*int(rgb[q+1]) + int(rgb[q+2])) / 4)
	}
}

// Gradients fills gx, gy with central differences (same layout as gray).
func Gradients(gray []uint8, gx, gy []int16, w, h int) {
	for y := 0; y < h; y++ {
		row := y * w
		up := row - w
		dn := row + w
		if y == 0 {
			up = row
		}
		if y == h-1 {
			dn = row
		}
		for x := 0; x < w; x++ {
			l, r := x-1, x+1
			if x == 0 {
				l = 0
			}
			if x == w-1 {
				r = x
			}
			gx[row+x] = int16(int(gray[row+r]) - int(gray[row+l]))
			gy[row+x] = int16(int(gray[dn+x]) - int(gray[up+x]))
		}
	}
}

// Corners returns Shi-Tomasi corners picked best-per-grid-cell, skipping
// pixels set in bits (the subtitle itself must not be tracked). minEig is the
// corner-strength floor; scale it down when scanning a downsampled frame.
func Corners(gray []uint8, w, h int, bits []uint8, grid, maxPts int, minEig float64) []Pt {
	if grid < 4 {
		grid = 4
	}
	var pts []Pt
	for cy := 0; cy < h; cy += grid {
		for cx := 0; cx < w; cx += grid {
			bestX, bestY, best := -1, -1, 0.0
			for y := cy + 3; y < min(cy+grid, h-3); y++ {
				for x := cx + 3; x < min(cx+grid, w-3); x++ {
					p := y*w + x
					if bits != nil && bits[p] != 0 {
						continue
					}
					var a, b, c int
					for dy := -1; dy <= 1; dy++ {
						for dx := -1; dx <= 1; dx++ {
							q := p + dy*w + dx
							gx := int(gray[q+1]) - int(gray[q-1])
							gy := int(gray[q+w]) - int(gray[q-w])
							a += gx * gx
							b += gx * gy
							c += gy * gy
						}
					}
					tr := float64(a + c)
					d := math.Sqrt(float64((a-c)*(a-c) + 4*b*b))
					eig := (tr - d) / 2
					if eig > best {
						best, bestX, bestY = eig, x, y
					}
				}
			}
			if best >= minEig {
				pts = append(pts, Pt{float64(bestX), float64(bestY)})
				if maxPts > 0 && len(pts) >= maxPts {
					return pts
				}
			}
		}
	}
	return pts
}

// DownsampleGray box-averages gray by an integer factor f, returning the
// downscaled buffer (dims ceil(w/f) x ceil(h/f)).
func DownsampleGray(gray []uint8, w, h, f int) ([]uint8, int, int) {
	dw, dh := (w+f-1)/f, (h+f-1)/f
	dst := make([]uint8, dw*dh)
	for y := 0; y < dh; y++ {
		for x := 0; x < dw; x++ {
			var sum, cnt int
			for sy := y * f; sy < min((y+1)*f, h); sy++ {
				row := sy * w
				for sx := x * f; sx < min((x+1)*f, w); sx++ {
					sum += int(gray[row+sx])
					cnt++
				}
			}
			dst[y*dw+x] = uint8((sum + cnt/2) / cnt)
		}
	}
	return dst, dw, dh
}

// DownsampleBits marks a downscaled cell set when any source pixel in its
// block is set.
func DownsampleBits(bits []uint8, w, h, f int) ([]uint8, int, int) {
	dw, dh := (w+f-1)/f, (h+f-1)/f
	dst := make([]uint8, dw*dh)
	for y := 0; y < dh; y++ {
		for x := 0; x < dw; x++ {
		block:
			for sy := y * f; sy < min((y+1)*f, h); sy++ {
				row := sy * w
				for sx := x * f; sx < min((x+1)*f, w); sx++ {
					if bits[row+sx] != 0 {
						dst[y*dw+x] = 1
						break block
					}
				}
			}
		}
	}
	return dst, dw, dh
}

func bilinear(gray []uint8, w, h int, x, y float64) (float64, bool) {
	if x < 0 || y < 0 || x >= float64(w-1) || y >= float64(h-1) {
		return 0, false
	}
	x0, y0 := int(x), int(y)
	fx, fy := x-float64(x0), y-float64(y0)
	p := y0*w + x0
	a := float64(gray[p])
	b := float64(gray[p+1])
	c := float64(gray[p+w])
	d := float64(gray[p+w+1])
	return a*(1-fx)*(1-fy) + b*fx*(1-fy) + c*(1-fx)*fy + d*fx*fy, true
}

// bilinearGrad samples the precomputed gradient fields at a fractional
// position; the gradient lives on the same pixel grid as gray.
func bilinearGrad(gx, gy []int16, w, h int, x, y float64) (float64, float64, bool) {
	if x < 0 || y < 0 || x >= float64(w-1) || y >= float64(h-1) {
		return 0, 0, false
	}
	x0, y0 := int(x), int(y)
	fx, fy := x-float64(x0), y-float64(y0)
	p := y0*w + x0
	interp := func(g []int16) float64 {
		a := float64(g[p])
		b := float64(g[p+1])
		c := float64(g[p+w])
		d := float64(g[p+w+1])
		return a*(1-fx)*(1-fy) + b*fx*(1-fy) + c*(1-fx)*fy + d*fx*fy
	}
	return interp(gx), interp(gy), true
}

// TrackLK tracks pts from prev to cur (forward-additive, single level).
// Returns the new positions and per-point convergence flags.
func TrackLK(prev, cur []uint8, w, h int, pts []Pt, win, iters int) ([]Pt, []bool) {
	return trackCore(prev, cur, w, h, pts, nil, win, iters)
}

// TrackLKSeeded is TrackLK with per-point initial guesses instead of
// starting from the source position; used to verify a coarse motion model
// (e.g. a quarter-resolution homography) at full resolution.
func TrackLKSeeded(prev, cur []uint8, w, h int, pts, seeds []Pt, win, iters int) ([]Pt, []bool) {
	return trackCore(prev, cur, w, h, pts, seeds, win, iters)
}

// trackCore runs forward-additive LK: each iteration re-samples cur's
// intensity and gradient at the current fractional position and solves the
// 2x2 normal equations fresh. Recomputing the gradient matrix per iteration
// (instead of freezing it at the seed) matters because the frozen variant's
// fixed point is biased away from the SSD minimum whenever the seed sits off
// the truth — measured ~1.2px mean residual from a 3px seed on coarse
// texture, which then survives the homography fit as a systematic
// underestimate of the motion.
func trackCore(prev, cur []uint8, w, h int, pts, seeds []Pt, win, iters int) ([]Pt, []bool) {
	gx := make([]int16, w*h)
	gy := make([]int16, w*h)
	Gradients(cur, gx, gy, w, h)
	out := make([]Pt, len(pts))
	ok := make([]bool, len(pts))
	for i, p := range pts {
		ux, uy := p.X, p.Y
		if seeds != nil {
			ux, uy = seeds[i].X, seeds[i].Y
		}
		px, py := int(p.X), int(p.Y)
		if px < win+1 || py < win+1 || px >= w-win-1 || py >= h-win-1 {
			continue
		}
		if ux < float64(win+1) || uy < float64(win+1) || ux >= float64(w-win-1) || uy >= float64(h-win-1) {
			continue
		}
		good := false
		for it := 0; it < iters; it++ {
			// Symmetric sampling: splitting the fractional offset between
			// template and match equalizes bilinear blur on both sides, so
			// the error minimum sits at the true shift instead of trading
			// alignment against interpolation sharpness (~0.45px bias).
			hx := (ux - math.Floor(ux)) / 2
			hy := (uy - math.Floor(uy)) / 2
			ax, ay := p.X-hx, p.Y-hy
			cx, cy := ux-hx, uy-hy
			var ga, gbc, gc float64
			var bx, by float64
			for dy := -win; dy <= win; dy++ {
				for dx := -win; dx <= win; dx++ {
					a, b, gok := bilinearGrad(gx, gy, w, h, cx+float64(dx), cy+float64(dy))
					if !gok {
						continue
					}
					iv, inb := bilinear(cur, w, h, cx+float64(dx), cy+float64(dy))
					if !inb {
						continue
					}
					tv, tnb := bilinear(prev, w, h, ax+float64(dx), ay+float64(dy))
					if !tnb {
						continue
					}
					ga += a * a
					gbc += a * b
					gc += b * b
					e := tv - iv
					bx += a * e
					by += b * e
				}
			}
			det := ga*gc - gbc*gbc
			if det < 800 {
				break
			}
			dx := (gc*bx - gbc*by) / det
			dy := (ga*by - gbc*bx) / det
			ux += dx
			uy += dy
			if math.Abs(dx) < 0.03 && math.Abs(dy) < 0.03 {
				good = true
				break
			}
			if ux < 1 || uy < 1 || ux >= float64(w-2) || uy >= float64(h-2) {
				break
			}
		}
		if !good || ux < 1 || uy < 1 || ux >= float64(w-2) || uy >= float64(h-2) {
			continue
		}
		out[i] = Pt{ux, uy}
		ok[i] = true
	}
	return out, ok
}

// FitHomography returns an H mapping src→dst via RANSAC plus a least-squares
// refit on the inliers. fit is false when the estimate is unstable.
func FitHomography(src, dst []Pt, ok []bool, tol float64) (h [9]float64, inliers int, fit bool) {
	var s, d []Pt
	for i := range src {
		if ok[i] {
			s = append(s, src[i])
			d = append(d, dst[i])
		}
	}
	n := len(s)
	if n < 12 {
		return h, 0, false
	}
	ns, Ts := normalize(s)
	nd, Td := normalize(d)
	// RANSAC compares in the normalized space; scale tol from pixels so both
	// gates mean the same distance.
	tolN := tol * (Ts[0] + Td[0]) / 2

	bestInl := 0
	var bestH [9]float64
	rnd := uint32(12345)
	for it := 0; it < 200; it++ {
		rnd = rnd*1664525 + 1013904223
		i0 := int(rnd>>16) % n
		rnd = rnd*1664525 + 1013904223
		i1 := int(rnd>>16) % n
		rnd = rnd*1664525 + 1013904223
		i2 := int(rnd>>16) % n
		rnd = rnd*1664525 + 1013904223
		i3 := int(rnd>>16) % n
		if i0 == i1 || i0 == i2 || i0 == i3 || i1 == i2 || i1 == i3 || i2 == i3 {
			continue
		}
		var cand [9]float64
		if !solveDLT(ns, nd, i0, i1, i2, i3, &cand) {
			continue
		}
		cnt := 0
		for i := 0; i < n; i++ {
			fx, fy := applyH(cand, ns[i].X, ns[i].Y)
			if math.Hypot(fx-nd[i].X, fy-nd[i].Y) < tolN {
				cnt++
			}
		}
		if cnt > bestInl {
			bestInl = cnt
			bestH = cand
		}
	}
	if bestInl < 10 || bestInl*4 < n {
		return h, 0, false
	}
	var is, id []Pt
	for i := 0; i < n; i++ {
		fx, fy := applyH(bestH, ns[i].X, ns[i].Y)
		if math.Hypot(fx-nd[i].X, fy-nd[i].Y) < tolN {
			is = append(is, ns[i])
			id = append(id, nd[i])
		}
	}
	if !solveLS(is, id, &bestH) {
		return h, 0, false
	}
	h = mul3(mul3(inv3(Td), bestH), Ts)
	res := 0.0
	for i := 0; i < n; i++ {
		fx, fy := applyH(h, s[i].X, s[i].Y)
		res += math.Hypot(fx-d[i].X, fy-d[i].Y)
	}
	res /= float64(n)
	if res > tol {
		return h, bestInl, false
	}
	return h, bestInl, true
}

func normalize(p []Pt) ([]Pt, [9]float64) {
	var cx, cy float64
	for _, q := range p {
		cx += q.X
		cy += q.Y
	}
	cx /= float64(len(p))
	cy /= float64(len(p))
	var m float64
	for _, q := range p {
		m += math.Hypot(q.X-cx, q.Y-cy)
	}
	m /= float64(len(p))
	if m < 1e-9 {
		m = 1
	}
	s := math.Sqrt(2) / m
	out := make([]Pt, len(p))
	for i, q := range p {
		out[i] = Pt{(q.X - cx) * s, (q.Y - cy) * s}
	}
	return out, [9]float64{s, 0, -cx * s, 0, s, -cy * s, 0, 0, 1}
}

// solveDLT computes H (with h22=1) from 4 correspondences via an 8x8 linear
// system; degenerate configurations fail.
func solveDLT(s, d []Pt, i0, i1, i2, i3 int, out *[9]float64) bool {
	idx := [4]int{i0, i1, i2, i3}
	var A [8][9]float64
	for r := 0; r < 4; r++ {
		x, y := s[idx[r]].X, s[idx[r]].Y
		u, v := d[idx[r]].X, d[idx[r]].Y
		A[2*r] = [9]float64{x, y, 1, 0, 0, 0, -u * x, -u * y, u}
		A[2*r+1] = [9]float64{0, 0, 0, x, y, 1, -v * x, -v * y, v}
	}
	return gauss8(&A, out)
}

// solveLS solves the stacked DLT system for all pairs by normal equations.
func solveLS(s, d []Pt, out *[9]float64) bool {
	var ata [8][9]float64
	for i := 0; i < len(s); i++ {
		x, y := s[i].X, s[i].Y
		u, v := d[i].X, d[i].Y
		rows := [2][9]float64{
			{x, y, 1, 0, 0, 0, -u * x, -u * y, u},
			{0, 0, 0, x, y, 1, -v * x, -v * y, v},
		}
		for r := 0; r < 2; r++ {
			for a := 0; a < 8; a++ {
				for b := 0; b < 9; b++ {
					ata[a][b] += rows[r][a] * rows[r][b]
				}
			}
		}
	}
	return gauss8(&ata, out)
}

// gauss8 solves an 8x9 augmented system by Gaussian elimination with partial
// pivoting.
func gauss8(A *[8][9]float64, out *[9]float64) bool {
	for col := 0; col < 8; col++ {
		piv := col
		for r := col + 1; r < 8; r++ {
			if math.Abs(A[r][col]) > math.Abs(A[piv][col]) {
				piv = r
			}
		}
		if math.Abs(A[piv][col]) < 1e-12 {
			return false
		}
		A[col], A[piv] = A[piv], A[col]
		for r := 0; r < 8; r++ {
			if r == col {
				continue
			}
			f := A[r][col] / A[col][col]
			for c := col; c < 9; c++ {
				A[r][c] -= f * A[col][c]
			}
		}
	}
	for i := 0; i < 8; i++ {
		out[i] = A[i][8] / A[i][i]
	}
	out[8] = 1
	return true
}

func applyH(h [9]float64, x, y float64) (float64, float64) {
	d := h[6]*x + h[7]*y + h[8]
	return (h[0]*x + h[1]*y + h[2]) / d, (h[3]*x + h[4]*y + h[5]) / d
}

// Apply maps a point through H.
func Apply(h [9]float64, x, y float64) (float64, float64) {
	return applyH(h, x, y)
}

func mul3(a, b [9]float64) [9]float64 {
	var out [9]float64
	for r := 0; r < 3; r++ {
		for c := 0; c < 3; c++ {
			var v float64
			for k := 0; k < 3; k++ {
				v += a[r*3+k] * b[k*3+c]
			}
			out[r*3+c] = v
		}
	}
	return out
}

func inv3(m [9]float64) [9]float64 {
	a, b, c := m[0], m[1], m[2]
	d, e, f := m[3], m[4], m[5]
	g, h, i := m[6], m[7], m[8]
	// Adjugate entries (row-major).
	A := e*i - f*h
	B := c*h - b*i
	C := b*f - c*e
	D := f*g - d*i
	E := a*i - c*g
	F := c*d - a*f
	G := d*h - e*g
	H := b*g - a*h
	I := a*e - b*d
	det := a*A + b*D + c*G
	if det < 1e-12 && det > -1e-12 {
		return [9]float64{1, 0, 0, 0, 1, 0, 0, 0, 1}
	}
	id := 1 / det
	return [9]float64{A * id, B * id, C * id, D * id, E * id, F * id, G * id, H * id, I * id}
}

// Invert returns the inverse of H (identity when singular).
func Invert(m [9]float64) [9]float64 {
	return inv3(m)
}

// Compose returns a·b.
func Compose(a, b [9]float64) [9]float64 {
	return mul3(a, b)
}

// Identity is the 3x3 identity homography.
var Identity = [9]float64{1, 0, 0, 0, 1, 0, 0, 0, 1}

// BilinearRGB samples an rgb24 frame at a fractional position.
func BilinearRGB(rgb []byte, w, h int, x, y float64) (r, g, b uint8, inb bool) {
	if x < 0 || y < 0 || x >= float64(w-1) || y >= float64(h-1) {
		return 0, 0, 0, false
	}
	x0, y0 := int(x), int(y)
	fx, fy := x-float64(x0), y-float64(y0)
	p := (y0*w + x0) * 3
	q := (y0*w + x0 + w) * 3
	w00, w10, w01, w11 := (1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy
	var rc, gc, bc float64
	for ch := 0; ch < 3; ch++ {
		v := float64(rgb[p+ch])*w00 + float64(rgb[p+3+ch])*w10 +
			float64(rgb[q+ch])*w01 + float64(rgb[q+3+ch])*w11
		switch ch {
		case 0:
			rc = v
		case 1:
			gc = v
		default:
			bc = v
		}
	}
	return uint8(rc + 0.5), uint8(gc + 0.5), uint8(bc + 0.5), true
}
