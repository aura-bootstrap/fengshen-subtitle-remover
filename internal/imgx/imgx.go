// Package imgx provides minimal grayscale image primitives, rectangle
// geometry and separable morphology used by the detector and engines.
package imgx

import "sort"

type Rect struct{ X, Y, W, H int }

func (r Rect) Empty() bool { return r.W <= 0 || r.H <= 0 }
func (r Rect) Right() int  { return r.X + r.W }
func (r Rect) Bottom() int { return r.Y + r.H }
func (r Rect) Area() int   { return r.W * r.H }

func (r Rect) Expand(p int) Rect { return Rect{r.X - p, r.Y - p, r.W + 2*p, r.H + 2*p} }

func (r Rect) Union(o Rect) Rect {
	if r.Empty() {
		return o
	}
	if o.Empty() {
		return r
	}
	x0, y0 := min(r.X, o.X), min(r.Y, o.Y)
	x1, y1 := max(r.Right(), o.Right()), max(r.Bottom(), o.Bottom())
	return Rect{x0, y0, x1 - x0, y1 - y0}
}

func (r Rect) Intersect(o Rect) Rect {
	x0, y0 := max(r.X, o.X), max(r.Y, o.Y)
	x1, y1 := min(r.Right(), o.Right()), min(r.Bottom(), o.Bottom())
	if x1 <= x0 || y1 <= y0 {
		return Rect{}
	}
	return Rect{x0, y0, x1 - x0, y1 - y0}
}

func (r Rect) Clamp(w, h int) Rect {
	x0, y0 := max(r.X, 0), max(r.Y, 0)
	x1, y1 := min(r.Right(), w), min(r.Bottom(), h)
	if x1 <= x0 || y1 <= y0 {
		return Rect{}
	}
	return Rect{x0, y0, x1 - x0, y1 - y0}
}

// sliding applies a sliding-window min (or max) over src[0:n] writing dst[0:n].
// Window size is k (odd, centered; k=1 is identity). O(n) via monotonic deque.
func sliding(src, dst []uint8, n, k int, maxMode bool) {
	if k <= 1 || n == 0 {
		copy(dst[:n], src[:n])
		return
	}
	half := k / 2
	dq := make([]int32, 0, n+1)
	head := 0
	for i := 0; i < n+half; i++ {
		if i < n {
			for len(dq) > head {
				last := src[dq[len(dq)-1]]
				if maxMode && last <= src[i] || !maxMode && last >= src[i] {
					dq = dq[:len(dq)-1]
				} else {
					break
				}
			}
			dq = append(dq, int32(i))
		}
		o := i - half
		if o < 0 {
			continue
		}
		lo := max(o-half, 0)
		for int(dq[head]) < lo {
			head++
		}
		dst[o] = src[dq[head]]
	}
}

// MorphRect applies a separable rectangular erosion (maxMode=false) or
// dilation (maxMode=true) to a w*h grayscale buffer.
func MorphRect(src, dst []uint8, w, h, kw, kh int, maxMode bool) {
	tmp := make([]uint8, w*h)
	row := make([]uint8, w)
	for y := 0; y < h; y++ {
		sliding(src[y*w:(y+1)*w], row, w, kw, maxMode)
		copy(tmp[y*w:(y+1)*w], row)
	}
	col := make([]uint8, h)
	out := make([]uint8, h)
	for x := 0; x < w; x++ {
		for y := 0; y < h; y++ {
			col[y] = tmp[y*w+x]
		}
		sliding(col, out, h, kh, maxMode)
		for y := 0; y < h; y++ {
			dst[y*w+x] = out[y]
		}
	}
}

// MorphBin applies erosion/dilation to a 0/1 buffer.
func MorphBin(src, dst []uint8, w, h, kw, kh int, maxMode bool) {
	MorphRect(src, dst, w, h, kw, kh, maxMode)
}

func boxLine(src, dst []uint8, n, r int) {
	if n == 0 {
		return
	}
	if r < 1 {
		copy(dst[:n], src[:n])
		return
	}
	sum := 0
	lo, hi := 0, min(r, n-1)
	for i := lo; i <= hi; i++ {
		sum += int(src[i])
	}
	for i := 0; i < n; i++ {
		dst[i] = uint8(sum / (hi - lo + 1))
		add, sub := i+r+1, i-r
		if add < n {
			sum += int(src[add])
			hi = add
		}
		if sub >= 0 {
			sum -= int(src[sub])
			lo = sub + 1
		}
		if lo > hi {
			lo, hi = min(i+1, n-1), min(i+r, n-1)
		}
	}
}

// BoxBlur applies a separable box blur with radius r.
func BoxBlur(src []uint8, w, h, r int, dst []uint8) {
	tmp := make([]uint8, w*h)
	row := make([]uint8, w)
	for y := 0; y < h; y++ {
		boxLine(src[y*w:(y+1)*w], row, w, r)
		copy(tmp[y*w:(y+1)*w], row)
	}
	col := make([]uint8, h)
	out := make([]uint8, h)
	for x := 0; x < w; x++ {
		for y := 0; y < h; y++ {
			col[y] = tmp[y*w+x]
		}
		boxLine(col, out, h, r)
		for y := 0; y < h; y++ {
			dst[y*w+x] = out[y]
		}
	}
}

// Otsu returns the Otsu threshold for a 256-bin histogram.
func Otsu(hist *[256]int, total int) int {
	if total == 0 {
		return 0
	}
	var sumAll float64
	for i := 0; i < 256; i++ {
		sumAll += float64(i) * float64(hist[i])
	}
	var wB, sumB float64
	best, th := -1.0, 0
	for t := 0; t < 256; t++ {
		wB += float64(hist[t])
		if wB == 0 {
			continue
		}
		wF := float64(total) - wB
		if wF == 0 {
			break
		}
		sumB += float64(t) * float64(hist[t])
		mB := sumB / wB
		mF := (sumAll - sumB) / wF
		v := wB * wF * (mB - mF) * (mB - mF)
		if v > best {
			best, th = v, t
		}
	}
	return th
}

// LabelBinary labels 8-connected (or 4-connected) components of a 0/1 buffer.
// Returned labels are 1-based; 0 marks background.
func LabelBinary(src []uint8, w, h int, eight bool) ([]int32, int) {
	labels := make([]int32, w*h)
	stack := make([]int32, 0, 4096)
	var next int32
	for i := 0; i < w*h; i++ {
		if src[i] == 0 || labels[i] != 0 {
			continue
		}
		next++
		labels[i] = next
		stack = append(stack[:0], int32(i))
		for len(stack) > 0 {
			p := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			px, py := int(p)%w, int(p)/w
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					if dx == 0 && dy == 0 {
						continue
					}
					if !eight && dx != 0 && dy != 0 {
						continue
					}
					nx, ny := px+dx, py+dy
					if nx < 0 || ny < 0 || nx >= w || ny >= h {
						continue
					}
					q := ny*w + nx
					if src[q] != 0 && labels[q] == 0 {
						labels[q] = next
						stack = append(stack, int32(q))
					}
				}
			}
		}
	}
	return labels, int(next)
}

// SortRects sorts rectangles by (Y, X) in place.
func SortRects(rs []Rect) {
	sort.Slice(rs, func(i, j int) bool {
		if rs[i].Y != rs[j].Y {
			return rs[i].Y < rs[j].Y
		}
		return rs[i].X < rs[j].X
	})
}
